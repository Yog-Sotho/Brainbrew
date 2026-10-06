"""
tests/test_netguard.py

The connect-time SSRF guard: metadata addresses are refused after DNS
resolution and on redirects, the checked address is the one connected to, and
ordinary endpoints keep working.
"""
from __future__ import annotations

import asyncio
import socket
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer

import httpcore2
import httpx2
import pytest
from openai import APIConnectionError

from engine import netguard
from engine.client import ChatClient, EndpointSettings
from engine.netguard import BlockedAddressError, GuardedBackend, guard_client, is_metadata_host

PROXY_VARS = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")


@pytest.fixture(autouse=True)
def no_env_proxy(monkeypatch):
    # A proxy resolves the target itself, so these tests talk to hosts directly.
    for var in PROXY_VARS:
        monkeypatch.delenv(var, raising=False)


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/redirect":
            self.send_response(302)
            self.send_header("Location", "http://169.254.169.254/latest/meta-data/")
            self.end_headers()
            return
        body = b"ok"
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture
def server() -> Iterator[int]:
    httpd = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield httpd.server_address[1]
    httpd.shutdown()
    httpd.server_close()


def _fake_dns(monkeypatch, answers: dict[str, list[str]]) -> None:
    async def getaddrinfo(host, port, *, type=0, **_):
        if host not in answers:
            raise socket.gaierror(f"unknown host {host}")
        return [(socket.AF_INET6 if ":" in a else socket.AF_INET, type, 6, "", (a, port)) for a in answers[host]]

    monkeypatch.setattr(netguard.anyio, "getaddrinfo", getaddrinfo)


async def _get(url: str, **kwargs) -> httpx2.Response:
    async with guard_client(httpx2.AsyncClient(**kwargs)) as client:
        return await client.get(url)


class TestPredicates:

    @pytest.mark.parametrize(("host", "blocked"), [
        ("169.254.169.254", True), ("2852039166", True), ("0xa9fea9fe", True),
        ("::ffff:169.254.169.254", True), ("[fe80::1]", True), ("fd00:ec2::254", True),
        ("Metadata.Google.Internal.", True), ("instance-data", True),
        ("127.0.0.1", False), ("10.0.0.5", False), ("::1", False), ("api.openai.com", False), ("vllm", False),
    ])
    def test_is_metadata_host(self, host, blocked):
        assert is_metadata_host(host) is blocked


class TestGuardedClient:

    def test_ordinary_hosts_still_work(self, server, monkeypatch):
        _fake_dns(monkeypatch, {"vllm.internal": ["127.0.0.1"]})
        assert asyncio.run(_get(f"http://127.0.0.1:{server}/")).text == "ok"
        assert asyncio.run(_get(f"http://vllm.internal:{server}/")).text == "ok"

    def test_name_resolving_to_metadata_is_refused(self, monkeypatch):
        _fake_dns(monkeypatch, {"innocent.example": ["169.254.169.254"]})
        with pytest.raises(httpx2.ConnectError, match="link-local or cloud metadata"):
            asyncio.run(_get("http://innocent.example/v1"))

    def test_any_metadata_answer_refuses_the_host(self, monkeypatch):
        _fake_dns(monkeypatch, {"mixed.example": ["127.0.0.1", "::ffff:169.254.169.254"]})
        with pytest.raises(httpx2.ConnectError, match="metadata"):
            asyncio.run(_get("http://mixed.example/v1"))

    def test_redirect_to_metadata_is_refused(self, server):
        with pytest.raises(httpx2.ConnectError, match="metadata"):
            asyncio.run(_get(f"http://127.0.0.1:{server}/redirect", follow_redirects=True))

    def test_connects_to_the_checked_address(self, monkeypatch):
        # DNS rebinding: whatever the name resolves to later, the connection goes
        # to the address that was checked, as an IP literal.
        _fake_dns(monkeypatch, {"rebind.example": ["127.0.0.1"]})
        seen: list[str] = []

        class Recorder(httpcore2.AsyncNetworkBackend):
            async def connect_tcp(self, host, port, **kwargs):
                seen.append(host)
                raise httpcore2.ConnectError("recorded")

        backend = GuardedBackend(Recorder())
        with pytest.raises(httpcore2.ConnectError, match="recorded"):
            asyncio.run(backend.connect_tcp("rebind.example", 80))
        assert seen == ["127.0.0.1"]

    def test_falls_back_across_resolved_addresses(self, server, monkeypatch):
        _fake_dns(monkeypatch, {"two.example": ["127.0.0.2", "127.0.0.1"]})
        tried: list[str] = []
        inner = httpcore2.AnyIOBackend()

        class Flaky(httpcore2.AsyncNetworkBackend):
            async def connect_tcp(self, host, port, **kwargs):
                tried.append(host)
                if host == "127.0.0.2":
                    raise httpcore2.ConnectError("refused")
                return await inner.connect_tcp(host, port, **kwargs)

        async def connect():
            stream = await GuardedBackend(Flaky()).connect_tcp("two.example", server)
            await stream.aclose()

        asyncio.run(connect())
        assert tried == ["127.0.0.2", "127.0.0.1"]

    def test_every_pool_is_guarded_including_proxies(self, monkeypatch):
        monkeypatch.setenv("HTTPS_PROXY", "http://proxy.internal:3128")
        client = guard_client(httpx2.AsyncClient())
        pools = [t._pool for t in [client._transport, *client._mounts.values()] if t is not None]
        assert len(pools) >= 2
        assert all(isinstance(p._network_backend, GuardedBackend) for p in pools)

    def test_mock_transports_are_left_alone(self):
        client = httpx2.AsyncClient(transport=httpx2.MockTransport(lambda r: httpx2.Response(200)))
        assert guard_client(client) is client

    def test_blocked_error_is_a_connect_error(self):
        assert issubclass(BlockedAddressError, httpcore2.ConnectError)


class TestChatClientIsGuarded:

    def test_default_client_refuses_metadata_after_dns(self, monkeypatch):
        _fake_dns(monkeypatch, {"model.example": ["169.254.169.254"]})
        settings = EndpointSettings(model="m", base_url="http://model.example/v1", max_retries=0, timeout_s=5)

        async def call():
            async with ChatClient(settings) as client:
                await client.chat([{"role": "user", "content": "hi"}])

        with pytest.raises(APIConnectionError) as info:
            asyncio.run(call())
        chain, exc = [], info.value
        while exc is not None:
            chain.append(exc)
            exc = exc.__cause__ or exc.__context__
        assert any("metadata" in str(e) for e in chain), chain
