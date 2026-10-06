"""
Keep model requests away from cloud metadata services (SSRF).

`config.check_base_url` rejects metadata addresses written in the URL. That
cannot catch a host name that *resolves* to one, a DNS answer that changes
between the check and the request (rebinding), or a redirect. This module
closes those gaps at connect time: `GuardedBackend` resolves the host itself,
refuses if any address is a metadata or link-local address, and then connects
to the address it checked, so the address that was checked is the one used.
TLS still verifies the original host name (httpcore passes it as SNI).

Behind an HTTP proxy the proxy resolves the target, so only the URL check applies.
"""
from __future__ import annotations

import ipaddress
import socket
from collections.abc import Iterable
from typing import Any

import anyio
import httpcore2
import httpx2
import structlog

logger = structlog.get_logger(__name__)

# Host names of cloud instance-metadata services.
METADATA_HOSTS = frozenset({"metadata.google.internal", "metadata.goog", "metadata.azure.com", "instance-data"})
_AWS_IPV6_METADATA = ipaddress.ip_address("fd00:ec2::254")

IPAddress = ipaddress.IPv4Address | ipaddress.IPv6Address


def parse_ip(host: str) -> IPAddress | None:
    """*host* as an IP address, accepting the legacy IPv4 spellings resolvers take
    ("2852039166", "0xa9fea9fe", "169.254.43518"); None for a host name."""
    host = host.strip("[]")
    try:
        return ipaddress.ip_address(host)
    except ValueError:
        pass
    try:
        return ipaddress.IPv4Address(socket.inet_aton(host))
    except OSError:
        return None


def is_metadata_address(ip: IPAddress) -> bool:
    """Link-local (169.254.0.0/16, fe80::/10, incl. 169.254.169.254) or AWS's fd00:ec2::254."""
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
        ip = ip.ipv4_mapped
    return ip.is_link_local or ip == _AWS_IPV6_METADATA


def is_metadata_host(host: str) -> bool:
    """True if *host*, as written, names a metadata service or a link-local address."""
    host = host.lower().rstrip(".")
    if host in METADATA_HOSTS:
        return True
    ip = parse_ip(host)
    return ip is not None and is_metadata_address(ip)


class BlockedAddressError(httpcore2.ConnectError):
    """The host resolved to a metadata or link-local address."""


class GuardedBackend(httpcore2.AsyncNetworkBackend):
    """A network backend that refuses metadata addresses after DNS resolution."""

    def __init__(self, inner: httpcore2.AsyncNetworkBackend | None = None) -> None:
        self._inner = inner or httpcore2.AnyIOBackend()

    async def _resolve(self, host: str, port: int) -> list[str]:
        if is_metadata_host(host):
            return self._refuse(host, host)
        ip = parse_ip(host)
        if ip is not None:
            return [str(ip)]
        infos = await anyio.getaddrinfo(host, port, type=socket.SOCK_STREAM)
        addresses = list(dict.fromkeys(str(info[4][0]) for info in infos))
        for address in addresses:
            parsed = parse_ip(address.split("%", 1)[0])  # drop an IPv6 zone id
            if parsed is None or is_metadata_address(parsed):
                return self._refuse(host, address)
        return addresses

    @staticmethod
    def _refuse(host: str, address: str) -> list[str]:
        logger.warning("Blocked a request to a metadata or link-local address", host=host, address=address)
        raise BlockedAddressError(
            f"{host} resolves to {address}, a link-local or cloud metadata address; requests there are not allowed."
        )

    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Iterable[Any] | None = None,
    ) -> httpcore2.AsyncNetworkStream:
        addresses = await self._resolve(host, port)
        error: Exception | None = None
        for address in addresses:  # connect to the checked address, never re-resolve
            try:
                return await self._inner.connect_tcp(
                    address, port, timeout=timeout, local_address=local_address, socket_options=socket_options
                )
            except (httpcore2.ConnectError, httpcore2.ConnectTimeout) as exc:
                error = exc
        raise error or httpcore2.ConnectError(f"No address found for {host}")

    async def connect_unix_socket(
        self,
        path: str,
        timeout: float | None = None,
        socket_options: Iterable[Any] | None = None,
    ) -> httpcore2.AsyncNetworkStream:
        return await self._inner.connect_unix_socket(path, timeout=timeout, socket_options=socket_options)

    async def sleep(self, seconds: float) -> None:
        await self._inner.sleep(seconds)


def guard_client(client: httpx2.AsyncClient) -> httpx2.AsyncClient:
    """Install `GuardedBackend` in every connection pool of *client*, keeping its
    other settings (environment proxies, limits, TLS). Fails loudly if httpx's
    internals change, rather than silently sending requests unguarded."""
    transports = [client._transport, *client._mounts.values()]
    for transport in transports:
        if transport is None:
            continue
        if not isinstance(transport, httpx2.AsyncHTTPTransport):
            continue  # a test transport, or ASGI: no network
        pool = transport._pool
        if not hasattr(pool, "_network_backend"):
            raise RuntimeError("httpcore2 changed: cannot install the SSRF guard")
        pool._network_backend = GuardedBackend(pool._network_backend)
    return client
