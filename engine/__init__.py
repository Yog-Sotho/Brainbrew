"""Brainbrew generation engine: a thin async client for OpenAI-compatible chat APIs."""
from engine.client import ChatClient, EndpointSettings, StructuredOutputError, Usage

__all__ = ["ChatClient", "EndpointSettings", "StructuredOutputError", "Usage"]
