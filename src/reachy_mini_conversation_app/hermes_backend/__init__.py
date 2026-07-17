"""Hermes gateway backend: VAD → STT → Hermes api_server → TTS + robot actions.

This package implements the "everything through Hermes" architecture (see
plans/hermes-gateway-refactor.md). The robot app is a thin gateway: local turn
detection and speech recognition, an HTTP/SSE round trip to a Hermes agent, and
sentence-streamed speech synthesis plus action directives on the way back.

All modules are additive to upstream; the only upstream touch point is the
backend selector in main.build_handler (CONVERSATION_BACKEND=hermes).
"""

from reachy_mini_conversation_app.hermes_backend.handler import HermesTextHandler


__all__ = ["HermesTextHandler"]
