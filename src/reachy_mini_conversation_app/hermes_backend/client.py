"""HTTP/SSE client for Hermes's OpenAI-compatible api_server platform.

Speaks `POST /v1/chat/completions` with `stream: true`, authenticated with
`Authorization: Bearer <API_SERVER_KEY>`, and scopes conversation + long-term
memory with the `X-Hermes-Session-Key` header (see the plan's D1/D5).
"""

import json
import logging
from typing import Any
from collections.abc import AsyncIterator

import httpx


logger = logging.getLogger(__name__)


class HermesUnavailableError(Exception):
    """Raised when the Hermes api_server cannot be reached or errors out."""


class HermesClient:
    """Thin streaming client for the Hermes api_server."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        session_key: str,
        model: str,
        first_token_timeout_s: float,
        total_timeout_s: float,
    ) -> None:
        """Configure the client; no connection is made until used."""
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._session_key = session_key
        self._model = model
        self._first_token_timeout_s = first_token_timeout_s
        self._total_timeout_s = total_timeout_s
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=10.0, read=first_token_timeout_s, write=10.0, pool=10.0),
        )

    async def health(self) -> bool:
        """Return True when the api_server answers its /health endpoint."""
        try:
            response = await self._client.get(f"{self._base_url}/health")
            return response.status_code == 200
        except httpx.HTTPError as error:
            logger.warning("Hermes health check failed: %s", error)
            return False

    async def stream_chat(self, text: str) -> AsyncIterator[str]:
        """POST one user message and yield text deltas from the SSE stream.

        Raises HermesUnavailableError on connection failures or HTTP errors;
        mid-stream disconnects end the iterator with the same error after any
        already-yielded deltas (the handler speaks what it has, per D7).
        """
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "X-Hermes-Session-Key": self._session_key,
        }
        payload: dict[str, Any] = {
            "model": self._model,
            "stream": True,
            "messages": [{"role": "user", "content": text}],
        }
        try:
            async with self._client.stream(
                "POST",
                f"{self._base_url}/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=httpx.Timeout(
                    connect=10.0,
                    read=self._first_token_timeout_s,
                    write=10.0,
                    pool=10.0,
                ),
            ) as response:
                if response.status_code != 200:
                    body = (await response.aread())[:200]
                    raise HermesUnavailableError(f"Hermes returned HTTP {response.status_code}: {body!r}")
                async for line in response.aiter_lines():
                    delta = _parse_sse_line(line)
                    if delta is not None:
                        yield delta
        except httpx.HTTPError as error:
            raise HermesUnavailableError(f"Hermes stream failed: {error}") from error

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()


def _parse_sse_line(line: str) -> str | None:
    """Extract the content delta from one SSE line ('data: {...}'), if any."""
    line = line.strip()
    if not line.startswith("data:"):
        return None
    data = line[len("data:") :].strip()
    if not data or data == "[DONE]":
        return None
    try:
        event = json.loads(data)
    except json.JSONDecodeError:
        logger.debug("Skipping unparseable SSE line: %.120s", data)
        return None
    choices = event.get("choices") or []
    if not choices:
        return None
    delta = choices[0].get("delta") or {}
    content = delta.get("content")
    return content if isinstance(content, str) and content else None
