"""Pi-side thin handler: VAD locally, everything else via the bridge (D13v2).

Subclasses HermesTextHandler to reuse the VAD/turn/barge-in machinery and
directive dispatch, replacing the turn pipeline: the endpointed utterance is
POSTed to the bridge service, which streams back transcript/directive/sentence
events with ready-to-play audio. No STT, TTS, or LLM runs on the robot.
"""

import json
import time
import wave
import base64
import asyncio
import logging
from typing import Any
from collections.abc import AsyncIterator
from importlib.resources import files

import httpx
import numpy as np

from reachy_mini_conversation_app.hermes_backend.vad import TurnDetector, load_silero_model
from reachy_mini_conversation_app.hermes_backend.actions import Directive
from reachy_mini_conversation_app.hermes_backend.handler import HermesTextHandler
from reachy_mini_conversation_app.hermes_backend.settings import HermesSettings


logger = logging.getLogger(__name__)


class BridgeUnavailableError(Exception):
    """Raised when the bridge service cannot be reached."""


class BridgeClient:
    """Streams converse events from the bridge service."""

    def __init__(self, base_url: str, api_key: str, total_timeout_s: float) -> None:
        """Configure the client; connections are made per call."""
        self._base_url = base_url.rstrip("/")
        self._headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/octet-stream"}
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=5.0, read=total_timeout_s, write=10.0, pool=5.0)
        )

    async def health(self) -> bool:
        """Return True when the bridge answers its /health endpoint."""
        try:
            response = await self._client.get(f"{self._base_url}/health")
            return response.status_code == 200
        except httpx.HTTPError as error:
            logger.warning("Bridge health check failed: %s", error)
            return False

    async def converse(self, audio: "np.ndarray[Any, np.dtype[np.int16]]") -> AsyncIterator[dict[str, Any]]:
        """POST one utterance; yield ndjson events as they stream back."""
        try:
            async with self._client.stream(
                "POST",
                f"{self._base_url}/v1/converse",
                headers=self._headers,
                content=audio.tobytes(),
            ) as response:
                if response.status_code != 200:
                    body = (await response.aread())[:200]
                    raise BridgeUnavailableError(f"bridge returned HTTP {response.status_code}: {body!r}")
                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        logger.debug("Skipping unparseable bridge line: %.120s", line)
        except httpx.HTTPError as error:
            raise BridgeUnavailableError(f"bridge stream failed: {error}") from error

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()


def _load_packaged_wav(name: str) -> "np.ndarray[Any, np.dtype[np.int16]]":
    """Load a bundled 16 kHz mono wav from package data as int16 samples."""
    path = files("reachy_mini_conversation_app.hermes_backend").joinpath(f"data/{name}")
    with wave.open(str(path), "rb") as reader:
        frames = reader.readframes(reader.getnframes())
    return np.frombuffer(frames, dtype=np.int16)


class BridgeHandler(HermesTextHandler):
    """Thin robot-side backend: local VAD + bridge round trips."""

    def __init__(
        self,
        deps: Any,
        instance_path: Any = None,
        startup_voice: str | None = None,
        settings: HermesSettings | None = None,
        vad_model: Any = None,
        bridge: BridgeClient | None = None,
    ) -> None:
        """Wire the thin pipeline; the VAD model loads during start_up."""
        super().__init__(deps, instance_path=instance_path, startup_voice=startup_voice, settings=settings)
        self._bridge = bridge
        self._vad_model = vad_model
        self._fallback_pcm = _load_packaged_wav("hq_unreachable.wav")

    async def start_up(self) -> None:
        """Initialize VAD + bridge client, then serve until shutdown."""
        self.tool_manager.set_loop()
        settings = self.settings
        if self._bridge is None:
            self._bridge = BridgeClient(
                base_url=settings.bridge_url,
                api_key=settings.bridge_api_key,
                total_timeout_s=settings.total_timeout_s,
            )
        if self._vad_model is None:
            self._vad_model = await asyncio.to_thread(load_silero_model)
        self._detector = TurnDetector(
            model=self._vad_model,
            threshold=settings.vad_threshold,
            min_silence_ms=settings.vad_min_silence_ms,
            speech_pad_ms=settings.vad_speech_pad_ms,
        )

        healthy = await self._bridge.health()
        self.connection = self._bridge if healthy else None
        if not healthy:
            logger.error("Bridge unreachable at %s — will retry on first turn", settings.bridge_url)

        self.deps.movement_manager.set_head_tracking(True)
        self._started = True
        self._mark_activity("bridge_startup")
        logger.info("BridgeHandler ready (bridge=%s)", settings.bridge_url)
        await self._shutdown_event.wait()

    async def shutdown(self) -> None:
        """Stop serving and close the bridge client."""
        self._shutdown_event.set()
        self._cancel_turn()
        if self._detector is not None:
            self._detector.reset()
        if self._bridge is not None:
            await self._bridge.close()
        self._started = False
        self.connection = None

    async def _handle_turn(self, utterance: np.ndarray) -> None:
        """Gate locally, then relay the utterance and render bridge events."""
        assert self._bridge is not None
        try:
            if not self._passes_local_gates(utterance):
                return
            self._mark_activity("utterance")
            self._fire_thinking_motion()
            try:
                async for event in self._bridge.converse(utterance):
                    if await self._render_event(event):
                        return
                self.connection = self._bridge
            except BridgeUnavailableError as error:
                logger.warning("Bridge turn failed: %s", error)
                self.connection = None
                await self._play_pcm(self._fallback_pcm)
        except asyncio.CancelledError:
            logger.debug("Turn cancelled (barge-in or shutdown)")
            raise
        except Exception:
            logger.exception("Unexpected error handling bridge turn")
            await self._play_pcm(self._fallback_pcm)
        finally:
            self._set_speaking(False)
            self._turn_task = None

    def _passes_local_gates(self, utterance: np.ndarray) -> bool:
        """Pre-transcript gates: minimum duration and the forward rate cap."""
        if utterance.shape[-1] < 0.4 * 16_000:
            logger.debug("Gate: utterance too short (%d samples)", utterance.shape[-1])
            return False
        return self._passes_rate_cap()

    def _passes_rate_cap(self) -> bool:
        """Apply the D2 forward rate cap (word gate happens bridge-side)."""
        now = time.monotonic()
        self._forward_times = [t for t in self._forward_times if now - t < 60.0]
        if len(self._forward_times) >= self.settings.forwards_per_minute:
            logger.warning("Gate: forward rate cap reached; dropping utterance")
            return False
        self._forward_times.append(now)
        return True

    async def _render_event(self, event: dict[str, Any]) -> bool:
        """Apply one bridge event; returns True when the turn is finished."""
        kind = event.get("type")
        if kind == "transcript":
            logger.info("Bridge transcript: %r", event.get("text"))
            self._mark_activity("transcript")
        elif kind == "ignored":
            logger.info("Bridge: utterance ignored")
            return True
        elif kind == "directive":
            name = str(event.get("name") or "")
            argument = event.get("argument")
            self._dispatch_directives([Directive(name=name, argument=argument)])
        elif kind in ("sentence", "error"):
            audio_b64 = event.get("audio_b64")
            if audio_b64:
                pcm = np.frombuffer(base64.b64decode(audio_b64), dtype=np.int16)
                rate = int(event.get("rate") or self.settings.player_sample_rate)
                await self._play_pcm(pcm, rate)
        elif kind == "done":
            return True
        return False

    async def _play_pcm(self, pcm: "np.ndarray[Any, np.dtype[np.int16]]", rate: int | None = None) -> None:
        """Queue raw pcm for playback and mark speaking activity."""
        if pcm.size == 0:
            return
        self._set_speaking(True)
        self._mark_activity("bridge_audio")
        await self.output_queue.put((rate or self.settings.player_sample_rate, pcm.reshape(1, -1)))

    async def get_available_voices(self) -> list[str]:
        """Voices are a bridge-side concern; expose the single active one."""
        return [self.settings.tts_voice]

    def get_current_voice(self) -> str:
        """Return the configured bridge voice label."""
        return self.settings.tts_voice

    async def change_voice(self, voice: str) -> str:
        """Voice switching happens in the bridge service configuration."""
        return "Voice is configured on the bridge service (TTS_VOICE)."
