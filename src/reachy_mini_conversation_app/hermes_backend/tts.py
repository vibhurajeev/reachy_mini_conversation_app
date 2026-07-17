"""Text-to-speech engines for the Hermes backend (Kokoro default, Piper fallback).

Engines return int16 mono audio at the robot player's rate (play_loop never
resamples — see ARCHITECTURE.md), so resampling happens here.
"""

import asyncio
import logging
import subprocess
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from reachy_mini_conversation_app.hermes_backend.audio import resample_int16, float32_to_int16
from reachy_mini_conversation_app.hermes_backend.settings import HermesSettings


logger = logging.getLogger(__name__)


class TextToSpeech(Protocol):
    """Synthesize one sentence to int16 mono audio at the player rate."""

    async def synthesize(self, text: str) -> NDArray[np.int16]:
        """Return synthesized speech for the text (may be empty on failure)."""
        ...

    def available_voices(self) -> list[str]:
        """Return selectable voice names for the personality UI."""
        ...

    @property
    def voice(self) -> str:
        """Return the active voice name."""
        ...

    def set_voice(self, voice: str) -> None:
        """Select a voice for subsequent synthesis."""
        ...


class KokoroTTS:
    """Kokoro-82M via kokoro-onnx; natural prosody, CPU real-time on servers."""

    def __init__(self, settings: HermesSettings) -> None:
        """Store settings; the model loads lazily on first synthesis."""
        self._settings = settings
        self._voice = settings.tts_voice
        self._kokoro: Any = None

    def _ensure_model(self) -> Any:
        """Load Kokoro once (blocking; called from a worker thread)."""
        if self._kokoro is None:
            from kokoro_onnx import Kokoro

            logger.info("Loading Kokoro TTS model=%s", self._settings.kokoro_model_path)
            self._kokoro = Kokoro(self._settings.kokoro_model_path, self._settings.kokoro_voices_path)
        return self._kokoro

    async def synthesize(self, text: str) -> NDArray[np.int16]:
        """Synthesize off the event loop and resample to the player rate."""
        if not text.strip():
            return np.zeros(0, dtype=np.int16)

        def _run() -> NDArray[np.int16]:
            """Blocking synthesis in a worker thread."""
            kokoro = self._ensure_model()
            samples, rate = kokoro.create(text, voice=self._voice, speed=1.0)
            pcm = float32_to_int16(np.asarray(samples, dtype=np.float32))
            return resample_int16(pcm, int(rate), self._settings.player_sample_rate)

        return await asyncio.to_thread(_run)

    def available_voices(self) -> list[str]:
        """Return a curated set of Kokoro voices (full list lives in the voices file)."""
        return ["af_sarah", "af_bella", "am_adam", "am_michael", "bf_emma", "bm_george"]

    @property
    def voice(self) -> str:
        """Return the active voice name."""
        return self._voice

    def set_voice(self, voice: str) -> None:
        """Select a voice for subsequent synthesis."""
        self._voice = voice


class PiperTTS:
    """Piper via its CLI (`--output-raw`); lightweight fallback engine."""

    def __init__(self, settings: HermesSettings) -> None:
        """Store settings; each synthesis shells out to the piper binary."""
        self._settings = settings
        self._voice = settings.piper_model_path

    async def synthesize(self, text: str) -> NDArray[np.int16]:
        """Run piper for one sentence and resample its raw s16le output."""
        if not text.strip() or not self._settings.piper_model_path:
            return np.zeros(0, dtype=np.int16)

        def _run() -> NDArray[np.int16]:
            """Blocking piper subprocess call in a worker thread."""
            result = subprocess.run(
                ["piper", "--model", self._settings.piper_model_path, "--output-raw"],
                input=text.encode("utf-8"),
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                timeout=30,
                check=False,
            )
            pcm = np.frombuffer(result.stdout, dtype=np.int16)
            return resample_int16(pcm, self._settings.piper_sample_rate, self._settings.player_sample_rate)

        return await asyncio.to_thread(_run)

    def available_voices(self) -> list[str]:
        """Return the configured piper model as the single voice."""
        return [self._voice] if self._voice else []

    @property
    def voice(self) -> str:
        """Return the active voice (the piper model path)."""
        return self._voice

    def set_voice(self, voice: str) -> None:
        """Piper voices are model files; switching requires a new model path."""
        self._voice = voice


def build_tts(settings: HermesSettings) -> TextToSpeech:
    """Construct the configured TTS engine (kokoro default, piper fallback)."""
    if settings.tts_engine.lower() == "piper":
        return PiperTTS(settings)
    return KokoroTTS(settings)
