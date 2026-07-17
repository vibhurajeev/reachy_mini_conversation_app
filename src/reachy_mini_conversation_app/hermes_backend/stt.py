"""Speech-to-text engines for the Hermes backend (pluggable per D13)."""

import asyncio
import logging
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from reachy_mini_conversation_app.hermes_backend.settings import HermesSettings


logger = logging.getLogger(__name__)


class SpeechToText(Protocol):
    """Transcribe one mono 16 kHz int16 utterance to text."""

    async def transcribe(self, audio: NDArray[np.int16]) -> str:
        """Return the transcript (possibly empty) for the utterance."""
        ...


class FasterWhisperSTT:
    """faster-whisper based STT; model size/device/compute from settings."""

    def __init__(self, settings: HermesSettings) -> None:
        """Store settings; the model loads lazily on first use."""
        self._settings = settings
        self._model: Any = None

    def _ensure_model(self) -> Any:
        """Load the Whisper model once (blocking; called from a worker thread)."""
        if self._model is None:
            from faster_whisper import WhisperModel

            logger.info(
                "Loading faster-whisper model=%s device=%s compute=%s",
                self._settings.stt_model,
                self._settings.stt_device,
                self._settings.stt_compute_type,
            )
            self._model = WhisperModel(
                self._settings.stt_model,
                device=self._settings.stt_device,
                compute_type=self._settings.stt_compute_type,
            )
        return self._model

    async def transcribe(self, audio: NDArray[np.int16]) -> str:
        """Transcribe off the event loop; returns a stripped transcript."""
        if audio.size == 0:
            return ""

        def _run() -> str:
            """Blocking transcription in a worker thread."""
            model = self._ensure_model()
            samples = audio.astype(np.float32) / 32768.0
            segments, _info = model.transcribe(
                samples,
                language=self._settings.stt_language or None,
                beam_size=1,
                vad_filter=False,
            )
            return "".join(segment.text for segment in segments).strip()

        return await asyncio.to_thread(_run)
