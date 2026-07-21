"""Client-side turn detection for the Hermes backend.

The upstream app delegates all turn-taking to the realtime server's VAD; this
backend must endpoint utterances itself (see ARCHITECTURE.md "one voice turn").
A small state machine wraps a speech-probability model (Silero VAD by default;
tests inject a fake) and yields turn events with the buffered utterance audio.
"""

import enum
import logging
from typing import Protocol
from dataclasses import field, dataclass

import numpy as np
from numpy.typing import NDArray


logger = logging.getLogger(__name__)

SAMPLE_RATE = 16_000
CHUNK_SAMPLES = 512  # Silero's required chunk size at 16 kHz.


class SpeechProbModel(Protocol):
    """Callable returning speech probability for one 512-sample 16 kHz chunk."""

    def __call__(self, chunk: NDArray[np.float32]) -> float:
        """Return P(speech) in [0, 1] for the given chunk."""
        ...


class TurnEventKind(enum.Enum):
    """Kinds of events produced by the turn detector."""

    SPEECH_START = "speech_start"
    TURN_END = "turn_end"


@dataclass
class TurnEvent:
    """A turn-taking event; TURN_END carries the full utterance audio."""

    kind: TurnEventKind
    utterance: NDArray[np.int16] | None = None


@dataclass
class TurnDetector:
    """Streaming endpointer: feed int16 mono 16 kHz audio, receive turn events."""

    model: SpeechProbModel
    threshold: float = 0.5
    min_silence_ms: int = 600
    speech_pad_ms: int = 100
    max_utterance_s: float = 30.0

    _pending: NDArray[np.int16] = field(default_factory=lambda: np.zeros(0, dtype=np.int16))
    _utterance: list[NDArray[np.int16]] = field(default_factory=list)
    _pad: list[NDArray[np.int16]] = field(default_factory=list)
    _in_speech: bool = False
    _silence_samples: int = 0
    _trace_chunks: int = 0
    _trace_max_prob: float = 0.0
    _trace_peak: float = 0.0

    def feed(self, audio: NDArray[np.int16]) -> list[TurnEvent]:
        """Consume a frame of int16 mono 16 kHz audio and return any turn events."""
        events: list[TurnEvent] = []
        self._pending = np.concatenate([self._pending, audio])
        while self._pending.shape[0] >= CHUNK_SAMPLES:
            chunk = self._pending[:CHUNK_SAMPLES]
            self._pending = self._pending[CHUNK_SAMPLES:]
            events.extend(self._feed_chunk(chunk))
        return events

    def reset(self) -> None:
        """Drop all buffered state (used on shutdown/barge-in cleanup)."""
        self._pending = np.zeros(0, dtype=np.int16)
        self._utterance = []
        self._pad = []
        self._in_speech = False
        self._silence_samples = 0

    def _trace(self, chunk: NDArray[np.int16], prob: float) -> None:
        """Once per ~second, log peak mic level and max speech probability.

        Makes an otherwise-silent VAD observable: near-zero peak means no audio
        is reaching the detector; a healthy peak with low prob means the
        threshold is too high (or it is genuinely not speech).
        """
        peak = float(np.abs(chunk).max()) / 32768.0
        self._trace_peak = max(self._trace_peak, peak)
        self._trace_max_prob = max(self._trace_max_prob, prob)
        self._trace_chunks += 1
        if self._trace_chunks >= 30:  # ~1s at 512-sample chunks / 16 kHz
            logger.info(
                "VAD trace: peak=%.3f max_prob=%.2f threshold=%.2f in_speech=%s",
                self._trace_peak,
                self._trace_max_prob,
                self.threshold,
                self._in_speech,
            )
            self._trace_chunks = 0
            self._trace_peak = 0.0
            self._trace_max_prob = 0.0

    def _feed_chunk(self, chunk: NDArray[np.int16]) -> list[TurnEvent]:
        """Advance the state machine by one fixed-size chunk."""
        prob = self.model(chunk.astype(np.float32) / 32768.0)
        self._trace(chunk, prob)
        events: list[TurnEvent] = []

        if not self._in_speech:
            self._pad.append(chunk)
            pad_chunks = max(1, int(self.speech_pad_ms * SAMPLE_RATE / 1000 / CHUNK_SAMPLES))
            if len(self._pad) > pad_chunks:
                self._pad.pop(0)
            if prob >= self.threshold:
                self._in_speech = True
                self._silence_samples = 0
                self._utterance = list(self._pad)
                self._pad = []
                events.append(TurnEvent(TurnEventKind.SPEECH_START))
            return events

        self._utterance.append(chunk)
        if prob >= self.threshold:
            self._silence_samples = 0
        else:
            self._silence_samples += CHUNK_SAMPLES

        utterance_samples = sum(part.shape[0] for part in self._utterance)
        silence_needed = int(self.min_silence_ms * SAMPLE_RATE / 1000)
        if self._silence_samples >= silence_needed or utterance_samples >= self.max_utterance_s * SAMPLE_RATE:
            utterance = np.concatenate(self._utterance) if self._utterance else np.zeros(0, dtype=np.int16)
            self.reset()
            events.append(TurnEvent(TurnEventKind.TURN_END, utterance=utterance))
        return events


CONTEXT_SAMPLES = 64  # Silero v5 prepends 64 samples of prior context at 16 kHz.


class SileroOnnxModel:
    """Silero VAD v5 via onnxruntime — no torch dependency, Pi-friendly.

    The 2.3 MB MIT-licensed model ships in package data. Each call feeds the
    64-sample tail of the previous chunk prepended to the current 512 samples
    (576 total) — the model is trained that way, and omitting the context makes
    it output near-zero for real speech. Recurrent state and context are carried
    between calls, so one instance serves one audio stream.
    """

    def __init__(self, model_path: str | None = None) -> None:
        """Create the inference session and zeroed recurrent state + context."""
        import onnxruntime as ort  # heavy optional dep; only needed for real VAD

        if model_path is None:
            from importlib.resources import files

            model_path = str(files("reachy_mini_conversation_app.hermes_backend").joinpath("data/silero_vad.onnx"))
        self._session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, CONTEXT_SAMPLES), dtype=np.float32)
        self._sr = np.array(SAMPLE_RATE, dtype=np.int64)

    def __call__(self, chunk: NDArray[np.float32]) -> float:
        """Return P(speech) for one 512-sample chunk, advancing state + context."""
        x = np.concatenate([self._context, chunk.reshape(1, -1)], axis=1)
        prob, self._state = self._session.run(
            None,
            {"input": x, "state": self._state, "sr": self._sr},
        )
        self._context = x[:, -CONTEXT_SAMPLES:]
        return float(prob[0][0])


def load_silero_model() -> SpeechProbModel:
    """Load the packaged Silero VAD (ONNX) as a SpeechProbModel."""
    return SileroOnnxModel()
