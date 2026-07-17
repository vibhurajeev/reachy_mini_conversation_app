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

    def _feed_chunk(self, chunk: NDArray[np.int16]) -> list[TurnEvent]:
        """Advance the state machine by one fixed-size chunk."""
        prob = self.model(chunk.astype(np.float32) / 32768.0)
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


def load_silero_model() -> SpeechProbModel:
    """Load Silero VAD (ONNX) and adapt it to the SpeechProbModel protocol."""
    from silero_vad import load_silero_vad

    silero = load_silero_vad(onnx=True)

    def _prob(chunk: NDArray[np.float32]) -> float:
        """Return Silero's speech probability for one chunk."""
        import torch

        return float(silero(torch.from_numpy(chunk), SAMPLE_RATE).item())

    return _prob
