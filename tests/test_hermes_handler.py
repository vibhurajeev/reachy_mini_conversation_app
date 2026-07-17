"""Tests for the Hermes gateway backend (parser, chunker, VAD, handler flow)."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
from reachy_mini_conversation_app.hermes_backend.vad import TurnDetector, TurnEventKind
from reachy_mini_conversation_app.hermes_backend.client import HermesUnavailableError
from reachy_mini_conversation_app.hermes_backend.actions import SentenceChunker, StreamActionParser
from reachy_mini_conversation_app.hermes_backend.handler import (
    PHRASE_UNREACHABLE,
    PHRASE_MONOLOGUE_TRAILOFF,
    HermesTextHandler,
)
from reachy_mini_conversation_app.hermes_backend.settings import HermesSettings


# ----------------------------------------------------------------- actions


def test_parser_extracts_whole_tags() -> None:
    """Directives are removed from speech and reported."""
    parser = StreamActionParser()
    text, directives = parser.feed("Hello ⟦emotion:happy⟧ world.")
    assert text == "Hello  world."
    assert [(d.name, d.argument) for d in directives] == [("emotion", "happy")]


def test_parser_handles_tag_split_across_chunks() -> None:
    """A directive split across stream chunks is reassembled, never spoken."""
    parser = StreamActionParser()
    text1, directives1 = parser.feed("Sure thing ⟦emo")
    text2, directives2 = parser.feed("tion:excited⟧ boss.")
    assert text1 == "Sure thing "
    assert directives1 == []
    assert text2 == " boss."
    assert [(d.name, d.argument) for d in directives2] == [("emotion", "excited")]


def test_parser_drops_unknown_directives_and_dangling_tag() -> None:
    """Unknown names are dropped; a dangling half-tag never reaches speech."""
    parser = StreamActionParser()
    text, directives = parser.feed("Hi ⟦selfdestruct:now⟧ there ⟦brok")
    assert text == "Hi  there "
    assert directives == []
    assert parser.flush() == ""


def test_parser_ignore_directive() -> None:
    """The bare ignore marker parses with no argument."""
    parser = StreamActionParser()
    text, directives = parser.feed("⟦ignore⟧")
    assert text == ""
    assert [(d.name, d.argument) for d in directives] == [("ignore", None)]


def test_sentence_chunker_splits_and_flushes() -> None:
    """Complete sentences are released; the partial tail comes out on flush."""
    chunker = SentenceChunker()
    assert chunker.feed("One. Two! Thr") == ["One.", "Two!"]
    assert chunker.feed("ee? And then") == ["Three?"]
    assert chunker.flush() == "And then"


# ----------------------------------------------------------------- VAD


def test_turn_detector_endpoints_after_silence() -> None:
    """Speech onset then sustained silence yields SPEECH_START and TURN_END."""
    probs = iter([0.9] * 20 + [0.0] * 40)
    detector = TurnDetector(model=lambda chunk: next(probs, 0.0), min_silence_ms=300)
    events = []
    for _ in range(60):
        events.extend(detector.feed(np.ones(512, dtype=np.int16)))
    kinds = [event.kind for event in events]
    assert kinds == [TurnEventKind.SPEECH_START, TurnEventKind.TURN_END]
    utterance = events[1].utterance
    assert utterance is not None and utterance.size > 0


# ----------------------------------------------------------------- handler


class FakeStt:
    """STT stub returning a fixed transcript."""

    def __init__(self, text: str) -> None:
        """Store the canned transcript."""
        self.text = text

    async def transcribe(self, audio: np.ndarray) -> str:
        """Return the canned transcript."""
        return self.text


class FakeTts:
    """TTS stub returning one second of silence per sentence."""

    def __init__(self) -> None:
        """Track synthesized sentences."""
        self.sentences: list[str] = []

    async def synthesize(self, text: str) -> np.ndarray:
        """Record the sentence and return non-empty pcm."""
        self.sentences.append(text)
        return np.ones(160, dtype=np.int16)

    def available_voices(self) -> list[str]:
        """Return a single fake voice."""
        return ["fake"]

    @property
    def voice(self) -> str:
        """Return the fake voice."""
        return "fake"

    def set_voice(self, voice: str) -> None:
        """Ignore voice changes."""


class FakeClient:
    """Hermes client stub streaming scripted deltas or raising."""

    def __init__(self, deltas: list[str] | None = None, error: Exception | None = None) -> None:
        """Store scripted deltas and an optional error to raise."""
        self.deltas = deltas or []
        self.error = error

    async def health(self) -> bool:
        """Report healthy."""
        return True

    async def stream_chat(self, text: str):
        """Yield scripted deltas, or raise the scripted error immediately."""
        if self.error is not None:
            raise self.error
        for delta in self.deltas:
            yield delta

    async def close(self) -> None:
        """No-op."""


def make_handler(
    deltas: list[str] | None = None, error: Exception | None = None, **settings_kwargs
) -> tuple[HermesTextHandler, FakeTts]:
    """Build a handler with all external pieces faked."""
    deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock(), instance_path=None)
    settings = HermesSettings(**settings_kwargs)
    tts = FakeTts()
    handler = HermesTextHandler(
        deps,
        settings=settings,
        vad_model=lambda chunk: 0.0,
        stt=FakeStt("intern, say hi"),
        tts=tts,
        client=FakeClient(deltas=deltas, error=error),
    )
    return handler, tts


@pytest.mark.asyncio
async def test_turn_speaks_sentences_and_fires_directives() -> None:
    """A streamed reply produces queued audio per sentence, tags stripped."""
    handler, tts = make_handler(deltas=["Hey. ⟦emotion:", "happy⟧ All good!"])
    await handler._handle_turn(np.ones(1600, dtype=np.int16))
    assert tts.sentences == ["Hey.", "All good!"]
    assert handler.output_queue.qsize() == 2
    rate, pcm = handler.output_queue.get_nowait()
    assert pcm.ndim == 2 and pcm.dtype == np.int16


@pytest.mark.asyncio
async def test_ignore_marker_is_fully_silent() -> None:
    """An ignore reply produces no audio at all."""
    handler, tts = make_handler(deltas=["⟦ignore⟧"])
    await handler._handle_turn(np.ones(1600, dtype=np.int16))
    assert tts.sentences == []
    assert handler.output_queue.qsize() == 0


@pytest.mark.asyncio
async def test_unreachable_hermes_speaks_canned_phrase() -> None:
    """Connection failure speaks the canned unreachable phrase."""
    handler, tts = make_handler(error=HermesUnavailableError("boom"))
    await handler._handle_turn(np.ones(1600, dtype=np.int16))
    assert tts.sentences == [PHRASE_UNREACHABLE]


@pytest.mark.asyncio
async def test_monologue_cap_trails_off() -> None:
    """Replies longer than max_sentences are capped with the trail-off."""
    long_reply = " ".join(f"Sentence {i}." for i in range(10))
    handler, tts = make_handler(deltas=[long_reply], max_sentences=3)
    await handler._handle_turn(np.ones(1600, dtype=np.int16))
    assert len(tts.sentences) == 4
    assert tts.sentences[-1] == PHRASE_MONOLOGUE_TRAILOFF


@pytest.mark.asyncio
async def test_short_transcript_is_gated() -> None:
    """Transcripts under the word gate never reach Hermes or TTS."""
    handler, tts = make_handler(deltas=["Should never stream."])
    handler._stt = FakeStt("hi")
    await handler._handle_turn(np.ones(1600, dtype=np.int16))
    assert tts.sentences == []
    assert handler.output_queue.qsize() == 0
