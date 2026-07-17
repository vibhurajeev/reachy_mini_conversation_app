"""Tests for the bridge service pipeline and the thin robot-side handler."""

import base64
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient

from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
from reachy_mini_conversation_app.hermes_backend.client import HermesUnavailableError
from reachy_mini_conversation_app.hermes_backend.phrases import PHRASE_UNREACHABLE
from reachy_mini_conversation_app.hermes_backend.settings import HermesSettings
from reachy_mini_conversation_app.hermes_backend.bridge_server import BridgePipeline, create_app
from reachy_mini_conversation_app.hermes_backend.bridge_handler import BridgeHandler, BridgeUnavailableError


class FakeStt:
    """STT stub returning a fixed transcript."""

    def __init__(self, text: str) -> None:
        """Store the canned transcript."""
        self.text = text

    async def transcribe(self, audio: np.ndarray) -> str:
        """Return the canned transcript."""
        return self.text


class FakeTts:
    """TTS stub emitting constant pcm and recording sentences."""

    def __init__(self) -> None:
        """Track synthesized sentences."""
        self.sentences: list[str] = []

    async def synthesize(self, text: str) -> np.ndarray:
        """Record the sentence and return non-empty pcm."""
        self.sentences.append(text)
        return np.ones(80, dtype=np.int16)

    def available_voices(self) -> list[str]:
        """Return a single fake voice."""
        return ["fake"]

    @property
    def voice(self) -> str:
        """Return the fake voice."""
        return "fake"

    def set_voice(self, voice: str) -> None:
        """Ignore voice changes."""


class FakeHermes:
    """Hermes client stub streaming scripted deltas or raising."""

    def __init__(self, deltas: list[str] | None = None, error: Exception | None = None) -> None:
        """Store scripted deltas and an optional error."""
        self.deltas = deltas or []
        self.error = error

    async def health(self) -> bool:
        """Report healthy."""
        return True

    async def stream_chat(self, text: str):
        """Yield scripted deltas or raise the scripted error."""
        if self.error is not None:
            raise self.error
        for delta in self.deltas:
            yield delta

    async def close(self) -> None:
        """No-op."""


def make_pipeline(deltas: list[str] | None = None, error: Exception | None = None, **settings_kwargs):
    """Build a BridgePipeline with faked engines."""
    settings = HermesSettings(**settings_kwargs)
    tts = FakeTts()
    pipeline = BridgePipeline(settings, FakeStt("intern, say hi"), tts, FakeHermes(deltas, error))
    return pipeline, tts


async def collect(pipeline: BridgePipeline) -> list[dict]:
    """Run the pipeline on dummy audio and collect all events."""
    return [event async for event in pipeline.run(np.ones(1600, dtype=np.int16))]


# ----------------------------------------------------------------- pipeline


@pytest.mark.asyncio
async def test_pipeline_streams_transcript_directives_sentences() -> None:
    """A reply produces transcript, directive, and per-sentence audio events."""
    pipeline, tts = make_pipeline(deltas=["⟦emotion:happy⟧ Hey. All good!"])
    events = await collect(pipeline)
    kinds = [event["type"] for event in events]
    assert kinds == ["transcript", "directive", "sentence", "sentence"]
    assert events[1] == {"type": "directive", "name": "emotion", "argument": "happy"}
    assert tts.sentences == ["Hey.", "All good!"]
    assert base64.b64decode(events[2]["audio_b64"])


@pytest.mark.asyncio
async def test_pipeline_ignore_short_circuits() -> None:
    """The ignore directive ends the stream with an ignored event."""
    pipeline, tts = make_pipeline(deltas=["⟦ignore⟧"])
    events = await collect(pipeline)
    assert [event["type"] for event in events] == ["transcript", "ignored"]
    assert tts.sentences == []


@pytest.mark.asyncio
async def test_pipeline_hermes_down_yields_error_event() -> None:
    """Hermes failure produces a spoken error event."""
    pipeline, tts = make_pipeline(error=HermesUnavailableError("boom"))
    events = await collect(pipeline)
    assert events[-1]["type"] == "error"
    assert events[-1]["phrase"] == PHRASE_UNREACHABLE
    assert tts.sentences == [PHRASE_UNREACHABLE]


# ----------------------------------------------------------------- HTTP app


def make_app(monkeypatch: pytest.MonkeyPatch, deltas: list[str] | None = None):
    """Build the FastAPI app with faked engines and a valid key."""
    monkeypatch.setenv("BRIDGE_API_KEY", "k" * 32)
    return create_app(
        settings=HermesSettings(),
        stt=FakeStt("intern, say hi"),
        tts=FakeTts(),
        hermes=FakeHermes(deltas or ["Hello there."]),
    )


def test_converse_requires_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Requests without the bearer key are rejected."""
    client = TestClient(make_app(monkeypatch))
    response = client.post("/v1/converse", content=b"\x00\x01")
    assert response.status_code == 401


def test_converse_streams_ndjson(monkeypatch: pytest.MonkeyPatch) -> None:
    """A valid request streams ndjson ending with done."""
    client = TestClient(make_app(monkeypatch))
    response = client.post(
        "/v1/converse",
        content=np.ones(1600, dtype=np.int16).tobytes(),
        headers={"Authorization": "Bearer " + "k" * 32},
    )
    assert response.status_code == 200
    import json

    events = [json.loads(line) for line in response.text.strip().splitlines()]
    assert events[0]["type"] == "transcript"
    assert events[-1]["type"] == "done"


def test_create_app_refuses_weak_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing/short key prevents startup."""
    monkeypatch.setenv("BRIDGE_API_KEY", "short")
    with pytest.raises(RuntimeError):
        create_app(settings=HermesSettings())


# ----------------------------------------------------------------- handler


class FakeBridge:
    """Bridge client stub yielding scripted events or raising."""

    def __init__(self, events: list[dict] | None = None, error: Exception | None = None) -> None:
        """Store scripted events and an optional error."""
        self.events = events or []
        self.error = error

    async def health(self) -> bool:
        """Report healthy."""
        return True

    async def converse(self, audio: np.ndarray):
        """Yield scripted events or raise."""
        if self.error is not None:
            raise self.error
        for event in self.events:
            yield event

    async def close(self) -> None:
        """No-op."""


def make_handler(events: list[dict] | None = None, error: Exception | None = None) -> BridgeHandler:
    """Build a BridgeHandler with the bridge faked."""
    deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock(), instance_path=None)
    return BridgeHandler(
        deps, settings=HermesSettings(), vad_model=lambda chunk: 0.0, bridge=FakeBridge(events, error)
    )


@pytest.mark.asyncio
async def test_handler_plays_sentence_audio() -> None:
    """Sentence events land on the output queue as int16 audio."""
    pcm = base64.b64encode(np.ones(80, dtype=np.int16).tobytes()).decode()
    handler = make_handler(
        events=[{"type": "sentence", "text": "Hey.", "audio_b64": pcm, "rate": 16000}, {"type": "done"}]
    )
    await handler._handle_turn(np.ones(16000, dtype=np.int16))
    rate, audio = handler.output_queue.get_nowait()
    assert rate == 16000 and audio.dtype == np.int16 and audio.ndim == 2


@pytest.mark.asyncio
async def test_handler_ignored_is_silent() -> None:
    """An ignored event produces no audio."""
    handler = make_handler(events=[{"type": "ignored"}])
    await handler._handle_turn(np.ones(16000, dtype=np.int16))
    assert handler.output_queue.qsize() == 0


@pytest.mark.asyncio
async def test_handler_bridge_down_plays_packaged_fallback() -> None:
    """Transport failure plays the bundled offline wav."""
    handler = make_handler(error=BridgeUnavailableError("down"))
    await handler._handle_turn(np.ones(16000, dtype=np.int16))
    rate, audio = handler.output_queue.get_nowait()
    assert audio.size > 10000  # the packaged 3.4 s phrase


@pytest.mark.asyncio
async def test_handler_short_utterance_gated() -> None:
    """Utterances under the duration gate never reach the bridge."""
    handler = make_handler(events=[{"type": "sentence", "text": "x", "audio_b64": "", "rate": 16000}])
    await handler._handle_turn(np.ones(1000, dtype=np.int16))
    assert handler.output_queue.qsize() == 0
