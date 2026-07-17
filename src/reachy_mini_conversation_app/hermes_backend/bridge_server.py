"""Bridge service: the VM-side brains for a thin on-robot app (plan D13v2).

Runs next to Hermes. One authenticated endpoint takes an endpointed utterance
(raw int16 mono 16 kHz) and streams back newline-delimited JSON events:

    {"type": "transcript", "text": ...}
    {"type": "ignored"}
    {"type": "directive", "name": ..., "argument": ...}
    {"type": "sentence", "text": ..., "audio_b64": ..., "rate": ...}
    {"type": "error", "phrase": ..., "audio_b64": ..., "rate": ...}
    {"type": "done"}

STT, Hermes (localhost), directive parsing, and TTS all happen here, so the
robot side only captures audio, plays audio, and executes directives.
Launch with the `avail-intern-bridge` console script.
"""

import os
import sys
import json
import base64
import logging
from typing import Any, Optional
from collections.abc import AsyncIterator

import numpy as np
from fastapi import Depends, FastAPI, Request, Response, HTTPException
from fastapi.responses import StreamingResponse

from reachy_mini_conversation_app.hermes_backend.stt import SpeechToText, FasterWhisperSTT
from reachy_mini_conversation_app.hermes_backend.tts import TextToSpeech, build_tts
from reachy_mini_conversation_app.hermes_backend.client import HermesClient, HermesUnavailableError
from reachy_mini_conversation_app.hermes_backend.actions import SentenceChunker, StreamActionParser
from reachy_mini_conversation_app.hermes_backend.phrases import (
    PHRASE_UNREACHABLE,
    PHRASE_STREAM_DROPPED,
    PHRASE_MONOLOGUE_TRAILOFF,
)
from reachy_mini_conversation_app.hermes_backend.settings import HermesSettings


logger = logging.getLogger(__name__)

MIN_KEY_LENGTH = 16


class BridgePipeline:
    """Utterance → transcript → Hermes → directives + spoken sentences."""

    def __init__(self, settings: HermesSettings, stt: SpeechToText, tts: TextToSpeech, hermes: HermesClient) -> None:
        """Store the engines this pipeline orchestrates."""
        self.settings = settings
        self.stt = stt
        self.tts = tts
        self.hermes = hermes

    async def run(self, audio: "np.ndarray[Any, np.dtype[np.int16]]") -> AsyncIterator[dict[str, Any]]:
        """Yield converse events for one utterance."""
        transcript = await self.stt.transcribe(audio)
        logger.info("Bridge transcript: %r", transcript)
        if len(transcript.split()) < self.settings.min_transcript_words:
            yield {"type": "ignored"}
            return
        yield {"type": "transcript", "text": transcript}

        parser = StreamActionParser()
        chunker = SentenceChunker()
        sentences = 0
        got_any = False
        try:
            async for delta in self.hermes.stream_chat(transcript):
                got_any = True
                speakable, directives = parser.feed(delta)
                for directive in directives:
                    if directive.name == "ignore":
                        yield {"type": "ignored"}
                        return
                    yield {"type": "directive", "name": directive.name, "argument": directive.argument}
                for sentence in chunker.feed(speakable):
                    yield await self._sentence_event(sentence)
                    sentences += 1
                    if sentences >= self.settings.max_sentences:
                        yield await self._sentence_event(PHRASE_MONOLOGUE_TRAILOFF)
                        return
        except HermesUnavailableError as error:
            logger.warning("Bridge: Hermes stream failed: %s", error)
            phrase = PHRASE_STREAM_DROPPED if got_any else PHRASE_UNREACHABLE
            yield await self._error_event(phrase)
            return

        remainder = (parser.flush() + chunker.flush()).strip()
        if remainder and sentences < self.settings.max_sentences:
            yield await self._sentence_event(remainder)

    async def _sentence_event(self, text: str) -> dict[str, Any]:
        """Synthesize one sentence into a sentence event."""
        pcm = await self.tts.synthesize(text)
        return {
            "type": "sentence",
            "text": text,
            "audio_b64": base64.b64encode(pcm.tobytes()).decode("ascii"),
            "rate": self.settings.player_sample_rate,
        }

    async def _error_event(self, phrase: str) -> dict[str, Any]:
        """Synthesize a canned failure phrase into an error event."""
        event = await self._sentence_event(phrase)
        return {"type": "error", "phrase": phrase, "audio_b64": event["audio_b64"], "rate": event["rate"]}


def create_app(
    settings: Optional[HermesSettings] = None,
    stt: Optional[SpeechToText] = None,
    tts: Optional[TextToSpeech] = None,
    hermes: Optional[HermesClient] = None,
) -> FastAPI:
    """Build the bridge FastAPI app; engines are injectable for tests."""
    resolved = settings or HermesSettings()
    key = os.getenv("BRIDGE_API_KEY", "")
    if len(key) < MIN_KEY_LENGTH:
        raise RuntimeError(
            "BRIDGE_API_KEY missing or too short (<16 chars). This endpoint runs STT/LLM work — "
            "generate one with `openssl rand -hex 32` and export BRIDGE_API_KEY before starting."
        )

    app = FastAPI(title="Avail Intern bridge", docs_url=None, redoc_url=None)
    state: dict[str, BridgePipeline] = {}

    def pipeline() -> BridgePipeline:
        """Build (once) and return the pipeline with lazy real engines."""
        if "pipeline" not in state:
            hermes_client = hermes or HermesClient(
                base_url=resolved.api_url,
                api_key=resolved.api_key,
                session_key=resolved.session_key,
                model=resolved.model,
                first_token_timeout_s=resolved.first_token_timeout_s,
                total_timeout_s=resolved.total_timeout_s,
            )
            state["pipeline"] = BridgePipeline(
                resolved,
                stt or FasterWhisperSTT(resolved),
                tts or build_tts(resolved),
                hermes_client,
            )
        return state["pipeline"]

    def require_key(request: Request) -> None:
        """Reject requests without the bearer key."""
        header = request.headers.get("authorization", "")
        if header != f"Bearer {key}":
            raise HTTPException(status_code=401, detail="invalid bridge key")

    @app.get("/health")
    async def health() -> Response:
        """Report bridge liveness and Hermes reachability."""
        hermes_ok = await pipeline().hermes.health()
        return Response(
            content=json.dumps({"status": "ok", "hermes": hermes_ok}),
            media_type="application/json",
        )

    @app.post("/v1/converse", dependencies=[Depends(require_key)])
    async def converse(request: Request) -> StreamingResponse:
        """Run one utterance through the pipeline, streaming ndjson events."""
        body = await request.body()
        if not body:
            raise HTTPException(status_code=400, detail="empty audio")
        audio = np.frombuffer(body, dtype=np.int16)

        async def stream() -> AsyncIterator[bytes]:
            """Serialize pipeline events as ndjson."""
            try:
                async for event in pipeline().run(audio):
                    yield (json.dumps(event) + "\n").encode()
            except Exception:
                logger.exception("Bridge pipeline crashed")
                yield (json.dumps({"type": "error", "phrase": PHRASE_STREAM_DROPPED}) + "\n").encode()
            yield (json.dumps({"type": "done"}) + "\n").encode()

        return StreamingResponse(stream(), media_type="application/x-ndjson")

    return app


def main() -> None:
    """Console entry point: run the bridge under uvicorn."""
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    host = os.getenv("BRIDGE_HOST", "0.0.0.0")
    port = int(os.getenv("BRIDGE_PORT", "8643"))
    try:
        app = create_app()
    except RuntimeError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
    uvicorn.run(app, host=host, port=port)
