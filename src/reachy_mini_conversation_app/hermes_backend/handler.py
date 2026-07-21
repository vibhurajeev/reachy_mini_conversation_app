"""HermesTextHandler: the gateway backend where all intelligence is Hermes.

Pipeline per turn (see plans/hermes-gateway-refactor.md):
mic frames → TurnDetector (client VAD) → STT → Hermes api_server (SSE) →
StreamActionParser (directives) → SentenceChunker → TTS → output_queue.

Design invariants inherited from the app: audio out only via output_queue
(base emit() drains it); motion only via existing tool instances; every tool
call already runs without blocking audio; barge-in drains queues in place.
"""

import time
import asyncio
import logging
from typing import Any

import numpy as np

from reachy_mini_conversation_app.prompts import get_session_instructions
from reachy_mini_conversation_app.tools.core_tools import ALL_TOOLS, ToolDependencies
from reachy_mini_conversation_app.hermes_backend.stt import SpeechToText, FasterWhisperSTT
from reachy_mini_conversation_app.hermes_backend.tts import TextToSpeech, build_tts
from reachy_mini_conversation_app.hermes_backend.vad import (
    SAMPLE_RATE,
    TurnDetector,
    TurnEventKind,
    SpeechProbModel,
    load_silero_model,
)
from reachy_mini_conversation_app.conversation_handler import AudioFrame, ConversationHandler
from reachy_mini_conversation_app.hermes_backend.audio import to_mono_int16, resample_int16
from reachy_mini_conversation_app.hermes_backend.client import HermesClient, HermesUnavailableError
from reachy_mini_conversation_app.hermes_backend.actions import Directive, SentenceChunker, StreamActionParser
from reachy_mini_conversation_app.hermes_backend.phrases import (  # noqa: F401  (re-exported for back-compat)
    PHRASE_TIMEOUT,
    PHRASE_UNREACHABLE,
    PHRASE_STREAM_DROPPED,
    PHRASE_MONOLOGUE_TRAILOFF,
)
from reachy_mini_conversation_app.hermes_backend.settings import HermesSettings
from reachy_mini_conversation_app.hermes_backend.diagnostics import supervise, install_loop_exception_handler
from reachy_mini_conversation_app.tools.background_tool_manager import BackgroundToolManager


logger = logging.getLogger(__name__)


class HermesTextHandler(ConversationHandler):
    """Conversation backend that routes every turn through Hermes."""

    def __init__(
        self,
        deps: ToolDependencies,
        instance_path: Any = None,
        startup_voice: str | None = None,
        settings: HermesSettings | None = None,
        vad_model: SpeechProbModel | None = None,
        stt: SpeechToText | None = None,
        tts: TextToSpeech | None = None,
        client: HermesClient | None = None,
    ) -> None:
        """Wire the pipeline; heavy models load lazily during start_up."""
        super().__init__()
        self.deps = deps
        self.instance_path = instance_path
        self.settings = settings or HermesSettings()
        self.output_queue: asyncio.Queue[Any] = asyncio.Queue()
        self.tool_manager = BackgroundToolManager()
        self.connection: object | None = None  # Truthy only when Hermes is reachable (UI pill).

        self._vad_model = vad_model
        self._detector: TurnDetector | None = None
        self._stt = stt
        self._tts = tts
        if startup_voice and self._tts is not None:
            self._tts.set_voice(startup_voice)
        self._client = client
        self._shutdown_event = asyncio.Event()
        self._turn_task: asyncio.Task[None] | None = None
        self._speaking = False
        self._forward_times: list[float] = []
        self._started = False
        self._logged_first_frame = False

    # ------------------------------------------------------------------ lifecycle

    async def start_up(self) -> None:
        """Initialize engines, check Hermes health, then serve until shutdown."""
        self.tool_manager.set_loop()
        settings = self.settings
        if self._client is None:
            self._client = HermesClient(
                base_url=settings.api_url,
                api_key=settings.api_key,
                session_key=settings.session_key,
                model=settings.model,
                first_token_timeout_s=settings.first_token_timeout_s,
                total_timeout_s=settings.total_timeout_s,
            )
        if self._stt is None:
            self._stt = FasterWhisperSTT(settings)
        if self._tts is None:
            self._tts = build_tts(settings)
        if self._vad_model is None:
            self._vad_model = await asyncio.to_thread(load_silero_model)
        self._detector = TurnDetector(
            model=self._vad_model,
            threshold=settings.vad_threshold,
            min_silence_ms=settings.vad_min_silence_ms,
            speech_pad_ms=settings.vad_speech_pad_ms,
        )

        healthy = await self._client.health()
        self.connection = self._client if healthy else None
        if not healthy:
            logger.error("Hermes api_server unreachable at %s — will retry on first turn", settings.api_url)

        self._install_diagnostics()
        self.deps.movement_manager.set_head_tracking(True)
        self._started = True
        self._mark_activity("hermes_startup")
        logger.info("HermesTextHandler ready (api=%s, session_key=%s)", settings.api_url, settings.session_key)
        await self._shutdown_event.wait()

    def _install_diagnostics(self) -> None:
        """Route swallowed event-loop errors through the logger (crash visibility)."""
        try:
            install_loop_exception_handler(asyncio.get_running_loop())
        except RuntimeError:
            logger.debug("No running loop for diagnostics install (ignored)")

    async def shutdown(self) -> None:
        """Stop serving: cancel any in-flight turn and close the HTTP client."""
        self._shutdown_event.set()
        self._cancel_turn()
        if self._detector is not None:
            self._detector.reset()
        if self._client is not None:
            await self._client.close()
        self._started = False
        self.connection = None

    def _is_connected(self) -> bool:
        """Report readiness (used by the idle machinery and status UI)."""
        return self._started

    # ------------------------------------------------------------------ audio in

    async def receive(self, frame: AudioFrame) -> None:
        """Feed a mic frame to the turn detector and react to turn events."""
        if self._detector is None:
            return
        sample_rate, audio = frame
        arr = np.asarray(audio)
        if not self._logged_first_frame:
            self._logged_first_frame = True
            logger.info("First mic frame: rate=%s shape=%s dtype=%s", sample_rate, arr.shape, arr.dtype)
        mono = to_mono_int16(arr)
        if sample_rate != SAMPLE_RATE:
            mono = resample_int16(mono, sample_rate, SAMPLE_RATE)

        for event in self._detector.feed(mono):
            if event.kind is TurnEventKind.SPEECH_START:
                logger.debug("VAD: speech start")
                self._on_speech_start()
            elif event.kind is TurnEventKind.TURN_END and event.utterance is not None:
                logger.debug(
                    "VAD: turn end (%d samples, %.2fs)", event.utterance.size, event.utterance.size / SAMPLE_RATE
                )
                self._on_turn_end(event.utterance)

    def _on_speech_start(self) -> None:
        """User started talking: mark activity, freeze antennas, barge in."""
        self._mark_activity("user_speech_start")
        self.deps.movement_manager.set_listening(True)
        if self._speaking or self._turn_task is not None:
            self._barge_in()

    def _on_turn_end(self, utterance: np.ndarray) -> None:
        """User stopped talking: unfreeze and process the utterance."""
        self._mark_activity("user_turn_end")
        self.deps.movement_manager.set_listening(False)
        self._cancel_turn()
        self._turn_task = supervise(self._handle_turn(utterance), name="hermes-turn")

    def _barge_in(self) -> None:
        """Stop current output: cancel the turn and drain queued audio."""
        logger.debug("Barge-in: cancelling in-flight turn and clearing audio")
        self._cancel_turn()
        if self._clear_queue is not None:
            self._clear_queue()

    def _cancel_turn(self) -> None:
        """Cancel the in-flight turn task, if any."""
        if self._turn_task is not None and not self._turn_task.done():
            self._turn_task.cancel()
        self._turn_task = None
        self._set_speaking(False)

    # ------------------------------------------------------------------ turn flow

    async def _handle_turn(self, utterance: np.ndarray) -> None:
        """STT → gates → Hermes stream → directives + sentence-chunked TTS."""
        assert self._stt is not None and self._tts is not None and self._client is not None
        try:
            transcript = await self._stt.transcribe(utterance)
            logger.info("Transcript: %r", transcript)
            if not self._passes_gates(transcript):
                return
            self._mark_activity("transcript")
            self._fire_thinking_motion()

            parser = StreamActionParser()
            chunker = SentenceChunker()
            sentences_spoken = 0
            ignored = False
            got_any_delta = False

            try:
                async for delta in self._client.stream_chat(transcript):
                    got_any_delta = True
                    speakable, directives = parser.feed(delta)
                    if self._dispatch_directives(directives):
                        ignored = True
                        break
                    for sentence in chunker.feed(speakable):
                        sentences_spoken += await self._speak(sentence)
                        if sentences_spoken >= self.settings.max_sentences:
                            await self._speak(PHRASE_MONOLOGUE_TRAILOFF)
                            return
            except HermesUnavailableError as error:
                logger.warning("Hermes stream error: %s", error)
                if got_any_delta:
                    tail = chunker.flush()
                    if tail:
                        await self._speak(tail)
                    await self._speak(PHRASE_STREAM_DROPPED)
                else:
                    self.connection = None
                    await self._speak(PHRASE_UNREACHABLE)
                return

            if ignored:
                logger.info("Hermes chose to ignore this utterance")
                return

            remainder = parser.flush() + chunker.flush()
            tail_sentences = SentenceChunker().feed(remainder)
            trailing = remainder if not tail_sentences else None
            for sentence in tail_sentences:
                if sentences_spoken >= self.settings.max_sentences:
                    break
                sentences_spoken += await self._speak(sentence)
            if trailing and sentences_spoken < self.settings.max_sentences:
                await self._speak(trailing)
            self.connection = self._client
        except asyncio.CancelledError:
            logger.debug("Turn cancelled (barge-in or shutdown)")
            raise
        except Exception:
            logger.exception("Unexpected error handling turn")
            await self._speak(PHRASE_STREAM_DROPPED)
        finally:
            self._set_speaking(False)
            self._turn_task = None

    def _passes_gates(self, transcript: str) -> bool:
        """Apply D10 local gates: minimum words and the forward rate cap."""
        words = transcript.split()
        if len(words) < self.settings.min_transcript_words:
            logger.debug("Gate: transcript too short (%r)", transcript)
            return False
        now = time.monotonic()
        self._forward_times = [t for t in self._forward_times if now - t < 60.0]
        if len(self._forward_times) >= self.settings.forwards_per_minute:
            logger.warning("Gate: forward rate cap reached; dropping %r", transcript)
            return False
        self._forward_times.append(now)
        return True

    async def _speak(self, sentence: str) -> int:
        """Synthesize one sentence onto the output queue; returns 1 if spoken."""
        assert self._tts is not None
        text = sentence.strip()
        if not text:
            return 0
        pcm = await self._tts.synthesize(text)
        if pcm.size == 0:
            return 0
        self._set_speaking(True)
        self._mark_activity("tts")
        await self.output_queue.put((self.settings.player_sample_rate, pcm.reshape(1, -1)))
        return 1

    def _set_speaking(self, speaking: bool) -> None:
        """Track speaking state and mirror it to the movement manager."""
        if speaking != self._speaking:
            self._speaking = speaking
            self.deps.movement_manager.set_speaking(speaking)

    def _fire_thinking_motion(self) -> None:
        """Fire-and-forget a thinking emotion while Hermes works (D7)."""
        tool = ALL_TOOLS.get("play_emotion")
        if tool is None:
            return

        async def _run() -> None:
            """Run the emotion tool, ignoring failures (liveliness is optional)."""
            try:
                await tool(self.deps, emotion=self.settings.thinking_emotion)
            except Exception:
                logger.debug("Thinking motion failed (ignored)", exc_info=True)

        supervise(_run(), name="hermes-thinking-motion")

    def _dispatch_directives(self, directives: list[Directive]) -> bool:
        """Run action directives via existing tools; returns True on ⟦ignore⟧."""
        for directive in directives:
            if directive.name == "ignore":
                return True
            if directive.name == "emotion" and directive.argument:
                self._fire_tool("play_emotion", emotion=directive.argument)
            elif directive.name == "dance":
                if directive.argument:
                    self._fire_tool("dance", move=directive.argument)
                else:
                    self._fire_tool("dance")
            elif directive.name == "look" and directive.argument:
                self._fire_tool("move_head", direction=directive.argument)
        return False

    def _fire_tool(self, name: str, **kwargs: Any) -> None:
        """Fire-and-forget one registered tool call (motion never blocks audio)."""
        tool = ALL_TOOLS.get(name)
        if tool is None:
            logger.info("Directive requested unavailable tool %r", name)
            return

        logger.debug("Dispatching directive tool %r args=%r", name, kwargs)

        async def _run() -> None:
            """Run the tool, logging (not raising) any failure."""
            try:
                await tool(self.deps, **kwargs)
            except Exception:
                logger.warning("Directive tool %r failed", name, exc_info=True)

        supervise(_run(), name=f"hermes-action-{name}")

    # ------------------------------------------------------------------ UI surface

    async def apply_personality(self, profile: str | None) -> str:
        """Acknowledge; the app restarts the backend, which re-reads the profile."""
        # get_session_instructions stays the local-prompt source for parity/logging.
        _ = get_session_instructions(self.instance_path)
        return f"Profile {profile or 'default'} applies on backend restart (personality lives in Hermes SOUL.md)."

    async def get_available_voices(self) -> list[str]:
        """Expose the TTS engine's voices to the personality UI."""
        return self._tts.available_voices() if self._tts is not None else []

    def get_current_voice(self) -> str:
        """Return the active TTS voice."""
        return self._tts.voice if self._tts is not None else self.settings.tts_voice

    async def change_voice(self, voice: str) -> str:
        """Switch the TTS voice for subsequent sentences."""
        if self._tts is None:
            return "TTS engine not ready yet."
        self._tts.set_voice(voice)
        return f"Voice changed to {voice}."
