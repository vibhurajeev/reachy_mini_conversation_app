# ♻️ refactor: make the robot app a thin voice/body gateway to Hermes

**Date:** 2026-07-17 · **Branch:** `avail-intern` · **Type:** architectural refactor
**Rule:** no `git push` without Vibhu's review (CLAUDE.md).

## Overview

Replace the two-brain design (HF realtime speech-to-speech model as conversational
brain + `ask_hermes` SSH tool for delegation) with a single brain: **all intelligence
goes through Hermes** on the sandboxed VM. The robot app becomes a gateway:

```
mic → Silero VAD → local STT → HTTP (SSE) → Hermes api_server (VM, warm)
                                                │  Kimi K3, SOUL.md, memory, Slack
speaker ← sentence-chunked TTS ← text stream ←──┘
robot motions ← action directives parsed from the stream
```

New `HermesTextHandler` implements the existing `ConversationHandler` ABC
(`conversation_handler.py:26`) — audio loops, 60 Hz motion system, tool registry,
idle policy all reused unchanged. Divergence stays additive: new modules + an env-var
branch in `build_handler` (`main.py:167`).

## Key research findings this plan builds on

1. **Transport: Hermes `api_server`, not SSH.** Hermes ships an OpenAI-compatible
   HTTP platform (`gateway/platforms/api_server.py`): `POST /v1/chat/completions`
   with `stream:true` → SSE token deltas. Warm process (no per-turn CLI cold start),
   `API_SERVER_KEY` auth, port 8642. `X-Hermes-Session-Key` header scopes
   conversation + long-term (Honcho) memory per channel. Per-key requests serialize;
   distinct keys run concurrently — robot and Slack don't fight.
2. **The app's only turn-taking is server-side VAD** (`huggingface_realtime.py:231`)
   — the new handler must bring client-side VAD.
3. **`play_loop` never resamples** (`console.py:892` ignores the rate) — TTS output
   must match the media player rate. ⚠️ Verify the player's expected rate on real
   hardware before finalizing TTS config (SDK not installed on the workstation).
4. **Pi 5 voice stack** (benchmarked): Silero VAD ONNX (<1 ms/chunk), faster-whisper
   `base` int8 (~real-time) or Moonshine (sub-200 ms, streaming), Piper TTS
   (~40 ms to first audio; use maintained `OHF-Voice/piper1-gpl`, GPL-3.0 — Kokoro-82M
   as later quality upgrade).
5. **Motion from text:** call existing tool instances directly —
   `await ALL_TOOLS["play_emotion"](deps, emotion=...)` (`core_tools.py:561`,
   intent mapping at `tools/play_emotion.py:195`). No new motion code.
6. **Inline action tags are the fragile option** (documented "tags spoken aloud"
   failures) — mitigate with a tiny grammar, stream-aware parser, and a defensive
   pre-TTS scrubber.

## Design decisions (resolving the spec-flow gaps)

| # | Gap | Decision (v1) |
|---|---|---|
| D1 | Transport | `api_server` over Tailscale (or user-run SSH tunnel). Never exposed publicly; `API_SERVER_KEY` secret on the Pi. |
| D2 | Ambient noise → real agent with Slack tools | **Always-on client, server-side judgment** (user decision, 2026-07-17): no wake word in v1 — the robot forwards every VAD/STT-passing utterance, and Hermes decides whether it was addressed (SOUL.md robot-channel rules: respond when named/directly asked/in a live exchange; otherwise reply exactly `⟦ignore⟧`, which the robot treats as silence — no TTS, no ack cue). **Mandatory companion controls:** (a) `platform_toolsets.api_server` restriction so the robot channel has no side-effect tools (an unaddressed or misheard utterance can never touch Slack); (b) local pre-filters per D10 plus a rate cap (e.g. ≤10 forwards/min) to bound token spend; (c) disclosure: ambient office speech transits STT→VM→OpenRouter — tell the team. Wake word (openWakeWord) remains a phase-3 option if always-on proves noisy or costly; gaze-gating remains the eventual upgrade. |
| D3 | Barge-in during think-time | Client aborts the SSE request and sends the new utterance as the next message on the same session key (Hermes serializes per key). Because of D2's toolset restriction, an abandoned server-side run cannot have harmful side effects. Barge-in mid-TTS: stop playback, flush sentence buffer, drain `output_queue` (existing `_clear_queue` path), abort stream. |
| D4 | Dual memory | **Hermes/Honcho is the only memory.** Remove `remember`/`forget` from the hermes profile's tools.txt; `memory.v1.json` unused in this mode. |
| D5 | Identity across surfaces | Robot uses one persistent session key `agent:main:robot:office` (persisted on the Pi, survives reboots). Siloed from Slack DMs in v1 — stated limitation; unification is a roadmap item. Session id (transcript) rotates on wake after >4 h sleep; key (memory) never rotates. |
| D6 | Camera questions | v1 non-goal. SOUL.md gets an explicit robot-channel instruction: "you cannot see; decline vision questions gracefully and suggest asking again later." Frame-attachment via OpenAI-style image content parts is a phase-3 investigation. |
| D7 | Failures | Canned local phrases (never silent): VM unreachable → "HQ's not picking up, typical"; timeout to first token (ceiling 45 s, progress quip at ~10 s); mid-stream drop → speak up to last complete sentence + short error note. One retry with backoff for connect errors only. Thinking animation always has a timeout-to-idle. |
| D8 | Monologue cap | Client-side cap: 6 sentences per turn, then trail off ("…there's more, ask me"). Independent of barge-in. |
| D9 | Action channel | Single-line directives `⟦emotion:happy⟧` / `⟦dance⟧` / `⟦look:left⟧` / `⟦ignore⟧` on their own tokens, tiny vocabulary (play_emotion intents + dance + look + the D2 ignore marker). Stream-aware parser handles tags split across chunks; scrubber strips *anything* matching `⟦…⟧` before TTS; unknown names → log + ignore; action-only replies get a soft acknowledgment earcon so silence never reads as deafness — except `⟦ignore⟧`, which is fully silent by design. SOUL.md documents the grammar for the robot channel. |
| D10 | STT gating | Drop transcripts < 2 words or below confidence threshold; never forward empties. |
| D11 | Proactive push (Slack → robot speaks) | v1 non-goal (api_server is request/reply). Phase 3: custom "robot" platform plugin over websocket via `register_platform` (`gateway/platform_registry.py`) — Hermes supports this with zero core changes. |
| D12 | Profile switch semantics | Robot-side profile now controls: TTS voice, always-on flag, and the local instructions fragment (thin). Personality lives in SOUL.md. Switch applies on backend restart (existing `handler_factory` rebuild path, `console.py:266`) — no mid-stream switching. |
| D13 | Deployment topology (user decision, 2026-07-17) | **The app runs server-side, co-located with Hermes — the Pi runs only the stock Reachy daemon.** The SDK is location-agnostic: mic/speaker/camera stream over WebRTC (~100 ms) and motion commands go over the LAN, exactly like the Lite variant. Consequences: STT/TTS upgrade to server-class models (faster-whisper `small`/`large-v3-turbo`, Kokoro-82M) — the Pi-feasibility constraint is void; bridge↔Hermes is localhost; one deploy target; phase-3 vision gets frames where the compute is. Requirements: VM must reach the robot's LAN (Tailscale/on-prem); **day-1 spike: validate motion smoothness + audio jitter with the app remote over the real office Wi-Fi**. Fallback (same code, config-only): run the app on the Pi with the lightweight model set (Silero + whisper-base int8 + Piper) if Wi-Fi proves unreliable — keep STT/TTS pluggable so both profiles work. |

## Implementation phases

### Phase 1 — `HermesTextHandler` MVP (robot talks through Hermes)

New modules (no upstream edits except the `build_handler` branch + config var):

- `src/reachy_mini_conversation_app/hermes_backend/handler.py` — `HermesTextHandler(ConversationHandler)`:
  `receive()` → ring buffer → Silero VAD endpointing → STT → POST SSE → sentence
  chunker → TTS → `output_queue.put((rate, pcm))`. Implements the 8 abstract methods;
  `self.connection` truthy when the VM `/health` check passes (keeps the UI pill working);
  `_mark_activity()` at speech/STT/TTS events; `set_listening/set_speaking` wired to
  VAD/TTS state; thinking-motion trigger on endpoint (existing emotion tools).
- `hermes_backend/vad.py` (Silero ONNX, 250 ms onset / 600 ms endpoint),
  `hermes_backend/stt.py` (faster-whisper base int8; interface allows Moonshine swap),
  `hermes_backend/tts.py` (Piper, sentence-streamed; resample to player rate),
  `hermes_backend/client.py` (httpx SSE client, session-key header, abort, retry, timeouts).
- `main.py`: `CONVERSATION_BACKEND=hermes|hf` branch in `build_handler` (`:167`);
  config.py: register new env vars (`HERMES_API_URL`, `HERMES_API_KEY`,
  `HERMES_SESSION_KEY`, `CONVERSATION_BACKEND`, STT/TTS knobs).
- `profiles/avail_intern/`: drop `ask_hermes` (obsolete — the handler *is* the bridge),
  drop `remember`/`forget` (D4); instructions.txt shrinks to voice/format hints.
- VM side (documented in INSTALL.md, user-run): `API_SERVER_ENABLED=1`,
  `API_SERVER_KEY`, `platform_toolsets.api_server` restriction (D2), SOUL.md robot-channel
  section (D6, D9 grammar), Tailscale/tunnel setup.
- Tests: `tests/test_hermes_handler.py` (mock STT/TTS/transport per
  `test_huggingface_realtime.py` conventions): endpoint→request flow, streamed
  sentences → ordered `output_queue` audio, barge-in abort, error phrases, D8 cap,
  handler-selection branch.

**Success:** full conversation loop with the app running server-side next to Hermes
(per D13; Pi runs only the stock daemon), HF backend untouched behind the env var;
failure modes speak, never hang; the D13 remote-app spike (motion smoothness + audio
jitter over real office Wi-Fi) passes before the rest of the phase proceeds.
**Est:** 3-5 days.

### Phase 2 — Expressiveness & robustness

Action-directive parser + scrubber (D9) with leak-rate test (≤1 spoken tag per 100
turns on a scripted corpus); `⟦ignore⟧` addressee flow + forward rate cap (D2);
thinking-motion loop with cutoff (D7); startup greeting via local canned line +
async Hermes warm-up ping; sleep/wake session-id rotation (D5); STT gating
thresholds (D10); latency instrumentation (per-stage timings logged). **Est:** 3-4 days.

### Phase 3 — Investigations (separate mini-plans)

Camera frames to Hermes (D6); proactive push via custom robot platform plugin (D11);
Kokoro TTS upgrade; cross-surface identity unification (D5); semantic turn detection.

## Acceptance criteria (v1 = phases 1-2)

- [ ] Question → spoken answer via Hermes with first audio ≤ 3 s after Hermes's first
      token; thinking motion starts ≤ 500 ms after end-of-turn.
- [ ] Barge-in mid-TTS stops speech ≤ 300 ms; barge-in during think-time cancels and
      re-asks per D3; no orphaned audio in the queue.
- [ ] VM down / 5xx / timeout / mid-stream drop each produce their canned spoken
      response; nothing hangs; thinking motion always times out to idle.
- [ ] No spoken action tags in the leak-rate test; unknown directives ignored+logged;
      action-only reply produces the acknowledgment cue.
- [ ] Robot channel's Hermes toolset verified unable to post to Slack (attempt fails).
- [ ] Addressee gating: utterances not aimed at the robot get `⟦ignore⟧` and produce
      zero audio/motion; addressed utterances answer normally; forwards respect the
      rate cap.
- [ ] ≤ 6-sentence monologue cap enforced client-side.
- [ ] Same-session memory: fact told to the robot is recalled by the robot next day
      (Honcho, key persistence across reboot).
- [ ] `CONVERSATION_BACKEND=hf` still runs the upstream path; upstream tests green;
      lint/mypy/uv-lock CI pass.
- [ ] TTS output rate matches player rate, verified on hardware (finding #3).

## Risks

| Risk | Mitigation |
|---|---|
| Pi 5 CPU contention (STT + TTS + VAD + app) | int8 models, benchmark early in phase 1; Moonshine/cloud-STT fallback path in `stt.py` interface |
| Kimi K3 tag discipline worse than expected | scrubber guarantees nothing leaks to TTS; worst case actions just fire less; can constrain via SOUL.md examples |
| api_server semantics differ from report | Verify `/health`, streaming, session headers against the VM early (day 1 spike, user runs the VM-side commands) |
| Latency disappointment vs old realtime backend | Set expectations: this trades snappiness for one brain with real memory/tools; keep HF backend switchable for demos |
| Upstream drift on `main.py`/handler ABC | additive modules + tiny branch = cheap merges |

## References

- `ARCHITECTURE.md` (this repo) — full app map.
- Seam analysis: handler contract `conversation_handler.py:97-132`; emit/output_queue
  `conversation_handler.py:62` + `console.py:872-915`; no-resample `console.py:892`;
  server-VAD `huggingface_realtime.py:231`; motion APIs `moves.py:240-293`,
  `dance_emotion_moves.py`, `tools/play_emotion.py:195`; selection hook `main.py:167`;
  UI pill `console.py:254`.
- Hermes (source at `/home/vibhu/.hermes/hermes-agent`): `gateway/platforms/api_server.py`
  (:1533 headers, :2567 SSE), `gateway/config.py:1834` (env), `gateway/session.py:893`
  (keys), `gateway/run.py:5401` (per-key queue), `gateway/platform_registry.py` (plugins).
- Voice stack: Silero VAD (github.com/snakers4/silero-vad), faster-whisper Pi benchmarks
  (promptquorum.com/power-local-llm/local-whisper-stt-comparison-2026), Moonshine
  (github.com/moonshine-ai/moonshine), Piper (github.com/OHF-Voice/piper1-gpl),
  Kokoro comparison (contracollective.com/blog/kokoro-vs-piper-vs-xtts-local-text-to-speech-m5-max-2026),
  latency-masking (pipecat-ai/pipecat#1694, livekit.com/blog/prompting-voice-agents-to-sound-more-realistic).
- Prior art: The-Focus-AI/hermes-body LLM-brains survey; huggingface.co/blog/local-reachy-mini-conversation
  (Silero+Parakeet+separate-process pattern); brevdev/reachy-personal-assistant.
