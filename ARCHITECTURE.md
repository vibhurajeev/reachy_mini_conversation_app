# Architecture — how this app works

A map of the upstream app for anyone changing core code. Derived from a full source
read of this fork (July 2026). Paths are relative to `src/reachy_mini_conversation_app/`
unless noted.

**Mental model:** one robot audio pipeline feeds a realtime voice model (Hugging
Face's OpenAI-compatible Realtime endpoint) over a websocket; the model drives speech
and function calls; function calls dispatch to async Python tools; a separate 60 Hz
thread animates the robot. Vision today is strictly pull-based — the model calls a
`camera` tool when it wants to look.

```
mic ─► LocalStream.record_loop ─► HuggingFaceRealtimeHandler ─► HF realtime WS
                                        │  function calls              │
speaker ◄─ play_loop ◄─ output_queue ◄──┘  audio deltas ◄──────────────┘
                    │
                    ▼ (tool calls, backgrounded)
        BackgroundToolManager ─► Tool.__call__(deps, ...)
                    │
                    ▼ (motion commands via Queue)
        MovementManager 60 Hz thread ─► robot.set_target(...)
```

## Entry points & lifecycle

- Console script `reachy-mini-conversation-app` → `main.py:main()`.
- Robot-daemon app: `ReachyMiniConversationApp(ReachyMiniApp)` (`main.py:340`), which
  the daemon runs with its robot handle and stop event.

`run()` (`main.py:79`): load `<instance_path>/.env` (override=True) → startup settings
→ connect `ReachyMini` (exits with troubleshooting on timeout) → wake robot → build
`MovementManager` + `ToolDependencies` → build handler → `LocalStream` → optional
uvicorn web UI on `:7860` → `initialize_tools` → start movement thread → enable
speech wobble → `stream_manager.launch()` (blocks). Shutdown goes to sleep, disables
wobble, closes media + client.

## One voice turn

1. `LocalStream.record_loop` (`console.py:861`) pulls mic samples from
   `robot.media.get_audio_sample()` → `handler.receive()`.
2. `HuggingFaceRealtimeHandler.receive` (`huggingface_realtime.py:939`) base64s PCM
   into the websocket. Turn-taking is **server-side VAD** (`interrupt_response=True`).
3. The receive loop (`_run_realtime_session`, `:736`) handles events: speech started
   (barge-in: clears audio queue), transcription deltas, response audio deltas →
   `output_queue`.
4. `play_loop` (`console.py:872`) → `robot.media.push_audio_sample()`. The daemon's
   wobbler taps this to nod along with speech.
5. Barge-in: server interrupts the response; client drains `output_queue` **in place**
   (`_drain_output_queue`, `console.py:850`) since `emit()` awaits that same object.

## Backend (singular)

`ConversationHandler` (`conversation_handler.py:26`) is the abstract seam for
backends, but the **only implementation** is `HuggingFaceRealtimeHandler`. It uses the
`openai` SDK against HF's OpenAI-compatible realtime endpoint. Legacy
`BACKEND_PROVIDER`/`MODEL_NAME` env vars are warned about and ignored
(`config.py:90`). Two connection modes: `deployed` (default; session proxy returns a
connect URL — free) and `local` (`HF_REALTIME_WS_URL`, self-hosted speech-to-speech).
Only one model response may be active at a time — `_response_sender_loop` (`:478`)
serializes all `response.create()` calls; anything that makes the robot speak must go
through `_safe_response_create()`.

## Motion system

`MovementManager` (`moves.py:167`) runs a 60 Hz daemon thread (`working_loop`,
`:722`) — the *sole* writer of `robot.set_target`. All control flows through a
thread-safe command queue. Per tick it blends three layers:

1. **Primary move** from a sequential `move_queue` (dances, emotions, goto,
   breathing) — mutually exclusive, popped when elapsed ≥ duration.
2. **Head-tracking anchor overlay** — emotions compose onto the tracked anchor;
   the actual face-following math lives in the robot daemon, not here.
3. **Antenna listening blend** — antennas freeze while listening, blend back in 0.4 s.

Dances come from `reachy_mini_dances_library` (import time); emotions from the
`pollen-robotics/reachy-mini-emotions-library` HF dataset (lazy, first
`play_emotion` call downloads it). Speech wobble is composited by the SDK/daemon
downstream of `set_target`.

## Tool system

- `Tool` ABC (`tools/core_tools.py:60`): `name`, `description`, `parameters_schema`,
  `async __call__(self, deps, **kwargs) -> dict`. `needs_response=False` suppresses
  the spoken follow-up. First arg is `ToolDependencies` (robot handle, movement
  manager, instance path, camera flag, `go_to_sleep`).
- **Loading** (`initialize_tools`, `:487`): profile `tools.txt` → remote MCP tools →
  per-name resolution: profile-local `.py` → shared `tools/` module → external dir.
  `task_status` + `task_cancel` are always appended. Duplicate names raise.
- **Every tool call runs backgrounded** as its own asyncio task via
  `BackgroundToolManager` — long tools (like our `ask_hermes`) never block audio.
  Only the two system tools can see the manager (introspect/cancel); ordinary tools
  can't touch each other.

## Profiles & personality

Profile dir = `instructions.txt` (system prompt) + `tools.txt` + optional
`voice.txt`, `greeting.txt`, and tool `.py` files. Three sources: built-in
(`profiles/`), external (`REACHY_MINI_EXTERNAL_PROFILES_DIRECTORY`), user-created
(web UI, under the instance path). Active profile: `LOCKED_PROFILE` constant →
`REACHY_MINI_CUSTOM_PROFILE` env → persisted startup settings → `default`.
`get_session_instructions` (`prompts.py:42`) = memory block **prepended** to the
profile's `instructions.txt`; tools are passed to the session separately, not in the
prompt. Applying a personality re-inits tools and restarts the realtime session.

Our profile: `profiles/avail_intern/` — persona + `ask_hermes` (SSH bridge to the
Hermes agent VM; env `HERMES_SSH_HOST`, `HERMES_TIMEOUT_S`).

## Idle policy

The trigger lives in `ConversationHandler.emit()` (`conversation_handler.py:62`),
checked every audio tick: fires only when idle > 180 s, ≥180 s since last idle act,
backend not mid-response, and `movement_manager.is_idle()`. Actions are chosen by
weighted random (`idle_policy.py:68`: do-nothing 0.60, dance 0.16, emotion 0.16,
look 0.08), run as background tools, and **never involve the model** — results are
not sent to it.

## Camera / vision

One tool (`tools/camera.py`): grabs a JPEG via
`deps.reachy_mini.media.get_frame_jpeg()`, returns base64. The handler strips the
image from the tool-result echo and injects it as a synthesized **user message**
with an `input_image` data-URI (`huggingface_realtime.py:639-660`). One-shot,
model-initiated; nothing captures on a timer.

## Memory

`memory.py`: `memory.v1.json` (instance path or XDG data dir), ≤60 facts ×
280 chars, atomic writes, case-insensitive dedupe. `remember`/`forget` tools write
it; the whole block is prepended to the system prompt at session start — so new
memories take effect on the next session restart, not mid-conversation.

## Config (env vars that matter)

| Var | Meaning |
|---|---|
| `HF_REALTIME_CONNECTION_MODE` | `deployed` (default, free proxy) or `local` |
| `HF_REALTIME_WS_URL` | websocket target in local mode |
| `HF_TOKEN` | optional bearer for gated assets |
| `REALTIME_TRANSCRIPTION_LANGUAGE` | STT language (default `en`) |
| `REACHY_MINI_CUSTOM_PROFILE` | active profile name (e.g. `avail_intern`) |
| `REACHY_MINI_EXTERNAL_PROFILES_DIRECTORY` | extra profiles dir |
| `REACHY_MINI_EXTERNAL_TOOLS_DIRECTORY` (+`AUTOLOAD_EXTERNAL_TOOLS`) | extra tools |
| `REACHY_MINI_APP_TIMEOUT_MINUTES` | inactivity shutdown (default 1440; ≤0 off) |
| `HERMES_SSH_HOST`, `HERMES_TIMEOUT_S` | **fork-only** — ask_hermes bridge |

`.env` is loaded with `override=True` from the instance path at startup.

## Concurrency

Threads: main (asyncio audio loop), movement (60 Hz), optional uvicorn UI,
watchdogs. Within the audio loop: startup/reconnect loop, record, play, response
sender, tool-manager listener/cleanup, one task per tool call. Invariants: one
active model response at a time; tools never block the receive loop; motion never
blocks audio; barge-in drains the audio queue in place.

## Extension points (phase-2 roadmap)

- **Ambient vision loop** (react unprompted): new background task capturing frames
  via `media.get_frame_jpeg()` on a cadence — do *not* route through the `camera`
  tool. Gate on the idle machinery (`ConversationHandler.emit()` branch +
  `_idle_behavior_ready`) so it never fights a live turn. To make the robot speak
  unprompted: inject an `input_image` user item (pattern at
  `huggingface_realtime.py:649`) then `_safe_response_create()` — the startup
  greeting (`:448-474`) is a working template for model-speaks-first. Build as new
  modules + minimal hooks per CLAUDE.md.
- **Wake-word gate**: client-side, in `LocalStream.record_loop` (`console.py:861`)
  — gate `handler.receive()` or drive the existing `_mic_muted` flag. (Turn
  detection is all server-side VAD, so the gate must be before audio is sent.)

## Gotchas & flags

- Control loop is **60 Hz**; some docstrings still say 100 Hz.
- `stop_dance` and `stop_emotion` are identical — both clear the whole primary queue.
- Wobble + face-tracking math live in the daemon/SDK, not this repo.
- Upstream CI includes HF-Space sync workflows that will fail or need
  secrets/disabling on this fork.
- Python ≥3.11; `uv.lock` is checked (`uv-lock-check` CI); lint is ruff with strict
  pydocstyle, mypy strict on `src/`.
