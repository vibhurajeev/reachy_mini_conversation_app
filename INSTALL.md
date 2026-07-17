# Avail Intern — Installation Guide

End-to-end setup for the Avail Intern: a Reachy Mini Wireless (Pi 5) running this app
as its voice/body layer, bridged to a Hermes Agent "brain" on a sandboxed VM with
company Slack access.

```
you ──voice──► Reachy Mini (Pi 5: thin app, VAD+motion) ──http──► VM: bridge (STT→TTS) ──► Hermes ──► Slack
               realtime voice model                               Kimi K3 via OpenRouter
               motion / camera / emotions                         memory, tools, MCP
```

Two credentials are needed in total: an **OpenRouter API key** (the brain's model) and
a **Slack app token pair** (the brain's Slack access). The robot itself holds no
company credentials — only an SSH key to the VM.

---

## Part A — Hermes VM (the brain)

Use a dedicated, sandboxed VM (cloud VPS or local hypervisor). Treat it as
single-purpose: no other credentials or checkouts on it.

```bash
# 1. Prereqs
sudo apt update && sudo apt install -y git curl ffmpeg

# 2. Install Hermes Agent (official installer; skips the interactive wizard)
curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash -s -- --skip-setup
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc && source ~/.bashrc

# 3. Model: Kimi K3 via OpenRouter (get a key at https://openrouter.ai/keys)
hermes config set model.default moonshotai/kimi-k3
hermes config set OPENROUTER_API_KEY sk-or-v1-...

# 4. Verify
hermes doctor                      # expect: ✓ OpenRouter API
echo "say hi in one sentence" | hermes
```

Install the personality: copy this repo's soul file (kept at `~/avail/reachy/SOUL.md`
on the workstation, or write your own) to `~/.hermes/SOUL.md` on the VM.

Gotchas we hit:
- `hermes doctor` complaining about PATH → the `.bashrc` line above fixes it.
- `HTTP 401 invalid x-api-key` against Anthropic → a stale/garbled `ANTHROPIC_*` value
  in `~/.hermes/.env`; clear with `hermes config set ANTHROPIC_API_KEY ""` and
  `hermes config set ANTHROPIC_TOKEN ""`. Note: Claude Pro/Max subscription login is
  not a valid credential for third-party agents — use a real API key or OpenRouter.

## Part B — Slack (the brain's office presence)

1. Generate the app manifest: `hermes slack manifest --agent-view` (a pre-branded copy
   lives at `~/avail/reachy/slack-app-manifest.json`).
2. At https://api.slack.com/apps → **Create New App → From an app manifest** → pick the
   workspace → paste the manifest JSON.
3. **Settings → Socket Mode** → enable; create an app-level token with
   `connections:write` → copy the `xapp-...` token.
4. **Settings → Install App → Install to Workspace** → copy the `xoxb-...` bot token.
5. Get your Slack member ID (profile → ⋮ → Copy member ID).
6. On the VM, add to `~/.hermes/.env`:
   ```
   SLACK_BOT_TOKEN=xoxb-...
   SLACK_APP_TOKEN=xapp-...
   SLACK_ALLOWED_USERS=U...        # deny-by-default; list allowed member IDs
   ```
7. `hermes gateway install` — registers the gateway as a systemd service (survives
   reboots). Invite the bot to channels with `/invite @Avail Intern`.

Socket Mode is outbound-only: the VM needs no public inbound port.

## Part C — Bridge service (on the VM, next to Hermes)

The bridge does STT (Whisper) → Hermes (localhost) → TTS (Kokoro) on the
robot's behalf, streaming back ready-to-play audio events. It is the only
component the robot talks to.

```bash
cd ~/reachy/reachy_mini_conversation_app
uv sync && uv pip install -r requirements-hermes.txt
mkdir -p models && cd models   # Kokoro voice models, once (~340 MB)
curl -sLO https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
curl -sLO https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin
cd ..
cat >> .env <<'ENV'
HERMES_API_URL=http://127.0.0.1:8642
HERMES_API_KEY=<the API_SERVER_KEY from Part A>
HERMES_SESSION_KEY=agent:main:robot:office
KOKORO_MODEL_PATH=models/kokoro-v1.0.onnx
KOKORO_VOICES_PATH=models/voices-v1.0.bin
ENV
echo "BRIDGE_API_KEY=$(openssl rand -hex 32)" >> .env   # copy this for Part D
uv run avail-intern-bridge                    # :8643; first run downloads Whisper (~500 MB)
curl -s http://127.0.0.1:8643/health          # expect {"status":"ok","hermes":true}
```

## Part D — Robot (Reachy Mini Wireless, thin app)

The robot runs only mic capture + VAD + motion; no models, no torch.

1. Power the robot, join it to the office Wi-Fi (its dashboard handles this).
2. From a machine with this repo, install the fork into the robot's app venv:
   ```bash
   scp -r . pollen@reachy-mini.local:/tmp/avail_intern
   ssh pollen@reachy-mini.local "/venvs/apps_venv/bin/pip install /tmp/avail_intern onnxruntime"
   ```
3. Give the app its env (inherited from the daemon) on the Pi:
   `sudo systemctl edit reachy-mini-daemon` and add:
   ```
   [Service]
   Environment=CONVERSATION_BACKEND=bridge
   Environment=BRIDGE_URL=http://<vm-ip>:8643
   Environment=BRIDGE_API_KEY=<the key from Part C>
   Environment=REACHY_MINI_CUSTOM_PROFILE=avail_intern
   ```
   then `sudo systemctl restart reachy-mini-daemon`.
4. Start the app from the dashboard — or make it the default wake-up
   experience with the daemon's `--startup-app reachy_mini_conversation_app`
   (touching an antenna then wakes the robot straight into the intern).

## Part E — First conversation

Smoke tests, in order of what they exercise:

1. "Hey intern, how's it going?" — VAD → bridge → Hermes → voice round trip.
2. Talk to a colleague near the robot — expect silence (the `⟦ignore⟧` gate).
3. "Intern, do a little dance." — the directive channel.
4. DM the bot in Slack (once Part B is done) — same brain, different surface.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Robot plays the canned "HQ's not picking up" line | bridge unreachable from the Pi: check `BRIDGE_URL`, the VM firewall on :8643, and that `avail-intern-bridge` is running |
| Bridge log shows 401s | `BRIDGE_API_KEY` mismatch between Pi env and bridge `.env` |
| Bridge `/health` shows `"hermes": false` | Hermes gateway not running or api_server disabled — see Part A |
| Robot answers everything it overhears | the SOUL.md robot-channel rules are missing — Part A step 3 |
| 401 from provider on VM | stale `ANTHROPIC_*` entries — see Part A gotchas |
| Bot silent in Slack channels | missing `channels:history` scope or bot not invited; also check `SLACK_ALLOWED_USERS` |
