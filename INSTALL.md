# Avail Intern — Installation Guide

End-to-end setup for the Avail Intern: a Reachy Mini Wireless (Pi 5) running this app
as its voice/body layer, bridged to a Hermes Agent "brain" on a sandboxed VM with
company Slack access.

```
you ──voice──► Reachy Mini (Pi 5, this app) ──ssh (ask_hermes)──► Hermes VM ──► Slack
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

## Part C — Robot (Reachy Mini Wireless, Pi 5)

1. Power the robot, join it to Wi-Fi, open its dashboard (`http://reachy-mini.local`),
   and SSH into the Pi.
2. Install this fork on the Pi:
   ```bash
   git clone https://github.com/vibhurajeev/reachy_mini_conversation_app.git
   cd reachy_mini_conversation_app
   uv sync
   ```
3. Configure `.env` in the checkout:
   ```
   REACHY_MINI_CUSTOM_PROFILE=avail_intern
   HERMES_SSH_HOST=<user>@<vm-host>
   # optional: HERMES_TIMEOUT_S=180
   ```
   The `avail_intern` profile ships in this repo's `profiles/` dir, so no external
   profiles directory is needed. The voice backend is Hugging Face's realtime
   endpoint (the app's only backend; free `deployed` mode is the default —
   `HF_REALTIME_CONNECTION_MODE=local` + `HF_REALTIME_WS_URL` can point at a
   self-hosted speech-to-speech server later). See ARCHITECTURE.md.

## Part D — The bridge (robot → brain)

On the Pi:

```bash
ssh-keygen -t ed25519 -N ""
# append ~/.ssh/id_ed25519.pub to ~/.ssh/authorized_keys on the VM, then:
echo "say hi" | ssh -o BatchMode=yes <user>@<vm-host> bash -lc hermes
```

That command is exactly what the `ask_hermes` tool runs. If it answers, the bridge
works. If the VM is not publicly reachable from the office network, put the Pi and the
VM on a shared Tailscale tailnet instead of opening ports.

## Part E — First conversation

Start the robot daemon, then launch the app (`reachy-mini-conversation-app`, or via
the dashboard). Smoke tests, in order of what they exercise:

1. "Hey, how's it going?" — voice loop + personality.
2. "What am I holding?" — camera tool (vision).
3. "Ask HQ to summarize what Avail Nexus is." — full SSH → Hermes → Kimi round trip.
4. DM the bot in Slack — same brain, same memory, different surface.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Robot talks but `ask_hermes` errors "HERMES_SSH_HOST is not configured" | `.env` not loaded or var missing on the Pi |
| `ask_hermes` times out | VM unreachable from Pi's network (use Tailscale), or first-token latency > `HERMES_TIMEOUT_S` |
| Hermes answers in Slack but not via robot | SSH key not authorized on VM — rerun Part D |
| 401 from provider on VM | stale `ANTHROPIC_*` entries — see Part A gotchas |
| Bot silent in Slack channels | missing `channels:history` scope or bot not invited; also check `SLACK_ALLOWED_USERS` |
