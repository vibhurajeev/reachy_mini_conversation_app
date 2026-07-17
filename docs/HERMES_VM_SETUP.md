# Hermes VM setup for the robot gateway

VM-side configuration for the Hermes gateway backend (plan D1/D2/D13). These
steps run **on the Hermes VM** (user-run — SSH access is Vibhu's).

## 1. Enable the api_server platform

Add to `~/.hermes/.env`:

```
API_SERVER_ENABLED=1
API_SERVER_KEY=<generate a long random secret>
# Optional: API_SERVER_HOST=127.0.0.1  API_SERVER_PORT=8642
```

Restart the gateway (`hermes gateway restart` or the systemd service). Verify:

```bash
curl -s http://127.0.0.1:8642/health
```

The app (running on the same server per D13) points at it with:

```
CONVERSATION_BACKEND=hermes
HERMES_API_URL=http://127.0.0.1:8642
HERMES_API_KEY=<same secret>
HERMES_SESSION_KEY=agent:main:robot:office
```

If the app runs elsewhere (fallback: on the Pi), do NOT expose the port —
use Tailscale or an SSH tunnel, and keep `API_SERVER_HOST=127.0.0.1`.

## 2. Restrict the robot channel's toolset (mandatory — plan D2)

Ambient office speech reaches this channel. It must have no side-effect tools
so a misheard utterance can never post to Slack. In `~/.hermes/config.yaml`:

```yaml
platform_toolsets:
  api_server: [web_search]   # read-only; adjust to taste, never Slack/send tools
```

Verify: ask the robot to "post hi to #general" — Hermes must refuse/fail.

## 3. Teach Hermes the robot channel rules (SOUL.md)

Append to `~/.hermes/SOUL.md`:

```markdown
## Robot channel (api_server)

Messages on this channel come from your robot body's microphone in the Avail
office. They are raw speech-to-text: expect transcription noise, and expect
speech that was never addressed to you at all.

- Decide first: was this addressed to you? You're addressed when named
  (intern, Reachy), directly asked, or continuing an exchange from the last
  minute. If not addressed, reply with exactly: ⟦ignore⟧ — nothing else.
- Spoken replies: 1-3 short sentences, no markdown, no links, no emoji.
  Complaining-then-delivering stays your signature; brevity is mandatory.
- Body language: you may embed at most 2 directives anywhere in a reply:
  ⟦emotion:NAME⟧ (happy, excited, sad, curious, yes, no, random),
  ⟦dance⟧, ⟦look:left|right|up|down|front⟧.
  Never mention the directives in speech; they are stage directions.
- You cannot see through the camera on this channel. If asked what you see,
  say so gracefully and move on.
- Never speak credentials or private data aloud; the room may have guests.
```

## 4. Smoke test end to end

From the server (with the app's venv active is not required):

```bash
curl -s http://127.0.0.1:8642/v1/chat/completions \
  -H "Authorization: Bearer $API_SERVER_KEY" \
  -H "X-Hermes-Session-Key: agent:main:robot:office" \
  -H "Content-Type: application/json" \
  -d '{"model":"hermes","stream":false,"messages":[{"role":"user","content":"intern, say hi in one sentence"}]}'
```

Expect an in-character reply. Then repeat with an unaddressed sentence
("so anyway the quarterly numbers look fine") and expect `⟦ignore⟧`.
