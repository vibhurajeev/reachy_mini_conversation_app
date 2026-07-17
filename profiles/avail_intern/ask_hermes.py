"""Bridge tool: delegate a task to the Hermes agent running on the sandbox VM.

The robot's realtime voice model handles chit-chat and motion; anything that
needs company Slack, docs, research, or multi-step work goes to Hermes over
SSH. Requires key-based SSH from the machine running this app to the VM:

    HERMES_SSH_HOST=user@hermes-vm     (required, e.g. in the app's .env)
    HERMES_TIMEOUT_S=180               (optional)

Test the transport manually first:
    echo "say hi" | ssh -o BatchMode=yes $HERMES_SSH_HOST bash -lc hermes
"""

import os
import asyncio
import logging
from typing import Any, Dict

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies

logger = logging.getLogger(__name__)


class AskHermes(Tool):
    """Send a task to the Avail Intern's backend brain (Hermes on the VM)."""

    name = "ask_hermes"
    description = (
        "Delegate a task to HQ (the Hermes agent): anything involving Slack, "
        "company docs, web research, scheduling, or multi-step work. Returns "
        "Hermes's answer as text. Takes up to a few minutes for big tasks — "
        "tell the user you're on it before calling this."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": (
                    "The full task or question for Hermes, self-contained, "
                    "with any names/channels/details the user mentioned."
                ),
            },
        },
        "required": ["task"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        task = (kwargs.get("task") or "").strip()
        if not task:
            return {"error": "task must be a non-empty string"}

        host = os.getenv("HERMES_SSH_HOST", "").strip()
        if not host:
            return {"error": "HERMES_SSH_HOST is not configured on this machine"}

        timeout_s = float(os.getenv("HERMES_TIMEOUT_S", "180"))
        logger.info("ask_hermes: %s", task[:120])

        proc = await asyncio.create_subprocess_exec(
            "ssh",
            "-o", "BatchMode=yes",
            "-o", "ConnectTimeout=10",
            host,
            "bash", "-lc", "hermes",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(input=task.encode()), timeout=timeout_s
            )
        except asyncio.TimeoutError:
            proc.kill()
            return {"error": f"Hermes did not answer within {int(timeout_s)}s"}

        if proc.returncode != 0:
            err = stderr.decode(errors="replace").strip()[-500:]
            logger.error("ask_hermes failed rc=%s: %s", proc.returncode, err)
            return {"error": f"Hermes call failed: {err or 'unknown error'}"}

        answer = stdout.decode(errors="replace").strip()
        if not answer:
            return {"error": "Hermes returned an empty answer"}
        return {"result": answer[-4000:]}
