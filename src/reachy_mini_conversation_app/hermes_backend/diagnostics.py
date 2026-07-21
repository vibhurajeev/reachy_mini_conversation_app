"""Diagnostics for the Hermes/bridge backends: crash visibility + tracing.

Fire-and-forget asyncio tasks (turn handling, thinking motion, action tools)
otherwise fail silently — their exceptions only surface as "Task exception was
never retrieved" at GC time, or not at all. `supervise` attaches a done
callback that logs a full traceback the moment such a task dies, and
`install_loop_exception_handler` catches everything else the loop swallows.
"""

import asyncio
import logging
from typing import Any
from collections.abc import Coroutine


logger = logging.getLogger("reachy_mini_conversation_app.hermes_backend")


def install_loop_exception_handler(loop: asyncio.AbstractEventLoop) -> None:
    """Route unhandled event-loop exceptions through the logger with tracebacks."""

    def handler(loop: asyncio.AbstractEventLoop, context: dict[str, Any]) -> None:
        """Log the loop-level exception context (never swallow it silently)."""
        message = context.get("message", "unhandled event loop error")
        exception = context.get("exception")
        if exception is not None:
            logger.error("Event loop error: %s", message, exc_info=exception)
        else:
            logger.error("Event loop error: %s | context=%r", message, context)

    loop.set_exception_handler(handler)
    logger.debug("Loop exception handler installed")


def supervise(coro: Coroutine[Any, Any, Any], *, name: str) -> "asyncio.Task[Any]":
    """Schedule a fire-and-forget task whose failure is logged, not lost."""
    task = asyncio.create_task(coro, name=name)

    def _done(finished: "asyncio.Task[Any]") -> None:
        """Log any exception (other than a normal cancel) from the task."""
        if finished.cancelled():
            logger.debug("Task %s cancelled", name)
            return
        error = finished.exception()
        if error is not None:
            logger.error("Task %s crashed", name, exc_info=error)

    task.add_done_callback(_done)
    return task
