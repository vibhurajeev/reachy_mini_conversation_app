"""Stream-aware action-directive parsing and sentence chunking (plan D8/D9).

Hermes embeds directives like ⟦emotion:happy⟧, ⟦dance⟧, ⟦look:left⟧, or
⟦ignore⟧ in its reply stream. The parser separates them from speakable text
even when a directive is split across stream chunks, and the scrubber
guarantees nothing tag-shaped ever reaches TTS.
"""

import re
import logging
from dataclasses import field, dataclass


logger = logging.getLogger(__name__)

OPEN = "⟦"
CLOSE = "⟧"
_TAG_RE = re.compile(r"⟦([^⟧]{0,80})⟧")

#: Directive names the robot understands; anything else is logged and dropped.
KNOWN_ACTIONS = ("emotion", "dance", "look", "ignore")

_SENTENCE_END_RE = re.compile(r"([.!?…])(\s|$)")


@dataclass
class Directive:
    """One parsed action directive from the Hermes stream."""

    name: str
    argument: str | None = None


@dataclass
class StreamActionParser:
    """Incremental parser: feed text deltas, get (speakable_text, directives)."""

    _carry: str = ""

    def feed(self, delta: str) -> tuple[str, list[Directive]]:
        """Consume a stream delta; return clean speakable text and directives."""
        buffer = self._carry + delta
        directives: list[Directive] = []

        def _collect(match: re.Match[str]) -> str:
            """Record a matched directive and strip it from the text."""
            directive = _parse_directive(match.group(1))
            if directive is not None:
                directives.append(directive)
            return ""

        cleaned = _TAG_RE.sub(_collect, buffer)

        # Hold back any trailing unterminated directive across chunk boundaries.
        open_index = cleaned.rfind(OPEN)
        if open_index != -1 and CLOSE not in cleaned[open_index:]:
            self._carry = cleaned[open_index:]
            cleaned = cleaned[:open_index]
        else:
            self._carry = ""

        return cleaned, directives

    def flush(self) -> str:
        """Return and clear held-back text at end of stream, scrubbed of tags."""
        remainder = self._carry
        self._carry = ""
        # A dangling half-tag at stream end must never be spoken.
        return "" if remainder.startswith(OPEN) else remainder


def _parse_directive(body: str) -> Directive | None:
    """Parse '⟦name:arg⟧' bodies; unknown names are logged and dropped."""
    name, _, raw_argument = body.strip().partition(":")
    name = name.strip().lower()
    argument: str | None = raw_argument.strip() or None
    if name not in KNOWN_ACTIONS:
        logger.info("Ignoring unknown action directive: %r", body)
        return None
    return Directive(name=name, argument=argument)


@dataclass
class SentenceChunker:
    """Accumulate streamed text and release complete sentences for TTS."""

    _buffer: str = field(default="")

    def feed(self, text: str) -> list[str]:
        """Add text; return any completed sentences."""
        self._buffer += text
        sentences: list[str] = []
        while True:
            match = _SENTENCE_END_RE.search(self._buffer)
            if match is None:
                break
            end = match.end(1)
            sentence = self._buffer[:end].strip()
            self._buffer = self._buffer[end:]
            if sentence:
                sentences.append(sentence)
        return sentences

    def flush(self) -> str:
        """Return and clear any trailing partial sentence."""
        remainder = self._buffer.strip()
        self._buffer = ""
        return remainder
