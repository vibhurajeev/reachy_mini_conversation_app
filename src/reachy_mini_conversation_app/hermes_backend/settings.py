"""Environment-driven settings for the Hermes gateway backend.

Kept separate from the upstream config module so divergence stays additive.
Values are read once at handler construction; restart the backend to apply
changes (same semantics as the upstream profile switch).
"""

import os
from dataclasses import field, dataclass


def _env_float(name: str, default: float) -> float:
    """Read a float env var, falling back to the default on absence or garbage."""
    raw = os.getenv(name, "")
    try:
        return float(raw) if raw else default
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    """Read an int env var, falling back to the default on absence or garbage."""
    raw = os.getenv(name, "")
    try:
        return int(raw) if raw else default
    except ValueError:
        return default


@dataclass(frozen=True)
class HermesSettings:
    """All knobs for the Hermes gateway backend, sourced from the environment."""

    # Transport (D1/D13): Hermes api_server, localhost when co-located.
    api_url: str = field(default_factory=lambda: os.getenv("HERMES_API_URL", "http://127.0.0.1:8642"))
    api_key: str = field(default_factory=lambda: os.getenv("HERMES_API_KEY", ""))
    session_key: str = field(default_factory=lambda: os.getenv("HERMES_SESSION_KEY", "agent:main:robot:office"))
    model: str = field(default_factory=lambda: os.getenv("HERMES_MODEL", "hermes"))

    # Latency ceilings and canned-phrase behavior (D7).
    first_token_timeout_s: float = field(default_factory=lambda: _env_float("HERMES_FIRST_TOKEN_TIMEOUT_S", 45.0))
    total_timeout_s: float = field(default_factory=lambda: _env_float("HERMES_TOTAL_TIMEOUT_S", 120.0))
    progress_quip_after_s: float = field(default_factory=lambda: _env_float("HERMES_PROGRESS_QUIP_AFTER_S", 10.0))

    # Output shaping (D8).
    max_sentences: int = field(default_factory=lambda: _env_int("HERMES_MAX_SENTENCES", 6))

    # Always-on guards (D2/D10).
    forwards_per_minute: int = field(default_factory=lambda: _env_int("HERMES_FORWARDS_PER_MIN", 10))
    min_transcript_words: int = field(default_factory=lambda: _env_int("HERMES_MIN_WORDS", 2))

    # Liveliness (D7): emotion intent fired while Hermes thinks.
    thinking_emotion: str = field(default_factory=lambda: os.getenv("HERMES_THINKING_EMOTION", "curious"))

    # VAD (Silero) endpointing.
    vad_threshold: float = field(default_factory=lambda: _env_float("VAD_THRESHOLD", 0.5))
    vad_min_silence_ms: int = field(default_factory=lambda: _env_int("VAD_MIN_SILENCE_MS", 600))
    vad_speech_pad_ms: int = field(default_factory=lambda: _env_int("VAD_SPEECH_PAD_MS", 100))

    # STT (faster-whisper).
    stt_model: str = field(default_factory=lambda: os.getenv("STT_MODEL", "small"))
    stt_device: str = field(default_factory=lambda: os.getenv("STT_DEVICE", "cpu"))
    stt_compute_type: str = field(default_factory=lambda: os.getenv("STT_COMPUTE", "int8"))
    stt_language: str = field(default_factory=lambda: os.getenv("STT_LANGUAGE", "en"))

    # TTS engine selection and voices.
    tts_engine: str = field(default_factory=lambda: os.getenv("TTS_ENGINE", "kokoro"))
    kokoro_model_path: str = field(default_factory=lambda: os.getenv("KOKORO_MODEL_PATH", "kokoro-v1.0.onnx"))
    kokoro_voices_path: str = field(default_factory=lambda: os.getenv("KOKORO_VOICES_PATH", "voices-v1.0.bin"))
    tts_voice: str = field(default_factory=lambda: os.getenv("TTS_VOICE", "af_sarah"))
    piper_model_path: str = field(default_factory=lambda: os.getenv("PIPER_MODEL_PATH", ""))
    piper_sample_rate: int = field(default_factory=lambda: _env_int("PIPER_SAMPLE_RATE", 22050))

    # Output rate expected by the robot media player (play_loop never resamples).
    player_sample_rate: int = field(default_factory=lambda: _env_int("PLAYER_SAMPLE_RATE", 16000))

    # Bridge split (D13v2): the thin on-robot handler talks to this service.
    bridge_url: str = field(default_factory=lambda: os.getenv("BRIDGE_URL", "http://127.0.0.1:8643"))
    bridge_api_key: str = field(default_factory=lambda: os.getenv("BRIDGE_API_KEY", ""))
