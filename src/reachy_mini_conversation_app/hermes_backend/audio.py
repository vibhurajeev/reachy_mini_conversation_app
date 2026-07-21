"""Small audio helpers shared by the Hermes backend (mono-ize, resample)."""

import numpy as np
from numpy.typing import NDArray

from reachy_mini_conversation_app.streaming import audio_to_int16


def to_mono_int16(audio: NDArray[np.int16] | NDArray[np.float32]) -> NDArray[np.int16]:
    """Collapse a frame to mono int16 using the app's channels-last convention.

    Mirrors the downmix in HuggingFaceRealtimeHandler.receive and the play loop:
    the Reachy media stack delivers channels-last (N, C) frames, e.g. (256, 2);
    transpose if channels-first, then take the first channel (the daemon's
    beamformed primary), and cast via the shared streaming helper.
    """
    data = audio
    if data.ndim == 2:
        if data.shape[1] > data.shape[0]:
            data = data.T
        if data.shape[1] > 1:
            data = data[:, 0]
    return audio_to_int16(data)


def resample_int16(audio: NDArray[np.int16], src_rate: int, dst_rate: int) -> NDArray[np.int16]:
    """Linearly resample int16 mono audio between sample rates (identity when equal)."""
    if src_rate == dst_rate or audio.size == 0:
        return audio
    duration = audio.shape[-1] / float(src_rate)
    dst_len = max(1, int(round(duration * dst_rate)))
    src_x = np.linspace(0.0, duration, num=audio.shape[-1], endpoint=False)
    dst_x = np.linspace(0.0, duration, num=dst_len, endpoint=False)
    resampled = np.interp(dst_x, src_x, audio.astype(np.float32))
    return resampled.astype(np.int16)


def float32_to_int16(audio: NDArray[np.float32]) -> NDArray[np.int16]:
    """Convert float32 [-1, 1] samples to int16."""
    return (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
