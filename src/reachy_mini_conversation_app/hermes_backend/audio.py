"""Small audio helpers shared by the Hermes backend (mono-ize, resample)."""

import numpy as np
from numpy.typing import NDArray


def to_mono_int16(audio: NDArray[np.int16] | NDArray[np.float32]) -> NDArray[np.int16]:
    """Collapse a (C, N) or (N,) frame to mono int16 samples."""
    data = np.asarray(audio)
    if data.ndim == 2:
        # Channels-first (C, N) as produced by the media stack; average channels.
        data = data.mean(axis=0)
    if data.dtype == np.float32 or data.dtype == np.float64:
        data = np.clip(data, -1.0, 1.0) * 32767.0
    return data.astype(np.int16, copy=False)


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
