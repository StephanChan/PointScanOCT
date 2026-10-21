"""背景光谱与可选 X 归一化的加载。

与主程序 ``ThreadGPU.update_background`` 的约定一致：背景是 float32 光谱，
长度等于光谱采样数时按 ``(1, samples)`` 处理；长度等于 ``x_pixels × samples``
时按 X 求均值退化成一条光谱背景（兼容旧的 X-by-spectral 背景文件）。
"""

import os

import numpy as np

from .._log import get_logger
from ..errors import CalibrationError


def load_spectral_background(path, samples, x_pixels=None):
    """读取光谱背景，返回 ``(1, samples)`` 的 float32 数组。"""
    samples = int(samples)
    path = str(path or "").strip()
    if not path:
        raise CalibrationError("background path is empty")
    if not os.path.isfile(path):
        raise CalibrationError(f"background file not found: {path}")
    raw = np.fromfile(path, dtype=np.float32)
    if raw.size == samples:
        background = raw.reshape(1, samples)
        kind = "line-scan spectral background"
    elif x_pixels and raw.size == int(x_pixels) * samples:
        background = raw.reshape(int(x_pixels), samples).mean(axis=0, keepdims=True,
                                                              dtype=np.float32)
        kind = "legacy X-by-spectral background averaged to line-scan background"
    else:
        raise CalibrationError(
            "background size mismatch: "
            f"{raw.size} values, expected {samples}"
            + (f" or {int(x_pixels) * samples} (x_pixels={x_pixels})" if x_pixels else "")
            + f": {path}"
        )
    background = np.ascontiguousarray(background, dtype=np.float32)
    get_logger("background").info("loaded %s from %s", kind, path)
    return background


def load_x_normalization(path, x_pixels):
    """读取每个 X 位置的归一化系数（float32，长度 = 有效 X 位置数）；不可用返回 None。"""
    path = str(path or "").strip()
    if not path:
        return None
    if not os.path.isfile(path):
        raise CalibrationError(f"X normalization file not found: {path}")
    raw = np.fromfile(path, dtype=np.float32)
    x_pixels = int(x_pixels)
    if raw.size != x_pixels:
        raise CalibrationError(
            f"X normalization size mismatch: {raw.size} values, expected {x_pixels}: {path}"
        )
    if not np.all(np.isfinite(raw)) or np.any(raw == 0):
        raise CalibrationError(f"X normalization has non-finite or zero entries: {path}")
    get_logger("background").info("loaded X normalization from %s", path)
    return np.ascontiguousarray(raw, dtype=np.float32)
