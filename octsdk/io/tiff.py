"""TIFF 写盘：单文件多帧，复数帧自动写成"幅度 + 相位"交错。

与主程序一致：新建文件用 BigTIFF（64 位偏移），避免超过 4 GiB 时
``struct.error: argument out of range``。
"""

import numpy as np

from .._log import get_logger
from ..errors import ProcessingError


def encode_frame(frame):
    """把一帧整理成可写 TIFF 的数组：实数取 float32，复数交错成 ``[幅度, 相位(rad)]``。"""
    array = np.asarray(frame)
    if array.dtype.kind != "c":
        return np.ascontiguousarray(array, dtype=np.float32)
    depth = int(array.shape[-1])
    out = np.empty(array.shape[:-1] + (2 * depth,), dtype=np.float32)
    out[..., :depth] = np.abs(array)
    out[..., depth:] = np.angle(array)
    return np.ascontiguousarray(out)


def write_volume(path, volume, bigtiff=True, append=False):
    """把 ``(frames, rows, cols)`` 或 ``(frames, cols)`` 写成多帧 TIFF。"""
    try:
        import tifffile
    except Exception as error:
        raise ProcessingError(
            f"tifffile is required to write {path}: {error}"
        ) from error
    array = np.asarray(volume)
    if array.ndim not in (2, 3):
        raise ProcessingError(f"volume must be 2-D or 3-D, got {array.shape}")
    frames = [encode_frame(frame) for frame in (array if array.ndim == 3 else [array])]
    options = {"photometric": "minisblack"}
    if not append:
        options["bigtiff"] = bool(bigtiff)
    tifffile.imwrite(str(path), frames, append=bool(append), **options)
    get_logger("io").info("wrote %d frame(s) to %s", len(frames), path)
    return str(path)
