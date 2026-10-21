"""数据形状推导（无 UI 依赖，规则与主程序 ``DataShape.py`` 一致）。

SDK 内部真正决定形状的是 :class:`octsdk.processing.processor.OctProcessor`；
这里提供的是"给定一包数据如何解释"的公共函数，供宿主程序做显示/存盘命名时复用。
"""

from dataclasses import dataclass

from ..config import BLINE_MODES, CSCAN_MODES, ProcessingConfig
from ..errors import ProcessingError


@dataclass(frozen=True)
class DataShapeInfo:
    """一包数据的语义形状。"""

    frame_count: int
    y_pixels: int
    repeat_count: int
    x_pixels: int
    z_pixels: int


def spectral_output_frame_count(rows, aline_avg, bline_avg, x_pixels):
    """光谱域平均后剩下的 A-line 行数（不足一组的余数被丢弃）。"""
    aline_avg = max(1, int(aline_avg))
    bline_avg = max(1, int(bline_avg))
    group_rows = max(1, int(x_pixels)) * bline_avg
    complete = (int(rows) // group_rows) * group_rows
    return complete // (aline_avg * bline_avg)


def applied_averages(config: ProcessingConfig, rows):
    """返回这一包实际能做的 (aline_avg, bline_avg)；不足一个完整 Y 行就退回 (1, 1)。"""
    if not config.aline_avg > 1 and not config.bline_avg > 1:
        return 1, 1
    if int(rows) < config.group_rows:
        return 1, 1
    return max(1, int(config.aline_avg)), max(1, int(config.bline_avg))


def effective_x_pixels(config: ProcessingConfig, aline_avg_applied=1):
    """平均之后每条 B-line 的有效 X 位置数。"""
    x_pixels = max(1, int(config.x_pixels))
    aline_avg_applied = max(1, int(aline_avg_applied))
    if x_pixels % aline_avg_applied:
        raise ProcessingError(
            "AlineAVG must divide x_pixels: "
            f"x_pixels={x_pixels}, aline_avg={aline_avg_applied}"
        )
    return x_pixels // aline_avg_applied


def compute_data_shape(
    config: ProcessingConfig,
    acq_mode,
    frame_count,
    *,
    y_lines=1,
    is_raw=False,
    aline_avg_applied=1,
    bline_avg_applied=1,
):
    """解释一包 ``(frame_count, x, z)`` 数据的 Y/X/Z 与重复次数。"""
    frame_count = int(frame_count)
    x_pixels = effective_x_pixels(config, aline_avg_applied)
    bline_avg = max(1, int(config.bline_avg) // max(1, int(bline_avg_applied)))
    frames_per_y = max(1, x_pixels * bline_avg)
    z_pixels = config.spectral_samples if is_raw else config.resolved_depth_pixels

    if acq_mode in CSCAN_MODES:
        y_pixels = max(1, frame_count // frames_per_y)
    else:
        y_pixels = max(1, int(y_lines))

    if acq_mode in BLINE_MODES:
        repeat_count = max(1, frame_count // x_pixels)
    elif not is_raw:
        repeat_count = frame_count
    elif acq_mode in CSCAN_MODES and frame_count >= frames_per_y:
        repeat_count = bline_avg
    else:
        repeat_count = frame_count

    return DataShapeInfo(
        frame_count=frame_count,
        y_pixels=y_pixels,
        repeat_count=repeat_count,
        x_pixels=x_pixels,
        z_pixels=z_pixels,
    )
