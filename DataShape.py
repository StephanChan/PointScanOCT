from dataclasses import dataclass

from ActionTypes import AcqTypes
from CameraUi import effective_camera_sample_count, x_pixel_count


CSCAN_MODES = (
    AcqTypes.FINITE_CSCAN,
    AcqTypes.CONTINUOUS_CSCAN,
)


@dataclass(frozen=True)
class DataShapeInfo:
    frame_count: int
    y_pixels: int
    repeat_count: int
    x_pixels: int
    z_pixels: int


def depth_pixels(ui, data=None, raw=False):
    if data is not None and getattr(data, "ndim", 0) >= 3:
        return int(data.shape[2])
    if raw:
        return effective_camera_sample_count(ui)
    return int(ui.DepthRange.value())


def x_pixels(ui):
    return x_pixel_count(ui)


def repeat_count(ui, data=None, raw=False, acq_mode=None, gpu_avg_count=1):
    configured_bline_avg = max(1, int(ui.BlineAVG.value()))
    if data is None or getattr(data, "ndim", 0) < 1:
        if raw:
            return configured_bline_avg
        return max(1, configured_bline_avg // max(1, int(gpu_avg_count)))

    frame_count = int(data.shape[0])
    x_count = max(1, x_pixels(ui))
    if acq_mode in (AcqTypes.FINITE_BSCAN, AcqTypes.CONTINUOUS_BLINE, AcqTypes.X_FREQUENCY):
        return max(1, frame_count // x_count)
    if not raw:
        return frame_count

    if acq_mode in CSCAN_MODES:
        frames_per_y = max(1, x_count * configured_bline_avg)
        if frame_count >= frames_per_y:
            return configured_bline_avg
    return frame_count


def cscan_y_count(ui, data, raw=False):
    frame_count = int(data.shape[0])
    x_count = max(1, x_pixels(ui))
    configured_bline_avg = max(1, int(ui.BlineAVG.value()))
    frames_per_y = max(1, x_count * configured_bline_avg)
    if frame_count >= frames_per_y:
        return frame_count // frames_per_y
    return frame_count


def data_shape(ui, data=None, raw=False, acq_mode=None, gpu_avg_count=1):
    frame_count = 0
    if data is not None and getattr(data, "ndim", 0) >= 1:
        frame_count = int(data.shape[0])
    return DataShapeInfo(
        frame_count=frame_count,
        y_pixels=cscan_y_count(ui, data, raw) if data is not None else max(1, int(ui.Ypixels.value())),
        repeat_count=repeat_count(ui, data, raw, acq_mode, gpu_avg_count),
        x_pixels=x_pixels(ui),
        z_pixels=depth_pixels(ui, data, raw),
    )

