"""OCT 处理层：形状推导、数组算子、标定加载与 ``OctProcessor``。"""

from .calibration import DispersionCalibration
from .processor import (
    BACKEND_AUTO,
    BACKEND_CPU,
    BACKEND_GPU,
    OctProcessor,
    ProcessedPackage,
    resolve_backend,
)
from .shapes import (
    DataShapeInfo,
    applied_averages,
    compute_data_shape,
    effective_x_pixels,
    spectral_output_frame_count,
)

__all__ = [
    "BACKEND_AUTO",
    "BACKEND_CPU",
    "BACKEND_GPU",
    "DataShapeInfo",
    "DispersionCalibration",
    "OctProcessor",
    "ProcessedPackage",
    "applied_averages",
    "compute_data_shape",
    "effective_x_pixels",
    "resolve_backend",
    "spectral_output_frame_count",
]
