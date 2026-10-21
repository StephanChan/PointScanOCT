"""OCT SDK —— 相机 / AODO / 处理层的商用打包入口。

当前版本（0.1.x，M1 已完成）：只有处理层可用。

    >>> import octsdk
    >>> from octsdk import OctConfig, OctProcessor
    >>> cfg = OctConfig().resolved_processing()
    >>> processor = OctProcessor(cfg, backend="auto")
    >>> processor.prepare()
    >>> package = processor.process(raw_spectra)      # raw_spectra: (rows, samples)

相机（``CameraDevice``）与 AODO（``AodoDevice``）在 M2/M3 加入，
届时通过 ``octsdk.CameraDevice`` / ``octsdk.AodoDevice`` 导出，不在 import
阶段加载厂商 SDK（避免没有装驱动时连 ``import octsdk`` 都失败）。
"""

__version__ = "0.1.0"

from ._log import configure_logging, get_logger
from .aodo import (
    AodoDeviceBase,
    SimulatedAodoDevice,
    WaveformSpec,
    build_waveform,
    open_aodo,
)
from .camera import (
    CameraDeviceBase,
    CameraFrameInfo,
    SimulatedCameraDevice,
    SpectraPackage,
    open_camera,
)
from .config import (
    AcqModes,
    AodoConfig,
    CameraConfig,
    EngineConfig,
    OctConfig,
    ProcessingConfig,
    ScanConfig,
)
from .environment import environment_report
from .engine import OctEngine, ScanResult
from .errors import (
    BackendUnavailableError,
    CalibrationError,
    ConfigurationError,
    HardwareError,
    OctSdkError,
    ProcessingError,
)
from .processing import (
    DataShapeInfo,
    DispersionCalibration,
    OctProcessor,
    ProcessedPackage,
    applied_averages,
    compute_data_shape,
    effective_x_pixels,
    spectral_output_frame_count,
)
from .pipeline import Pipeline

__all__ = [
    "__version__",
    "AcqModes",
    "AodoConfig",
    "AodoDeviceBase",
    "BackendUnavailableError",
    "CalibrationError",
    "CameraConfig",
    "CameraDeviceBase",
    "CameraFrameInfo",
    "ConfigurationError",
    "DataShapeInfo",
    "EngineConfig",
    "DispersionCalibration",
    "HardwareError",
    "OctConfig",
    "OctEngine",
    "OctProcessor",
    "OctSdkError",
    "Pipeline",
    "ProcessedPackage",
    "ProcessingConfig",
    "ProcessingError",
    "ScanConfig",
    "ScanResult",
    "SimulatedAodoDevice",
    "SimulatedCameraDevice",
    "SpectraPackage",
    "WaveformSpec",
    "applied_averages",
    "build_waveform",
    "compute_data_shape",
    "configure_logging",
    "effective_x_pixels",
    "environment_report",
    "get_logger",
    "open_aodo",
    "open_camera",
    "spectral_output_frame_count",
]
