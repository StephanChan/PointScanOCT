"""SDK 配置对象：与 UI 完全解耦，所有参数在这里显式给出。

设计约定
--------
* 每个配置类是 ``dataclass``，可用 ``to_dict()`` / ``from_dict()`` 与 JSON/INI 互转。
* 单位写进字段名：``_um`` 微米、``_mm`` 毫米、``_hz`` 赫兹、``_us`` 微秒、``_px`` 像素。
* 构造时立即校验（``__post_init__``），非法组合抛 :class:`ConfigurationError`，
  不允许"跑起来才发现"。
"""

from dataclasses import asdict, dataclass, fields
from typing import Optional, Tuple

from .errors import ConfigurationError


class AcqModes:
    """采集模式（字符串与主程序 ``ActionTypes.AcqTypes`` 保持一致，便于对照）。"""

    FINITE_BSCAN = "FiniteBscan"
    CONTINUOUS_BLINE = "ContinuousBline"
    FINITE_CSCAN = "FiniteCscan"
    CONTINUOUS_CSCAN = "ContinuousCscan"
    X_FREQUENCY = "Xfrequency"


BLINE_MODES = (AcqModes.FINITE_BSCAN, AcqModes.CONTINUOUS_BLINE, AcqModes.X_FREQUENCY)
CSCAN_MODES = (AcqModes.FINITE_CSCAN, AcqModes.CONTINUOUS_CSCAN)
CONTINUOUS_MODES = (AcqModes.CONTINUOUS_BLINE, AcqModes.CONTINUOUS_CSCAN)

#: 处理层可输出的 dtype（见 ``octsdk.processing.ops.encode_output``）
OUTPUT_DTYPES = ("float32", "float16", "complex64", "amp_phase_fp16", "amp_phase_int16")


@dataclass
class ScanConfig:
    """扫描几何。``x_positions`` 由步距与长度推导，与主程序 ``CameraUi.x_pixel_count`` 同义。"""

    x_step_um: float = 10.0
    x_length_um: float = 12000.0
    y_step_um: float = 10.0
    y_lines: int = 1010
    flyback_samples: int = 300

    def __post_init__(self):
        if self.x_step_um <= 0 or self.y_step_um <= 0:
            raise ConfigurationError(
                f"step must be positive: x_step_um={self.x_step_um}, y_step_um={self.y_step_um}"
            )
        if self.x_length_um <= 0:
            raise ConfigurationError(f"x_length_um must be positive: {self.x_length_um}")
        if int(self.y_lines) < 1:
            raise ConfigurationError(f"y_lines must be >= 1: {self.y_lines}")
        if int(self.flyback_samples) < 1:
            raise ConfigurationError(f"flyback_samples must be >= 1: {self.flyback_samples}")

    @property
    def x_positions(self):
        """每条 B-line 的振镜位置数（不含 AlineAVG 过采样）。"""
        return max(1, int(round(float(self.x_length_um) / float(self.x_step_um))))

    def to_dict(self):
        return asdict(self)


@dataclass
class ProcessingConfig:
    """OCT 处理参数。``x_pixels=0`` 表示由 :meth:`OctConfig.resolved_processing` 推导。"""

    # 光谱采样
    raw_spectral_samples: int = 1440
    spectral_downsample: int = 1
    # 空间采样
    x_pixels: int = 0
    aline_avg: int = 1
    bline_avg: int = 1
    # 深度窗
    depth_start: int = 0
    depth_pixels: Optional[int] = None
    # 标定与背景
    calibration_dir: Optional[str] = None
    background_path: Optional[str] = None
    background_x_normalization_path: Optional[str] = None
    use_calibration: bool = True
    apply_background: bool = True
    # 陷波
    x_notch_enabled: bool = False
    x_notch_half_width_fraction: float = 0.25
    y_notch_enabled: bool = False
    y_notch_half_width_fraction: float = 0.25
    y_notch_z_chunk: int = 32
    # 输出
    amplification: float = 100.0
    keep_complex: bool = False
    output_dtype: str = "float32"
    # 分块
    chunk_frames: int = 32

    def __post_init__(self):
        if int(self.raw_spectral_samples) < 1:
            raise ConfigurationError(
                f"raw_spectral_samples must be >= 1: {self.raw_spectral_samples}"
            )
        if int(self.spectral_downsample) < 1:
            raise ConfigurationError(
                f"spectral_downsample must be >= 1: {self.spectral_downsample}"
            )
        if int(self.raw_spectral_samples) % int(self.spectral_downsample):
            raise ConfigurationError(
                "spectral_downsample must divide raw_spectral_samples: "
                f"{self.spectral_downsample} vs {self.raw_spectral_samples}"
            )
        if int(self.aline_avg) < 1 or int(self.bline_avg) < 1:
            raise ConfigurationError(
                f"averages must be >= 1: aline_avg={self.aline_avg}, bline_avg={self.bline_avg}"
            )
        if int(self.x_pixels) < 0:
            raise ConfigurationError(f"x_pixels must be >= 0: {self.x_pixels}")
        for name, value in (
            ("x_notch_half_width_fraction", self.x_notch_half_width_fraction),
            ("y_notch_half_width_fraction", self.y_notch_half_width_fraction),
        ):
            if not 0.0 <= float(value) <= 1.0:
                raise ConfigurationError(f"{name} must be within [0, 1]: {value}")
        if int(self.y_notch_z_chunk) < 1:
            raise ConfigurationError(f"y_notch_z_chunk must be >= 1: {self.y_notch_z_chunk}")
        if int(self.chunk_frames) < 1:
            raise ConfigurationError(f"chunk_frames must be >= 1: {self.chunk_frames}")
        if self.output_dtype not in OUTPUT_DTYPES:
            raise ConfigurationError(
                f"output_dtype must be one of {OUTPUT_DTYPES}: {self.output_dtype!r}"
            )
        if self.keep_complex and self.output_dtype in ("float16", "amp_phase_fp16", "amp_phase_int16"):
            raise ConfigurationError(
                "keep_complex output cannot also request a lossy encoder: "
                f"output_dtype={self.output_dtype!r}"
            )
        if self.depth_pixels is not None and int(self.depth_pixels) < 1:
            raise ConfigurationError(f"depth_pixels must be >= 1: {self.depth_pixels}")

    @property
    def spectral_samples(self):
        """下采样后的光谱采样数（FFT 长度）。"""
        return int(self.raw_spectral_samples) // int(self.spectral_downsample)

    @property
    def resolved_depth_pixels(self):
        if self.depth_pixels is not None:
            return int(self.depth_pixels)
        return max(1, self.spectral_samples - int(self.depth_start))

    @property
    def group_rows(self):
        """一个完整 Y 行占用的原始 A-line 行数（AlineAVG × BlineAVG × 位置数）。"""
        return max(1, int(self.x_pixels)) * int(self.bline_avg)

    def to_dict(self):
        return asdict(self)


@dataclass
class CameraConfig:
    """AK / IKap CameraLink 相机配置（对应主程序 ``ThreadCamera_AK``）。

    ``vlcf_path`` 优先；否则用 ``preset``（例如 ``"12bit4tap"``）在本包 ``camera/presets``
    里找；两者都为空时按 ``bit_depth`` + ``taps`` 选预设。``.ccf`` 缺省取与 ``.vlcf`` 同名文件。
    """

    vlcf_path: Optional[str] = None
    preset: Optional[str] = None
    ccf_path: Optional[str] = None
    bit_depth: int = 12
    taps: int = 4
    offset_x_px: int = 336
    line_rate_hz: float = 10000.0
    exposure_us: float = 4.2
    gain: str = "x1"
    digital_gain: float = 1.0
    trigger_selector: str = "ExternalPulse"
    trigger_source: str = "ShaftEncoder1"
    trigger_activation: str = "RisingEdge"
    trigger_delay_us: float = 0.0
    frame_buffers: int = 128
    board_timeout_ms: int = -1
    frame_timeout_ms: int = 1000
    vlcf_authoritative: bool = True
    vlcf_height_authority: bool = False
    sdk_root: Optional[str] = None
    simulate: bool = False

    def __post_init__(self):
        if int(self.bit_depth) not in (8, 10, 12, 14, 16):
            raise ConfigurationError(f"bit_depth must be one of 8/10/12/14/16: {self.bit_depth}")
        if int(self.taps) not in (1, 2, 4, 8, 10):
            raise ConfigurationError(f"taps must be one of 1/2/4/8/10: {self.taps}")
        if float(self.line_rate_hz) <= 0:
            raise ConfigurationError(f"line_rate_hz must be positive: {self.line_rate_hz}")
        if float(self.exposure_us) <= 0:
            raise ConfigurationError(f"exposure_us must be positive: {self.exposure_us}")
        if int(self.frame_buffers) < 1:
            raise ConfigurationError(f"frame_buffers must be >= 1: {self.frame_buffers}")
        if int(self.offset_x_px) < 0:
            raise ConfigurationError(f"offset_x_px must be >= 0: {self.offset_x_px}")

    def to_dict(self):
        return asdict(self)


@dataclass
class AodoConfig:
    """ART-DAQ 振镜 + 相机行触发配置（对应主程序 ``ThreadAODO_art``）。"""

    board: str = "Galvo"
    ao_channel_x: str = "ao1"
    ao_channel_y: str = "ao0"
    sync_do_line: str = "port0/line0"
    trigger_out_pfi: str = "PFI3"
    trigger_in_pfi: str = "PFI7"
    sample_rate_hz: Optional[float] = None
    x_mm_per_volt: Optional[float] = None
    y_mm_per_volt: Optional[float] = None
    x_voltage_bias: float = 0.0
    y_voltage_bias: float = 0.0
    ao_voltage_limit_v: float = 10.0
    sdk_root: Optional[str] = None
    simulate: bool = False

    def __post_init__(self):
        if self.ao_channel_x == self.ao_channel_y:
            raise ConfigurationError("ao_channel_x and ao_channel_y must differ")
        if float(self.ao_voltage_limit_v) <= 0:
            raise ConfigurationError(
                f"ao_voltage_limit_v must be positive: {self.ao_voltage_limit_v}"
            )
        if self.sample_rate_hz is not None and float(self.sample_rate_hz) <= 0:
            raise ConfigurationError(f"sample_rate_hz must be positive: {self.sample_rate_hz}")

    def resolved_sample_rate_hz(self, camera):
        """未显式给采样率时按主程序约定取 ``2 × 行频``（每个像素两个 AO 采样点）。"""
        if self.sample_rate_hz is not None:
            return float(self.sample_rate_hz)
        return 2.0 * float(camera.line_rate_hz)

    def to_dict(self):
        return asdict(self)


@dataclass
class EngineConfig:
    """编排层参数（不影响单包处理的数学）。"""

    package_y_lines: int = 64
    max_volume_gb: float = 4.0
    tiff_bigtiff: bool = True

    def __post_init__(self):
        if int(self.package_y_lines) < 1:
            raise ConfigurationError(f"package_y_lines must be >= 1: {self.package_y_lines}")
        if float(self.max_volume_gb) <= 0:
            raise ConfigurationError(f"max_volume_gb must be positive: {self.max_volume_gb}")

    def to_dict(self):
        return asdict(self)


@dataclass
class OctConfig:
    """SDK 总配置。四个子配置之间存在依赖，统一在这里做交叉校验。"""

    scan: Optional[ScanConfig] = None
    processing: Optional[ProcessingConfig] = None
    camera: Optional[CameraConfig] = None
    aodo: Optional[AodoConfig] = None
    engine: Optional[EngineConfig] = None

    def __post_init__(self):
        self.scan = self.scan or ScanConfig()
        self.processing = self.processing or ProcessingConfig()
        self.camera = self.camera or CameraConfig()
        self.aodo = self.aodo or AodoConfig()
        self.engine = self.engine or EngineConfig()
        for name, value, expected in (
            ("scan", self.scan, ScanConfig),
            ("processing", self.processing, ProcessingConfig),
            ("camera", self.camera, CameraConfig),
            ("aodo", self.aodo, AodoConfig),
            ("engine", self.engine, EngineConfig),
        ):
            if not isinstance(value, expected):
                raise ConfigurationError(f"{name} must be {expected.__name__}, got {type(value)}")

    def resolved_processing(self):
        """返回补全了 ``x_pixels`` 的 :class:`ProcessingConfig` 副本。"""
        processing = ProcessingConfig(**processing_as_dict(self.processing))
        if int(processing.x_pixels) <= 0:
            processing.x_pixels = int(self.scan.x_positions) * int(processing.aline_avg)
        if int(processing.x_pixels) % int(processing.aline_avg):
            raise ConfigurationError(
                "AlineAVG must divide x_pixels: "
                f"x_pixels={processing.x_pixels}, aline_avg={processing.aline_avg}"
            )
        if int(processing.depth_start) >= processing.spectral_samples:
            raise ConfigurationError(
                "depth_start must be smaller than spectral samples: "
                f"{processing.depth_start} vs {processing.spectral_samples}"
            )
        if int(processing.depth_start) + processing.resolved_depth_pixels > processing.spectral_samples:
            raise ConfigurationError(
                "depth window exceeds spectral samples: start="
                f"{processing.depth_start}, pixels={processing.resolved_depth_pixels}, "
                f"samples={processing.spectral_samples}"
            )
        return processing

    def to_dict(self):
        return {
            "scan": self.scan.to_dict(),
            "processing": self.processing.to_dict(),
            "camera": self.camera.to_dict(),
            "aodo": self.aodo.to_dict(),
            "engine": self.engine.to_dict(),
        }

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict):
            raise ConfigurationError(f"config dict expected, got {type(data)}")
        return cls(
            scan=build_dataclass(ScanConfig, data.get("scan")),
            processing=build_dataclass(ProcessingConfig, data.get("processing")),
            camera=build_dataclass(CameraConfig, data.get("camera")),
            aodo=build_dataclass(AodoConfig, data.get("aodo")),
            engine=build_dataclass(EngineConfig, data.get("engine")),
        )


def processing_as_dict(processing):
    return {item.name: getattr(processing, item.name) for item in fields(ProcessingConfig)}


def build_dataclass(cls, data):
    """按 dataclass 字段名过滤后构造，未知键忽略（方便版本间配置前向兼容）。"""
    if data is None:
        return cls()
    if not isinstance(data, dict):
        raise ConfigurationError(f"{cls.__name__} expects a dict, got {type(data)}")
    allowed = {item.name for item in fields(cls)}
    payload = {key: value for key, value in data.items() if key in allowed}
    return cls(**payload)
