"""``OctProcessor``：把一包原始光谱处理成幅度 / 复数深度图像。

对外承诺
--------
* 输入：``(rows, spectral_samples)`` 的原始光谱（uint16 / uint8 / float32 均可），
  行序 ``[Y][Bline重复][X位置][Aline重复]``。
* 输出：:class:`ProcessedPackage`，其中 ``data`` 的行数 = 光谱域平均之后剩下的 A-line
  行数，深度维 = ``depth_pixels``；复数模式下 ``keep_complex=True``。
* 后端：``gpu``（cupy/cuFFT）或 ``cpu``（numpy），``auto`` 先试 GPU 再退回 CPU；
  两条路径共用 :mod:`octsdk.processing.ops` 的公式，结果可逐元素对比。
"""

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .._log import emit_status, get_logger
from ..config import ProcessingConfig
from ..errors import BackendUnavailableError, ConfigurationError, ProcessingError
from . import ops
from .background import load_spectral_background, load_x_normalization
from .calibration import DispersionCalibration
from .shapes import applied_averages, effective_x_pixels, spectral_output_frame_count

BACKEND_AUTO = "auto"
BACKEND_GPU = "gpu"
BACKEND_CPU = "cpu"


@dataclass
class ProcessedPackage:
    """一包的处理结果与元数据。"""

    data: np.ndarray
    rows: int
    x_pixels: int
    depth_pixels: int
    dtype: str
    backend: str
    spectral_aline_avg: int
    spectral_bline_avg: int
    package_start: int = 0
    metadata: dict = field(default_factory=dict)
    timings: dict = field(default_factory=dict)

    def amplitude(self):
        """幅度视图（复数结果取模，实数结果原样返回）。"""
        if self.data.dtype.kind == "c":
            return np.abs(self.data)
        return self.data


class OctProcessor:
    """光谱域 OCT 处理核心（无 Qt、无硬件依赖）。"""

    def __init__(self, config: ProcessingConfig, backend=BACKEND_AUTO, status_callback=None):
        if not isinstance(config, ProcessingConfig):
            raise ConfigurationError(f"config must be ProcessingConfig, got {type(config)}")
        if int(config.x_pixels) < 1:
            raise ConfigurationError(
                "ProcessingConfig.x_pixels must be resolved (>0) before building OctProcessor; "
                "use OctConfig.resolved_processing()."
            )
        self.config = config
        self._status_callback = status_callback
        self._log = get_logger("processor")
        self.backend_name, self._xp, self._backend_error = resolve_backend(backend)
        self.calibration: Optional[DispersionCalibration] = None
        self.background = None
        self.x_normalization = None
        self._background_xp = None
        self._normalization_xp = None
        self.report = {}

    # -- 准备 -------------------------------------------------------------- #
    def prepare(self):
        """加载色散标定与背景。配置了但缺失/不匹配时抛异常（不静默降级）。"""
        config = self.config
        self.calibration = None
        if config.use_calibration and config.calibration_dir:
            self.calibration = DispersionCalibration.load(
                config.calibration_dir, config.spectral_samples
            )
        elif config.use_calibration:
            self._warn("no calibration_dir configured: dispersion compensation is OFF")
        else:
            self._warn("use_calibration=False: dispersion compensation is OFF")

        self.background = None
        if config.apply_background and config.background_path:
            self.background = load_spectral_background(
                config.background_path, config.spectral_samples, config.x_pixels
            )
        elif config.apply_background:
            self._warn("no background_path configured: background subtraction is OFF")

        self.x_normalization = None
        if config.background_x_normalization_path:
            x_eff = effective_x_pixels(config, config.aline_avg)
            self.x_normalization = load_x_normalization(
                config.background_x_normalization_path, x_eff
            )

        xp = self._xp
        self._background_xp = (
            None if self.background is None else xp.asarray(self.background, dtype=xp.float32)
        )
        self._normalization_xp = (
            None if self.x_normalization is None
            else xp.asarray(self.x_normalization, dtype=xp.float32)
        )
        self.report = self.describe()
        return self.report

    def describe(self):
        """当前生效配置与依赖状态（写日志/给客户看）。"""
        config = self.config
        return {
            "backend": self.backend_name,
            "backend_error": self._backend_error,
            "spectral_samples": config.spectral_samples,
            "raw_spectral_samples": int(config.raw_spectral_samples),
            "spectral_downsample": int(config.spectral_downsample),
            "x_pixels": int(config.x_pixels),
            "aline_avg": int(config.aline_avg),
            "bline_avg": int(config.bline_avg),
            "depth_start": int(config.depth_start),
            "depth_pixels": int(config.resolved_depth_pixels),
            "keep_complex": bool(config.keep_complex),
            "output_dtype": config.output_dtype,
            "x_notch": bool(config.x_notch_enabled),
            "y_notch": bool(config.y_notch_enabled),
            "amplification": float(config.amplification),
            "calibration": None if self.calibration is None else self.calibration.describe(),
            "background": self.background is not None,
            "x_normalization": self.x_normalization is not None,
        }

    def _warn(self, message):
        self._log.warning(message)
        emit_status(self._status_callback, message)

    # -- 主处理 ------------------------------------------------------------ #
    def process(self, spectra, package_start=0):
        """处理一包原始光谱，返回 :class:`ProcessedPackage`。"""
        config = self.config
        samples = config.spectral_samples
        spectra = np.asarray(spectra)
        if spectra.ndim == 3 and int(spectra.shape[1]) == 1:
            spectra = spectra[:, 0, :]
        if getattr(spectra, "ndim", 0) != 2:
            raise ProcessingError(f"spectra must be (rows, samples), got {spectra.shape}")
        if int(spectra.shape[1]) != samples:
            raise ProcessingError(
                f"spectral length mismatch: got {int(spectra.shape[1])}, expected {samples}"
            )
        rows = int(spectra.shape[0])
        if rows < 1:
            raise ProcessingError("empty spectra package")

        aline_avg, bline_avg = applied_averages(config, rows)
        x_pixels = effective_x_pixels(config, aline_avg)
        out_rows = spectral_output_frame_count(rows, aline_avg, bline_avg, config.x_pixels)
        if out_rows < 1:
            raise ProcessingError(
                "spectra do not contain one complete averaging group: "
                f"rows={rows}, x_pixels={config.x_pixels}, "
                f"aline_avg={config.aline_avg}, bline_avg={config.bline_avg}"
            )
        depth_pixels = config.resolved_depth_pixels
        internal_dtype = np.complex64 if config.keep_complex else np.float32
        result = np.empty((out_rows, depth_pixels), dtype=internal_dtype)

        xp = self._xp
        arrays = None if self.calibration is None else self.calibration.backend_arrays(xp)
        timings = {}
        offset = 0
        started = time.perf_counter()
        for start, stop in self._chunk_bounds(rows, aline_avg, bline_avg):
            chunk_started = time.perf_counter()
            data = xp.asarray(spectra[start:stop], dtype=xp.float32)
            if aline_avg > 1 or bline_avg > 1:
                data = ops.apply_spectral_average(
                    xp, data, config.x_pixels, aline_avg, bline_avg
                )
            if self._background_xp is not None:
                data = ops.subtract_background(xp, data, self._background_xp)
            if arrays is not None:
                data = ops.interpolate_spectra(xp, data, arrays)
            if config.x_notch_enabled and x_pixels >= 2 and int(data.shape[0]) % x_pixels == 0:
                data = ops.apply_x_notch(xp, data, x_pixels, config.x_notch_half_width_fraction)
            data = ops.apply_dispersion_and_fft(xp, data, arrays, samples)
            data = ops.crop_depth(xp, data, config.depth_start, depth_pixels)
            if not config.keep_complex:
                data = xp.abs(data)
            if float(config.amplification) != 1.0:
                data = data * xp.float32(config.amplification)
            if self._normalization_xp is not None:
                data = ops.apply_x_normalization(xp, data, self._normalization_xp, x_pixels)
            host = self._to_host(data)
            produced = int(host.shape[0])
            result[offset:offset + produced] = host
            offset += produced
            timings["chunk_s"] = timings.get("chunk_s", 0.0) + (
                time.perf_counter() - chunk_started
            )
        if offset != out_rows:
            raise ProcessingError(
                f"internal row accounting mismatch: produced {offset}, planned {out_rows}"
            )

        encode_mode = config.output_dtype
        if config.keep_complex and encode_mode != "complex64":
            encode_mode = "complex64"
        encoded, metadata = ops.encode_output(result, encode_mode)
        metadata.update(
            {
                "internal_dtype": str(internal_dtype),
                "requested_output_dtype": config.output_dtype,
                "spectral_aline_avg": int(aline_avg),
                "spectral_bline_avg": int(bline_avg),
                "x_pixels": int(x_pixels),
                "depth_pixels": int(depth_pixels),
                "spectral_samples": int(samples),
                "amplification": float(config.amplification),
                "x_notch": bool(config.x_notch_enabled),
                "calibration_used": arrays is not None,
                "background_subtracted": self._background_xp is not None,
            }
        )
        timings["total_s"] = time.perf_counter() - started
        return ProcessedPackage(
            data=encoded,
            rows=int(out_rows),
            x_pixels=int(x_pixels),
            depth_pixels=int(depth_pixels),
            dtype=str(encoded.dtype),
            backend=self.backend_name,
            spectral_aline_avg=int(aline_avg),
            spectral_bline_avg=int(bline_avg),
            package_start=int(package_start),
            metadata=metadata,
            timings=timings,
        )

    def apply_y_notch(self, volume, half_width_fraction=None):
        """对整卷 ``(Y, X, Z)`` 做 Y 向带阻（内部按 Z 分块），返回同形状结果。"""
        config = self.config
        fraction = (
            config.y_notch_half_width_fraction
            if half_width_fraction is None
            else float(half_width_fraction)
        )
        xp = self._xp
        filtered = ops.apply_y_notch(
            xp, xp.asarray(volume), fraction, config.y_notch_z_chunk
        )
        return self._to_host(filtered)

    # -- 内部 -------------------------------------------------------------- #
    def _chunk_bounds(self, rows, aline_avg, bline_avg):
        """分块边界；做光谱域平均时每块正好一条 Y 行（跨块会破坏平均分组）。"""
        if aline_avg > 1 or bline_avg > 1:
            step = max(1, int(self.config.group_rows))
        else:
            step = max(1, int(self.config.chunk_frames))
        start = 0
        while start < rows:
            yield start, min(start + step, rows)
            start += step

    def _to_host(self, data):
        if self._xp is np:
            return np.asarray(data)
        return self._xp.asnumpy(data)


def resolve_backend(name=BACKEND_AUTO):
    """解析后端，返回 ``(名称, xp 模块, 失败原因)``。"""
    name = str(name or BACKEND_AUTO).lower()
    if name not in (BACKEND_AUTO, BACKEND_GPU, BACKEND_CPU):
        raise ConfigurationError(f"backend must be auto/gpu/cpu, got {name!r}")
    if name in (BACKEND_AUTO, BACKEND_GPU):
        try:
            import cupy

            if int(cupy.cuda.runtime.getDeviceCount()) < 1:
                raise RuntimeError("no CUDA device reported by cupy")
            return BACKEND_GPU, cupy, None
        except Exception as error:
            if name == BACKEND_GPU:
                raise BackendUnavailableError(f"GPU backend unavailable: {error}") from error
            return BACKEND_CPU, np, str(error)
    return BACKEND_CPU, np, None
