"""``OctEngine``：把相机 / 振镜 / 处理 / 累积串成一条命令。

对应主程序 ``ThreadWeaver`` 的 ``SingleScan`` / ``RptScan``，但去掉了 UI 与 stage：

    prepare()                      # 处理层标定 + 相机/AODO 打开与几何配置
    single_scan(FiniteCscan)       # 起振镜 → 起相机 → 逐包处理 → 收尾（可选 Y 陷波 / 存盘）
    repeat_scan(ContinuousBline)   # 连续采集，直到时长/包数/外部停止

回调与线程：相机在采集线程里交出 ``SpectraPackage``，引擎在**同一线程**里同步调用
``processor.process``（与主程序一致：处理延迟会直接变成丢帧压力）。所以 ``single_scan``
/ ``repeat_scan`` 返回前不会重入。
"""

import threading
import time
from dataclasses import dataclass, field

import numpy as np

from ._log import emit_status, get_logger
from .aodo import open_aodo
from .camera import open_camera
from .config import CONTINUOUS_MODES, CSCAN_MODES, OctConfig
from .errors import ConfigurationError, HardwareError
from .io.tiff import write_volume
from .pipeline import Pipeline
from .processing import OctProcessor


@dataclass
class ScanResult:
    """一次扫描的结果与统计。"""

    mode: str
    completed: bool
    filled_y: int
    rows: int
    packages: int
    backend: str
    volume: np.ndarray = None
    aip: np.ndarray = None
    bline: np.ndarray = None
    y_notch_applied: bool = False
    y_notch_backend: str = None
    saved_path: str = None
    timings: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    def describe(self):
        return {
            "mode": self.mode,
            "completed": self.completed,
            "filled_y": int(self.filled_y),
            "rows": int(self.rows),
            "packages": int(self.packages),
            "backend": self.backend,
            "volume_shape": None if self.volume is None else tuple(int(v) for v in self.volume.shape),
            "aip_shape": None if self.aip is None else tuple(int(v) for v in self.aip.shape),
            "y_notch_applied": self.y_notch_applied,
            "y_notch_backend": self.y_notch_backend,
            "saved_path": self.saved_path,
            "timings": dict(self.timings),
        }


class OctEngine:
    """相机 + 振镜 + 处理 + 累积的编排层。"""

    def __init__(self, config, backend="auto", status_callback=None, camera_options=None):
        if not isinstance(config, OctConfig):
            raise ConfigurationError(f"config must be OctConfig, got {type(config)}")
        self.config = config
        self.processing = config.resolved_processing()
        self.engine_config = config.engine
        self._status_callback = status_callback
        self._log = get_logger("engine")
        self.processor = OctProcessor(
            self.processing, backend=backend, status_callback=status_callback
        )
        self.camera = open_camera(
            config.camera, status_callback=status_callback, **(camera_options or {})
        )
        self.aodo = open_aodo(
            config.aodo, camera_config=config.camera, status_callback=status_callback
        )
        self.pipeline = None
        self._finish_event = threading.Event()
        self._error = None
        self._prepared = False
        self._accept_packages = False
        self._finish_on_complete = False

    # -- 准备 / 关闭 ------------------------------------------------------- #
    def prepare(self):
        """加载标定、打开相机与 DAQ，并按几何配置相机。"""
        report = {"processing": self.processor.prepare(), "camera": None, "aodo": None}
        self.camera.open()
        report["camera"] = self.camera.configure(
            rows_per_bline=self.processing.x_pixels,
            spectral_samples=self.processing.spectral_samples,
        )
        self.aodo.open()
        self._prepared = True
        self._status(f"engine prepared: backend={self.processor.backend_name}")
        return report

    def describe(self):
        return {
            "prepared": self._prepared,
            "processing": self.processor.describe(),
            "camera": self.camera.describe(),
            "aodo": self.aodo.describe(),
            "pipeline": None if self.pipeline is None else self.pipeline.describe(),
        }

    def close(self):
        try:
            self.camera.close()
        finally:
            self.aodo.close()
        self._prepared = False

    def stop(self):
        """从其它线程请求停止（安全：不 join 采集线程）。"""
        self.camera.stop()
        self.aodo.stop()

    def __enter__(self):
        self.prepare()
        return self

    def __exit__(self, *exc_info):
        self.close()
        return False

    # -- 采集回调 ---------------------------------------------------------- #
    def _on_package(self, package):
        if not self._accept_packages:
            return
        try:
            processed = self.processor.process(
                package.spectra, package_start=package.package_start
            )
            self.pipeline.add_package(processed, package.package_start)
            if self._finish_on_complete and self.pipeline.is_complete:
                # 扫描已收满：不再处理后续包（相机线程可能还在送）
                self._accept_packages = False
                self._finish_event.set()
        except Exception as error:  # 采集线程里不能抛：记录后由调用方重抛
            self._error = error
            self._log.exception("package processing failed")
            self._finish_event.set()

    def _status(self, message):
        self._log.info(message)
        emit_status(self._status_callback, message)

    # -- 扫描 -------------------------------------------------------------- #
    def single_scan(self, mode, save_path=None, timeout_s=None):
        """跑完一次有限扫描（B-scan 或 C-scan），返回 :class:`ScanResult`。"""
        if not self._prepared:
            raise HardwareError("prepare() must run before single_scan()")
        if mode in CONTINUOUS_MODES:
            raise ConfigurationError(f"{mode} is continuous; use repeat_scan()")
        y_lines = int(self.config.scan.y_lines) if mode in CSCAN_MODES else 1
        self._begin(mode, y_lines, finish_on_complete=True)
        self.aodo.configure(self.config.scan, self.processing, mode)
        self.aodo.start()
        started = time.perf_counter()
        self.camera.start(
            self._on_package,
            x_pixels=self.processing.x_pixels,
            bline_avg=self.processing.bline_avg,
            package_y_lines=max(1, min(int(self.engine_config.package_y_lines), y_lines)),
            y_lines=y_lines,
        )
        completed = self._finish_event.wait(timeout=self._resolve_timeout(timeout_s))
        self._accept_packages = False
        self.camera.stop()
        self.aodo.stop()
        if self._error is not None:
            raise self._error
        result = self._build_result(mode, completed, save_path)
        result.timings["wall_s"] = time.perf_counter() - started
        return result

    def repeat_scan(self, mode, duration_s=None, max_packages=None, save_path=None):
        """连续采集，直到: 达到 ``duration_s`` / ``max_packages`` / 缓冲填满 / 外部 stop()。"""
        if not self._prepared:
            raise HardwareError("prepare() must run before repeat_scan()")
        if mode not in CONTINUOUS_MODES:
            raise ConfigurationError(f"{mode} is not a continuous mode; use single_scan()")
        y_lines = (
            int(self.config.scan.y_lines) if mode in CSCAN_MODES else 1
        )
        self._begin(mode, y_lines, finish_on_complete=(y_lines > 1))
        self.aodo.configure(self.config.scan, self.processing, mode)
        self.aodo.start()
        started = time.perf_counter()
        self.camera.start(
            self._on_package,
            x_pixels=self.processing.x_pixels,
            bline_avg=self.processing.bline_avg,
            package_y_lines=1,
            y_lines=y_lines,
        )
        deadline = None if duration_s is None else started + float(duration_s)
        while self.camera.is_running:
            if self._error is not None:
                break
            # 单行缓冲（ContinuousBline）永远"满"，不能据此退出
            if y_lines > 1 and self.pipeline.is_complete:
                break
            if max_packages is not None and self.pipeline.packages >= int(max_packages):
                break
            if deadline is not None and time.perf_counter() >= deadline:
                break
            time.sleep(0.01)
        self._accept_packages = False
        self.camera.stop()
        self.aodo.stop()
        if self._error is not None:
            raise self._error
        result = self._build_result(mode, self.pipeline.is_complete, save_path)
        result.timings["wall_s"] = time.perf_counter() - started
        return result

    # -- 内部 -------------------------------------------------------------- #
    def _begin(self, mode, y_lines, finish_on_complete=True):
        self.pipeline = Pipeline(
            self.processing, y_lines, max_volume_gb=self.engine_config.max_volume_gb
        )
        self._finish_event.clear()
        self._error = None
        self._accept_packages = True
        self._finish_on_complete = bool(finish_on_complete)
        self._status(f"{mode}: {self.pipeline.describe()}")

    def _resolve_timeout(self, timeout_s):
        if timeout_s is not None:
            return float(timeout_s)
        waveform = getattr(self.aodo, "waveform", None)
        duration = 0.0 if waveform is None else float(waveform.duration_s)
        return max(30.0, duration * 3.0 + 30.0)

    def _build_result(self, mode, completed, save_path):
        pipeline = self.pipeline
        volume = pipeline.volume_filled().copy()
        notch_applied = False
        notch_backend = None
        if mode in CSCAN_MODES and self.processing.y_notch_enabled and volume.shape[0] >= 2:
            volume = self.processor.apply_y_notch(volume)
            notch_applied = True
            notch_backend = self.processor.backend_name
        aip = None
        if volume.ndim == 3 and volume.size:
            aip = np.max(np.abs(volume), axis=2).astype(np.float32)
        saved_path = None
        if save_path:
            saved_path = write_volume(
                save_path, volume, bigtiff=self.engine_config.tiff_bigtiff
            )
        return ScanResult(
            mode=str(mode),
            completed=bool(completed and pipeline.is_complete),
            filled_y=int(pipeline.filled_y),
            rows=int(pipeline.rows),
            packages=int(pipeline.packages),
            backend=self.processor.backend_name,
            volume=volume if volume.size else None,
            aip=aip,
            bline=pipeline.bline(),
            y_notch_applied=notch_applied,
            y_notch_backend=notch_backend,
            saved_path=saved_path,
            metadata={
                "package_y_lines": int(self.engine_config.package_y_lines),
                "keep_complex": bool(self.processing.keep_complex),
                "depth_start": int(self.processing.depth_start),
                "depth_pixels": int(self.processing.resolved_depth_pixels),
                "spectral_aline_avg": int(pipeline.processing.aline_avg),
                "spectral_bline_avg": int(pipeline.processing.bline_avg),
            },
        )
