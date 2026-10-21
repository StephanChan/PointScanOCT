"""相机设备基类：生命周期、回调、包切分与统计（与厂商 SDK 解耦）。

对外契约
--------
* :meth:`open` / :meth:`configure` / :meth:`start` / :meth:`stop` / :meth:`close`
* :meth:`start` 传入回调 ``on_package(SpectraPackage)``，**回调运行在采集线程**，
  里面不要做长耗时操作；回调抛异常会中止采集并把异常放进 :attr:`error`。
* 包切分：按 ``x_pixels × bline_avg``（一条 Y 行）对齐，凑够 ``package_y_lines`` 条
  Y 行就交出一包；``package_total_frames`` 给出整次扫描的原始行数（未知时为 ``None``）。
* 行序始终是 ``[Y][Bline重复][X位置][Aline重复]``，与 :class:`octsdk.OctProcessor` 的输入一致。
"""

import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from .._log import emit_status, get_logger
from ..errors import ConfigurationError, HardwareError


@dataclass(frozen=True)
class CameraFrameInfo:
    """一帧（一条 B-line）的元信息。"""

    frame_index: int
    rows: int
    samples: int
    dtype: str
    timestamp_s: float


@dataclass
class SpectraPackage:
    """一包原始光谱（``rows`` 行 = ``package_y_lines`` 条完整 Y 行）。"""

    spectra: np.ndarray
    package_start: int
    package_total_frames: int
    is_final: bool
    frame_times: tuple = ()
    metadata: dict = field(default_factory=dict)

    @property
    def rows(self):
        return int(self.spectra.shape[0])

    @property
    def samples(self):
        return int(self.spectra.shape[1])


class CameraDeviceBase(ABC):
    """相机设备的公共骨架。子类只需实现厂商相关的那几个钩子。"""

    def __init__(self, config, status_callback=None, logger_name="camera"):
        if not hasattr(config, "line_rate_hz"):
            raise ConfigurationError(f"{type(config).__name__} is not a CameraConfig")
        self.config = config
        self._status_callback = status_callback
        self._log = get_logger(logger_name)

        self._thread = None
        self._stop_event = threading.Event()
        self._on_package = None
        self._rows_per_bline = 1
        self._spectral_samples = 1
        self._group_rows = 1
        self._package_rows = 1
        self._package_total_frames = None
        self._rows_emitted = 0

        self.is_open = False
        self.error = None
        self.frames_delivered = 0
        self.packages_delivered = 0
        self.frames_dropped = 0
        self.zero_frames = 0

    # -- 生命周期 ---------------------------------------------------------- #
    def open(self):
        if self.is_open:
            return self.describe()
        report = self._open()
        self.is_open = True
        self._status(f"camera opened: {self.describe()}")
        return report

    def configure(self, rows_per_bline, spectral_samples):
        """设定几何：每条 B-line 的 A-line 行数、光谱采样数。"""
        if not self.is_open:
            raise HardwareError("camera must be opened before configure()")
        rows_per_bline = int(rows_per_bline)
        spectral_samples = int(spectral_samples)
        if rows_per_bline < 1 or spectral_samples < 1:
            raise ConfigurationError(
                "rows_per_bline and spectral_samples must be >= 1: "
                f"{rows_per_bline}, {spectral_samples}"
            )
        self._rows_per_bline = rows_per_bline
        self._spectral_samples = spectral_samples
        report = self._configure(rows_per_bline, spectral_samples)
        self._status(f"camera configured: {report}")
        return report

    def start(self, on_package, *, x_pixels, bline_avg, package_y_lines=1, y_lines=None):
        """开始采集；``on_package`` 在采集线程里被调用。"""
        if not self.is_open:
            raise HardwareError("camera must be opened and configured before start()")
        if self.is_running:
            raise HardwareError("camera is already streaming")
        x_pixels = int(x_pixels)
        bline_avg = int(bline_avg)
        package_y_lines = int(package_y_lines)
        if x_pixels < 1 or bline_avg < 1 or package_y_lines < 1:
            raise ConfigurationError(
                "x_pixels/bline_avg/package_y_lines must be >= 1: "
                f"{x_pixels}, {bline_avg}, {package_y_lines}"
            )
        if x_pixels != self._rows_per_bline:
            raise ConfigurationError(
                "rows_per_bline (from configure) must equal x_pixels: "
                f"{self._rows_per_bline} vs {x_pixels}"
            )
        if not callable(on_package):
            raise ConfigurationError("on_package must be callable")
        self._on_package = on_package
        self._group_rows = x_pixels
        self._package_rows = x_pixels * package_y_lines
        self._package_total_frames = None if y_lines is None else int(y_lines) * x_pixels
        self._rows_emitted = 0
        self.error = None
        self.frames_delivered = 0
        self.packages_delivered = 0
        self._stop_event.clear()
        self._start_stream()
        self._thread = threading.Thread(target=self._run, name="octsdk-camera", daemon=True)
        self._thread.start()

    def stop(self, timeout=5.0):
        """停止采集：等在途包交完（含最后一个不满的包）再返回。"""
        self._stop_event.set()
        self._request_stop()
        thread = self._thread
        if thread is not None:
            thread.join(max(0.0, float(timeout)))
            if thread.is_alive():
                self._status("camera stop timed out; worker thread is still alive")
            else:
                self._thread = None

    def close(self):
        self.stop()
        if self.is_open:
            self._close()
            self.is_open = False
            self._status("camera closed")

    @property
    def is_running(self):
        return self._thread is not None and self._thread.is_alive()

    def describe(self):
        payload = {
            "device": type(self).__name__,
            "simulate": bool(getattr(self.config, "simulate", False)),
            "is_open": self.is_open,
            "is_running": self.is_running,
            "line_rate_hz": float(self.config.line_rate_hz),
            "exposure_us": float(self.config.exposure_us),
            "bit_depth": int(self.config.bit_depth),
            "taps": int(self.config.taps),
            "rows_per_bline": int(self._rows_per_bline),
            "spectral_samples": int(self._spectral_samples),
            "frames_delivered": int(self.frames_delivered),
            "packages_delivered": int(self.packages_delivered),
            "frames_dropped": int(self.frames_dropped),
            "zero_frames": int(self.zero_frames),
        }
        payload.update(self._describe_extra())
        return payload

    # -- 子类钩子 ---------------------------------------------------------- #
    @abstractmethod
    def _open(self):
        """打开设备并返回描述信息。"""

    @abstractmethod
    def _configure(self, rows_per_bline, spectral_samples):
        """按几何配置设备并返回实际生效值。"""

    @abstractmethod
    def _start_stream(self):
        """开始抓帧（板卡/相机侧）。"""

    @abstractmethod
    def _stop_stream(self):
        """停止抓帧。"""

    @abstractmethod
    def _next_frame(self):
        """返回下一帧 ``(rows, samples)``；超时返回 ``None``。"""

    def _request_stop(self):
        """``stop()`` 里最先调用，用于打断阻塞的 ``_next_frame``（默认无操作）。"""

    def _close(self):
        """释放设备资源。"""

    def _describe_extra(self):
        return {}

    def _frame_dtype(self):
        return np.uint16

    # -- 采集线程 ---------------------------------------------------------- #
    def _run(self):
        dtype = self._frame_dtype()
        buffer = np.empty((self._package_rows, self._spectral_samples), dtype=dtype)
        filled = 0
        frame_index = 0
        frame_times = []
        try:
            while not self._stop_event.is_set():
                frame = self._next_frame()
                if frame is None:
                    # 相机/板卡暂时没有满帧（触发没来或行频不匹配）：计数后重试
                    self.zero_frames += 1
                    continue
                frame = self._normalize_frame(frame, dtype)
                self.frames_delivered += 1
                frame_index += 1
                frame_times.append(time.perf_counter())
                start = 0
                while start < frame.shape[0]:
                    take = min(self._package_rows - filled, frame.shape[0] - start)
                    buffer[filled:filled + take] = frame[start:start + take]
                    filled += take
                    start += take
                    if filled == self._package_rows:
                        self._emit(buffer, filled, frame_times, is_final=False,
                                   frame_index=frame_index)
                        filled = 0
                        frame_times = []
        except Exception as error:  # 回调或设备异常：中止采集并保留现场
            self.error = error
            self._log.exception("camera stream failed")
            self._status(f"camera stream failed: {error}")
        finally:
            if filled > 0:
                try:
                    self._emit(buffer, filled, frame_times, is_final=True,
                               frame_index=frame_index)
                except Exception:
                    self._log.exception("final package emit failed")
            try:
                self._stop_stream()
            except Exception:
                self._log.exception("stop_stream failed")

    def _normalize_frame(self, frame, dtype):
        array = np.asarray(frame)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        elif array.ndim == 3:
            array = array[:, :, 0]
        if array.ndim != 2:
            raise HardwareError(f"frame must be 2-D (rows, samples), got {array.shape}")
        if int(array.shape[1]) != self._spectral_samples:
            raise HardwareError(
                "frame spectral length mismatch: "
                f"{int(array.shape[1])} vs {self._spectral_samples}"
            )
        return array.astype(dtype, copy=False)

    def _emit(self, buffer, rows, frame_times, is_final, frame_index):
        package = SpectraPackage(
            spectra=np.array(buffer[:rows], copy=True),
            package_start=self._rows_emitted,
            package_total_frames=self._package_total_frames,
            is_final=bool(is_final),
            frame_times=tuple(frame_times),
            metadata={
                "rows_per_bline": int(self._rows_per_bline),
                "spectral_samples": int(self._spectral_samples),
                "frame_index": int(frame_index),
                "group_rows": int(self._group_rows),
                "package_rows": int(self._package_rows),
            },
        )
        self._rows_emitted += int(rows)
        self.packages_delivered += 1
        self._on_package(package)

    def _status(self, message):
        self._log.info(message)
        emit_status(self._status_callback, message)