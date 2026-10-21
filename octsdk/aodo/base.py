"""AODO 设备基类：生命周期、波形持有与描述（与 DAQ 厂商解耦）。

对外契约
--------
* :meth:`open` → :meth:`configure(scan, processing, mode)` → :meth:`start` /
  :meth:`stop` / :meth:`close`；
* ``configure`` 内部调用 :func:`octsdk.aodo.waveform.build_waveform` 生成一次扫描的
  整段 AO/DO 波形，硬件层只负责把这段波形写进 DAQ。
"""

from abc import ABC, abstractmethod

from .._log import emit_status, get_logger
from ..config import AodoConfig, CONTINUOUS_MODES
from ..errors import ConfigurationError, HardwareError
from .waveform import build_waveform


class AodoDeviceBase(ABC):
    """振镜 + DO 设备骨架。"""

    def __init__(self, config, camera_config=None, status_callback=None, logger_name="aodo"):
        if not isinstance(config, AodoConfig):
            raise ConfigurationError(f"config must be AodoConfig, got {type(config)}")
        self.config = config
        self.camera_config = camera_config
        self._status_callback = status_callback
        self._log = get_logger(logger_name)
        self.waveform = None
        self.is_open = False
        self.is_running = False

    # -- 生命周期 ---------------------------------------------------------- #
    def open(self):
        if self.is_open:
            return self.describe()
        self._open()
        self.is_open = True
        self._status(f"aodo opened: {self.describe()}")
        return self.describe()

    def configure(self, scan, processing, mode, sample_rate_hz=None):
        """生成并下发一次扫描的波形；返回实际生效参数。"""
        if not self.is_open:
            raise HardwareError("aodo must be opened before configure()")
        rate = self._resolve_sample_rate(sample_rate_hz)
        self.waveform = build_waveform(scan, processing, self.config, mode, rate)
        report = self._configure(self.waveform)
        self._status(f"aodo configured: {report}")
        return report

    def start(self):
        if not self.is_open:
            raise HardwareError("aodo must be opened before start()")
        if self.waveform is None:
            raise HardwareError("aodo.configure() must be called before start()")
        if self.is_running:
            raise HardwareError("aodo is already running")
        self._start()
        self.is_running = True

    def stop(self):
        if not self.is_running:
            return
        self._stop()
        self.is_running = False

    def close(self):
        self.stop()
        if self.is_open:
            self._close()
            self.is_open = False
            self._status("aodo closed")

    # -- 描述 -------------------------------------------------------------- #
    def describe(self):
        payload = {
            "device": type(self).__name__,
            "simulate": bool(getattr(self.config, "simulate", False)),
            "is_open": self.is_open,
            "is_running": self.is_running,
            "board": self.config.board,
            "ao_channel_x": self.config.ao_channel_x,
            "ao_channel_y": self.config.ao_channel_y,
            "sync_do_line": self.config.sync_do_line,
            "waveform": None if self.waveform is None else self.waveform.describe(),
        }
        payload.update(self._describe_extra())
        return payload

    # -- 子类钩子 ---------------------------------------------------------- #
    @abstractmethod
    def _open(self):
        """打开 DAQ 设备。"""

    @abstractmethod
    def _configure(self, waveform):
        """按波形配置 DAQ 任务。"""

    @abstractmethod
    def _start(self):
        """写波形并启动任务。"""

    @abstractmethod
    def _stop(self):
        """停止任务。"""

    def _close(self):
        """释放设备。"""

    def _describe_extra(self):
        return {}

    def _resolve_sample_rate(self, sample_rate_hz):
        if sample_rate_hz is not None:
            return float(sample_rate_hz)
        if self.config.sample_rate_hz is not None:
            return float(self.config.sample_rate_hz)
        if self.camera_config is not None:
            return float(self.config.resolved_sample_rate_hz(self.camera_config))
        raise ConfigurationError(
            "sample rate is not resolved: pass sample_rate_hz, set AodoConfig.sample_rate_hz, "
            "or provide camera_config (default = 2 × line_rate_hz)"
        )

    def _status(self, message):
        self._log.info(message)
        emit_status(self._status_callback, message)


def parse_ao_index(channel_name):
    """``"ao1"`` → ``1``（非法名字直接报配置错误）。"""
    text = str(channel_name).strip().lower()
    if not text.startswith("ao") or not text[2:].isdigit():
        raise ConfigurationError(f"unsupported AO channel name: {channel_name!r} (expected aoN)")
    return int(text[2:])


def resolve_ao_channels(board, x_channel, y_channel):
    """返回 ``(通道串, 写入顺序)``。

    相邻通道用 ``board/ao0:1`` 这类范围写法，此时写入顺序按实际通道编号排列
    （例如 X=ao1、Y=ao0 → 顺序 ``["Y", "X"]``），否则 X/Y 会扫反。
    """
    x_index = parse_ao_index(x_channel)
    y_index = parse_ao_index(y_channel)
    if x_index == y_index:
        raise ConfigurationError("ao_channel_x and ao_channel_y must differ")
    low, high = min(x_index, y_index), max(x_index, y_index)
    if high - low == 1:
        order = ["X" if index == x_index else "Y" for index in range(low, high + 1)]
        return f"{board}/ao{low}:{high}", tuple(order)
    return f"{board}/ao{x_index},{board}/ao{y_index}", ("X", "Y")


def sample_mode_for(mode):
    """连续扫描模式返回 ``True``（DAQ 用 CONTINUOUS），否则 FINITE。"""
    return mode in CONTINUOUS_MODES
