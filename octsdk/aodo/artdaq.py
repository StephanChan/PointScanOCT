"""ART-DAQ 振镜 + 相机行触发设备（M3b）。

运行时依赖客户自己安装的 ART-DAQ（NI-DAQmx 兼容层）：路径优先取 ``AodoConfig.sdk_root``，
其次环境变量 ``ARTDAQ_PYTHON_LIB``，最后是默认安装目录。

任务结构（与主程序 ``ThreadAODO_art`` 一致）：
* AO 任务：多通道电压输出（X 快轴 / Y 慢轴），导出 ``START_TRIGGER`` 到 PFI3；
* DO 任务：单线相机行触发，起始触发挂在 PFI7（与 PFI3 对接），
  因此**必须先启动 DO、再启动 AO**，DO 才会跟随 AO 的起始触发。

AO 通道顺序由 ``ao_channel_x / ao_channel_y`` 决定：相邻通道用 ``board/ao0:1`` 这类
范围写法，写入顺序按实际通道编号重排（``X``/``Y`` 两个数组对不上会扫反）。
"""

import os
import sys
import types

import numpy as np

from ..config import AodoConfig
from ..errors import HardwareError
from .base import AodoDeviceBase, resolve_ao_channels, sample_mode_for

ARTDAQ_LIB_ENV = "ARTDAQ_PYTHON_LIB"
ARTDAQ_DEFAULT_LIB = r"C:\Program Files (x86)\ART Technology\ART-DAQ\Samples\Python\LIB"


def bootstrap_artdaq(root=None):
    """把 ART-DAQ 的 Python 库接进 ``sys.path`` 并返回常用符号集合。"""
    root = str(root or os.environ.get(ARTDAQ_LIB_ENV) or ARTDAQ_DEFAULT_LIB)
    if os.path.isdir(root) and root not in sys.path:
        sys.path.append(root)
    try:
        import artdaq as ni
        from artdaq.constants import AcquisitionType, Edge, Signal
        from artdaq.stream_writers import AnalogMultiChannelWriter, DigitalSingleChannelWriter
        try:
            from artdaq.constants import RegenerationMode
        except Exception:
            RegenerationMode = None
        try:
            from artdaq.constants import LineGrouping
        except Exception:
            LineGrouping = None
    except Exception as error:
        raise HardwareError(
            f"ART-DAQ Python library not found (root={root}): {error}. "
            f"Install ART-DAQ or set {ARTDAQ_LIB_ENV}/AodoConfig.sdk_root."
        ) from error
    return types.SimpleNamespace(
        ni=ni,
        AcquisitionType=AcquisitionType,
        Edge=Edge,
        Signal=Signal,
        RegenerationMode=RegenerationMode,
        LineGrouping=LineGrouping,
        AnalogMultiChannelWriter=AnalogMultiChannelWriter,
        DigitalSingleChannelWriter=DigitalSingleChannelWriter,
    )


class ArtDaqAodoDevice(AodoDeviceBase):
    """ART-DAQ 实现：一段波形 = 一次扫描。"""

    def __init__(self, config, camera_config=None, status_callback=None):
        super().__init__(
            config, camera_config=camera_config, status_callback=status_callback,
            logger_name="aodo.artdaq",
        )
        self._sdk = None
        self._ao_task = None
        self._do_task = None
        self._channel_spec = None
        self._channel_order = ("X", "Y")

    # -- 打开/关闭 --------------------------------------------------------- #
    def _open(self):
        self._sdk = bootstrap_artdaq(self.config.sdk_root)
        self._channel_spec, self._channel_order = self._resolve_channels()

    def _close(self):
        for task in (self._ao_task, self._do_task):
            if task is None:
                continue
            try:
                task.close()
            except Exception:
                self._log.exception("closing AODO task failed")
        self._ao_task = None
        self._do_task = None

    def _resolve_channels(self):
        return resolve_ao_channels(
            self.config.board, self.config.ao_channel_x, self.config.ao_channel_y
        )

    # -- 配置 / 启动 / 停止 ------------------------------------------------ #
    def _configure(self, waveform):
        sdk = self._sdk
        continuous = sample_mode_for(waveform.mode)
        sample_mode = (
            sdk.AcquisitionType.CONTINUOUS if continuous else sdk.AcquisitionType.FINITE
        )
        limit = float(self.config.ao_voltage_limit_v)

        ao_task = sdk.ni.Task("octsdk-ao")
        ao_task.ao_channels.add_ao_voltage_chan(
            physical_channel=self._channel_spec,
            min_val=-limit,
            max_val=limit,
            units=sdk.ni.constants.VoltageUnits.VOLTS,
        )
        ao_task.timing.cfg_samp_clk_timing(
            rate=float(waveform.sample_rate_hz),
            active_edge=sdk.Edge.RISING,
            sample_mode=sample_mode,
            samps_per_chan=int(waveform.samples),
        )
        _allow_regeneration(ao_task, sdk.RegenerationMode)
        ao_task.export_signals.export_signal(
            signal_id=sdk.Signal.START_TRIGGER,
            output_terminal=f"/{self.config.board}/{self.config.trigger_out_pfi}",
        )

        do_task = sdk.ni.Task("octsdk-do")
        self._add_do_channel(do_task, sdk.LineGrouping)
        do_task.timing.cfg_samp_clk_timing(
            rate=float(waveform.sample_rate_hz),
            sample_mode=sample_mode,
            samps_per_chan=int(waveform.do.size),
        )
        _allow_regeneration(do_task, sdk.RegenerationMode)
        do_task.triggers.start_trigger.cfg_dig_edge_start_trig(
            f"/{self.config.board}/{self.config.trigger_in_pfi}"
        )

        self._ao_task, self._do_task = ao_task, do_task
        report = dict(waveform.describe())
        report.update(
            {
                "continuous": continuous,
                "ao_channel_spec": self._channel_spec,
                "channel_order": list(self._channel_order),
                "trigger_out": f"/{self.config.board}/{self.config.trigger_out_pfi}",
                "trigger_in": f"/{self.config.board}/{self.config.trigger_in_pfi}",
                "sync_do_line": self._do_line(),
            }
        )
        return report

    def _start(self):
        sdk = self._sdk
        if self._ao_task is None or self._do_task is None:
            raise HardwareError("_configure() must run before _start()")
        ao_wave = np.ascontiguousarray(self._ordered_ao(), dtype=np.float64)
        do_wave = np.ascontiguousarray(self.waveform.do, dtype=np.uint32)
        ao_writer = sdk.AnalogMultiChannelWriter(self._ao_task.out_stream, auto_start=False)
        ao_writer.verify_array_shape = False
        ao_writer.write_many_sample(ao_wave)
        do_writer = sdk.DigitalSingleChannelWriter(self._do_task.out_stream, auto_start=False)
        do_writer.verify_array_shape = False
        do_writer.write_many_sample_port_u_32(do_wave)
        # DO 的起始触发挂在 PFI7（由 AO 的 PFI3 导出触发），所以先起 DO 再起 AO
        self._do_task.start()
        self._ao_task.start()

    def _stop(self):
        for task in (self._ao_task, self._do_task):
            if task is None:
                continue
            try:
                task.stop()
            except Exception:
                self._log.exception("stopping AODO task failed")

    # -- 内部 -------------------------------------------------------------- #
    def _ordered_ao(self):
        channels = {"X": self.waveform.ao[0], "Y": self.waveform.ao[1]}
        return np.vstack([channels[name] for name in self._channel_order])

    def _do_line(self):
        return f"{self.config.board}/{self.config.sync_do_line}"

    def _add_do_channel(self, task, line_grouping):
        lines = self._do_line()
        if line_grouping is None:
            task.do_channels.add_do_chan(lines=lines)
        else:
            task.do_channels.add_do_chan(
                lines=lines, line_grouping=line_grouping.CHAN_PER_LINE
            )

    def _describe_extra(self):
        return {
            "ao_channel_spec": self._channel_spec,
            "channel_order": None if self._channel_order is None else list(self._channel_order),
            "trigger_out": f"/{self.config.board}/{self.config.trigger_out_pfi}",
            "trigger_in": f"/{self.config.board}/{self.config.trigger_in_pfi}",
            "sample_rate_hz": None if self.waveform is None else self.waveform.sample_rate_hz,
        }


def _allow_regeneration(task, regeneration_mode):
    if regeneration_mode is None:
        return
    try:
        task.out_stream.regen_mode = regeneration_mode.ALLOW_REGENERATION
    except Exception:
        pass
