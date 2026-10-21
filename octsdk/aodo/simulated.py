"""SIM AODO：不接 DAQ 也能验证波形生成与上层编排。

它记录下发的波形与启停时刻，便于在没有 NI/ART-DAQ 的机器上跑引擎逻辑。
"""

import time

from .base import AodoDeviceBase, resolve_ao_channels, sample_mode_for


class SimulatedAodoDevice(AodoDeviceBase):
    """只保存波形、不做 IO 的 AODO。"""

    def __init__(self, config, camera_config=None, status_callback=None):
        super().__init__(
            config, camera_config=camera_config, status_callback=status_callback,
            logger_name="aodo.sim",
        )
        self.started_at = None
        self.stopped_at = None

    def _open(self):
        return None

    def _configure(self, waveform):
        self.started_at = None
        self.stopped_at = None
        report = dict(waveform.describe())
        channel_spec, channel_order = resolve_ao_channels(
            self.config.board, self.config.ao_channel_x, self.config.ao_channel_y
        )
        report.update(
            {
                "continuous": sample_mode_for(waveform.mode),
                "ao_channel_spec": channel_spec,
                "channel_order": list(channel_order),
                "trigger_out": f"/{self.config.board}/{self.config.trigger_out_pfi}",
                "trigger_in": f"/{self.config.board}/{self.config.trigger_in_pfi}",
                "sync_do_line": f"{self.config.board}/{self.config.sync_do_line}",
            }
        )
        return report

    def _start(self):
        self.started_at = time.perf_counter()

    def _stop(self):
        self.stopped_at = time.perf_counter()

    def _describe_extra(self):
        return {
            "started_at": self.started_at,
            "stopped_at": self.stopped_at,
        }
