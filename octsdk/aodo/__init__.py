"""AODO 设备层。

* :class:`SimulatedAodoDevice` —— 不接 DAQ，只生成/保存波形（联调、CI）；
* :class:`ArtDaqAodoDevice`  —— ART-DAQ 实机（M3b，需要厂商驱动）；
* :func:`open_aodo` —— 按 ``AodoConfig.simulate`` 选一个；
* :func:`build_waveform` —— 纯函数波形生成（不需要任何设备）。
"""

from .base import AodoDeviceBase, sample_mode_for
from .simulated import SimulatedAodoDevice
from .waveform import (
    SUPPORTED_MODES,
    WaveformSpec,
    build_waveform,
    camera_trigger_train,
    cos_flyback,
    scan_voltage_limits,
)


def open_aodo(config, camera_config=None, status_callback=None):
    """按配置返回 AODO 设备对象（尚未 ``open()``）。"""
    if bool(getattr(config, "simulate", False)):
        return SimulatedAodoDevice(config, camera_config=camera_config,
                                   status_callback=status_callback)
    from .artdaq import ArtDaqAodoDevice

    return ArtDaqAodoDevice(config, camera_config=camera_config,
                            status_callback=status_callback)


__all__ = [
    "SUPPORTED_MODES",
    "AodoDeviceBase",
    "SimulatedAodoDevice",
    "WaveformSpec",
    "build_waveform",
    "camera_trigger_train",
    "cos_flyback",
    "open_aodo",
    "sample_mode_for",
    "scan_voltage_limits",
]
