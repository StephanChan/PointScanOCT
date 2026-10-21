"""相机设备层。

* :class:`SimulatedCameraDevice` —— 无硬件可用（SIM），用于联调/回归/演示；
* :class:`AkCameraDevice` —— I-TEK IKap CameraLink 实机（M2b，需要厂商 SDK）；
* :func:`open_camera` —— 按 ``CameraConfig.simulate`` 选一个。

``ak`` 模块是延迟导入的：没装 IKap SDK 的环境仍然可以 ``import octsdk`` 并用 SIM。
"""

from .base import CameraDeviceBase, CameraFrameInfo, SpectraPackage
from .simulated import SimulatedCameraDevice
from .vlcf import (
    PRESET_FILES,
    available_presets,
    describe_vlcf,
    load_vlcf,
    preset_path,
    presets_dir,
    read_vlcf_settings,
    resolve_ccf_path,
    resolve_vlcf_path,
    vlcf_float,
    vlcf_int,
)


def open_camera(config, status_callback=None, **simulated_options):
    """按配置返回相机设备对象（尚未 ``open()``）。"""
    if bool(getattr(config, "simulate", False)):
        return SimulatedCameraDevice(
            config, status_callback=status_callback, **simulated_options
        )
    from .ak import AkCameraDevice

    return AkCameraDevice(config, status_callback=status_callback)


__all__ = [
    "PRESET_FILES",
    "CameraDeviceBase",
    "CameraFrameInfo",
    "SimulatedCameraDevice",
    "SpectraPackage",
    "available_presets",
    "describe_vlcf",
    "load_vlcf",
    "open_camera",
    "preset_path",
    "presets_dir",
    "read_vlcf_settings",
    "resolve_ccf_path",
    "resolve_vlcf_path",
    "vlcf_float",
    "vlcf_int",
]
