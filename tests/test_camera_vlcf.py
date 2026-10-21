"""``.vlcf`` / ``.ccf`` 预设与路径解析（不需要相机）。"""

import pytest

from octsdk.camera import (
    available_presets,
    open_camera,
    presets_dir,
    read_vlcf_settings,
    resolve_ccf_path,
    resolve_vlcf_path,
    vlcf_int,
)
from octsdk.camera.simulated import SimulatedCameraDevice
from octsdk.config import CameraConfig
from octsdk.errors import HardwareError


def test_presets_are_packaged():
    names = available_presets()
    assert "12bit4tap.vlcf" in names
    settings = read_vlcf_settings(presets_dir() / "12bit4tap.vlcf")
    assert vlcf_int(settings, "PIXEL_DEPTH") == 12
    assert vlcf_int(settings, "TAP_NUMBER") == 4
    assert vlcf_int(settings, "IMAGE_WIDTH") == 1440
    assert vlcf_int(settings, "IMAGE_OFFSET_X", 0) == 0


def test_resolve_by_preset_and_by_bit_depth():
    by_preset = resolve_vlcf_path(CameraConfig(preset="12bit4tap"))
    by_geometry = resolve_vlcf_path(CameraConfig(bit_depth=12, taps=4))
    assert by_preset.name == "12bit4tap.vlcf"
    assert by_preset == by_geometry


def test_missing_vlcf_raises():
    with pytest.raises(HardwareError):
        resolve_vlcf_path(CameraConfig(vlcf_path="does-not-exist.vlcf"))
    with pytest.raises(HardwareError):
        resolve_vlcf_path(CameraConfig(preset="9bit7tap"))
    with pytest.raises(HardwareError):
        resolve_vlcf_path(CameraConfig(bit_depth=12, taps=8))


def test_ccf_is_paired_with_vlcf_by_name():
    with_ccf = resolve_ccf_path(resolve_vlcf_path(CameraConfig(preset="12bit4tap")))
    assert with_ccf is not None
    assert with_ccf.name == "12bit4tap.ccf"
    without_ccf = resolve_ccf_path(resolve_vlcf_path(CameraConfig(preset="8bit8tap")))
    assert without_ccf is None


def test_explicit_ccf_wins_and_is_validated(tmp_path):
    vlcf = resolve_vlcf_path(CameraConfig(preset="8bit8tap"))
    custom = tmp_path / "custom.ccf"
    custom.write_text("[DeviceFeature]\nSynchronizationMode = ExternalPulse\n", encoding="latin-1")
    assert resolve_ccf_path(vlcf, CameraConfig(ccf_path=str(custom))) == custom
    with pytest.raises(HardwareError):
        resolve_ccf_path(vlcf, CameraConfig(ccf_path=str(tmp_path / "nope.ccf")))


def test_open_camera_factory_returns_simulated():
    config = CameraConfig(simulate=True)
    device = open_camera(config)
    assert isinstance(device, SimulatedCameraDevice)
    device.close()
