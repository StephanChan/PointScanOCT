"""相机设备层：SIM 包切分、与处理层联调（无需硬件）。"""

import ctypes
import time
import types

import numpy as np

from octsdk import OctConfig, OctProcessor, ProcessingConfig, ScanConfig
from octsdk.camera import SimulatedCameraDevice, open_camera
from octsdk.config import CameraConfig


def _sim_config(**overrides):
    params = {"bit_depth": 12, "line_rate_hz": 10000.0, "exposure_us": 4.2, "simulate": True}
    params.update(overrides)
    return CameraConfig(**params)


def _wait_for(predicate, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def test_sim_camera_emits_aligned_packages():
    camera = SimulatedCameraDevice(_sim_config(), pace=False, seed=1)
    camera.open()
    camera.configure(rows_per_bline=4, spectral_samples=64)
    packages = []
    camera.start(packages.append, x_pixels=4, bline_avg=1, package_y_lines=2)
    assert _wait_for(lambda: len(packages) >= 3)
    camera.stop()
    camera.close()

    first = packages[0]
    assert first.spectra.shape == (8, 64)
    assert first.spectra.dtype == np.uint16
    assert [package.package_start for package in packages[:3]] == [0, 8, 16]
    assert int(first.spectra.max()) <= 4095
    assert camera.frames_delivered > 0
    assert camera.describe()["simulate"] is True


def test_sim_camera_stop_is_idempotent():
    camera = open_camera(_sim_config(), pace=False)
    camera.open()
    camera.configure(rows_per_bline=2, spectral_samples=32)
    camera.start(lambda package: None, x_pixels=2, bline_avg=1)
    camera.stop()
    camera.stop()
    camera.close()
    camera.close()
    assert camera.is_running is False


def test_camera_counts_zero_frames():
    """相机超时（没有满帧）要计数：真机排查丢触发就靠这个数字。"""
    camera = SimulatedCameraDevice(_sim_config(), pace=False)
    camera.open()
    camera.configure(rows_per_bline=2, spectral_samples=32)
    frames = iter([None, None, np.ones((2, 32), dtype=np.uint16)])
    camera._next_frame = lambda: next(frames, None)
    packages = []
    camera.start(packages.append, x_pixels=2, bline_avg=1)
    assert _wait_for(lambda: len(packages) >= 1)
    camera.stop()
    camera.close()
    assert camera.zero_frames >= 2


def test_callback_error_surfaces_in_device():
    camera = SimulatedCameraDevice(_sim_config(), pace=False)
    camera.open()
    camera.configure(rows_per_bline=2, spectral_samples=32)

    def boom(_package):
        raise RuntimeError("consumer failed")

    camera.start(boom, x_pixels=2, bline_avg=1)
    assert _wait_for(lambda: camera.error is not None)
    assert "consumer failed" in str(camera.error)
    camera.close()


def test_sim_camera_feeds_processor_end_to_end():
    samples = 256
    camera = SimulatedCameraDevice(
        _sim_config(), pace=False, target_depth_fraction=0.25, seed=2
    )
    camera.open()
    camera.configure(rows_per_bline=4, spectral_samples=samples)
    packages = []
    camera.start(packages.append, x_pixels=4, bline_avg=1, package_y_lines=1)
    assert _wait_for(lambda: len(packages) >= 1)
    camera.stop()
    camera.close()

    processing = OctConfig(
        scan=ScanConfig(x_step_um=10.0, x_length_um=40.0),
        processing=ProcessingConfig(
            raw_spectral_samples=samples,
            x_pixels=4,
            amplification=1.0,
            depth_start=8,  # 相机输出非负计数 → 深度 0 有强 DC，真实用法也要裁掉
        ),
    ).resolved_processing()
    processor = OctProcessor(processing, backend="cpu")
    processor.prepare()
    result = processor.process(packages[0].spectra)
    peak = int(np.argmax(result.data.mean(axis=0))) + int(processing.depth_start)
    assert abs(peak - samples // 4) <= 1


# -- AK 适配器：假 SDK 验证参数映射与写入顺序（不需要真机） ------------------ #
class _FakeBoardModule:
    IK_RTN_OK = 0
    IKP_FRAME_SIZE = 101
    IKP_IMAGE_WIDTH = 102
    IKP_IMAGE_HEIGHT = 103
    IKP_IMAGE_OFFSET_X = 104
    IKP_DATA_FORMAT = 105
    IKP_DATA_FORMAT_VAL_12Bit = 12
    IKP_FRAME_COUNT = 106
    IKP_TIME_OUT = 107
    IKP_GRAB_MODE = 108
    IKP_GRAB_NON_BLOCK = 1
    IKP_FRAME_TRANSFER_MODE = 109
    IKP_FRAME_TRANSFER_SYNCHRONOUS_NEXT_EMPTY_WITH_PROTECT = 2

    def __init__(self):
        self.set_info = []
        self.vlcf = None

    def IKapOpen(self, host_interface, board_index):
        return 1

    def IKapClose(self, handle):
        return 0

    def IKapLoadConfigurationFromFile(self, handle, path):
        self.vlcf = path
        return 0

    def IKapSetInfo(self, handle, param, value):
        self.set_info.append((param, value))
        return 0

    def IKapGetInfo(self, handle, param):
        table = {self.IKP_FRAME_SIZE: 4096, self.IKP_FRAME_COUNT: 16}
        return 0, table.get(param, 0)

    def IKapStartGrab(self, handle, count):
        return 0

    def IKapStopGrab(self, handle):
        return 0


class _FakeCameraModule:
    def __init__(self):
        self.calls = []
        self.ccf = None

    def ItkManGetDeviceCount(self):
        return 0, 1

    def ItkManGetDeviceInfo(self, index):
        return 0, types.SimpleNamespace(
            DeviceClass=b"CameraLink", SerialNumber=b"SN1", FullName=b"FakeCam"
        )

    def ItkManGetCLDeviceInfo(self, index):
        return 0, types.SimpleNamespace(HostInterface=0, BoardIndex=0)

    def ItkDevOpen(self, index, mode):
        return 0, ctypes.c_void_p(1)

    def ItkDevClose(self, handle):
        return 0

    def ItkDevLoadConfigurationFromFile(self, handle, path):
        self.ccf = path
        return 0

    def ItkDevFromString(self, handle, name, value):
        self.calls.append((name.decode(), value.decode()))
        return 0

    def ItkDevSetDouble(self, handle, name, value):
        self.calls.append((name.decode(), value))
        return 0

    def ItkDevGetDouble(self, handle, name):
        return 0, 1234.5

    def ItkDevSetInt64(self, handle, name, value):
        self.calls.append((name.decode(), value))
        return 0


class _FakeDefs:
    ITKSTATUS_OK = 0
    ITKDEV_VAL_ACCESS_MODE_EXCLUSIVE = 1


def test_ak_configure_maps_config_onto_sdk(monkeypatch):
    from octsdk.camera import ak

    board = _FakeBoardModule()
    camera = _FakeCameraModule()
    monkeypatch.setattr(ak, "bootstrap_ikap", lambda root=None: (board, camera, _FakeDefs))

    device = ak.AkCameraDevice(CameraConfig(preset="12bit4tap"))
    device.open()
    report = device.configure(rows_per_bline=480, spectral_samples=1440)

    assert board.vlcf.endswith(b"12bit4tap.vlcf")
    assert camera.ccf.endswith(b"12bit4tap.ccf")
    assert (board.IKP_IMAGE_HEIGHT, 480) in board.set_info
    assert report["image_width"] == 1440
    assert report["offset_x"] == 0
    assert ("PixelFormat", "Mono12") in camera.calls
    assert ("SensorTaps", "FourTap") in camera.calls
    assert ("Height", 480) in camera.calls
    names = [call[0] for call in camera.calls]
    assert names.index("AcquisitionLineRate") < names.index("ExposureTime")
    assert "SynchronizationMode" not in names
    device.close()

