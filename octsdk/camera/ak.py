"""AK / I-TEK IKap CameraLink 相机适配器（M2b）。

运行时依赖客户自己安装的 I-TEK IKap SDK（DLL + Python 封装），本包只做定位与调用：
路径优先取 ``CameraConfig.sdk_root``，其次环境变量 ``IKAP_ROOT``，最后是默认安装目录。

抓帧采用"轮询缓冲状态"（``IKapGetBufferStatus`` → ``IKapGetBufferAddress``），与主程序
``ThreadCamera_AK`` 的回调体是同一套调用，但不注册 ctypes 回调，线程模型由基类负责。

参数权威顺序（与主程序一致）：
    板卡侧（位深/tap/ROI 宽度与偏移/行触发源） ← ``.vlcf``
    相机侧触发模式（SynchronizationMode）      ← ``.ccf``
    B-line 行数（IMAGE_HEIGHT）                ← 扫描几何（rows_per_bline）
    行频 / 曝光 / 增益 / 位深 / tap            ← ``CameraConfig``
"""

import ctypes
import os
import queue
import sys
import time

import numpy as np

from ..errors import HardwareError
from .base import CameraDeviceBase
from .vlcf import describe_vlcf, load_vlcf, vlcf_int

IKAP_ROOT_ENV = "IKAP_ROOT"
IKAP_DEFAULT_ROOT = r"C:\Program Files\I-TEK OptoElectronics\IKapLibrary"

#: 位深 -> 相机 PixelFormat 名（板卡侧常量按名字动态取，避免 SDK 版本差异）
CAMERA_PIXEL_FORMATS = {8: "Mono8", 10: "Mono10", 12: "Mono12", 14: "Mono14", 16: "Mono16"}
BOARD_DATA_FORMAT_CONSTANTS = {
    8: "IKP_DATA_FORMAT_VAL_8Bit",
    10: "IKP_DATA_FORMAT_VAL_10Bit",
    12: "IKP_DATA_FORMAT_VAL_12Bit",
    14: "IKP_DATA_FORMAT_VAL_14Bit",
    16: "IKP_DATA_FORMAT_VAL_16Bit",
}
CAMERA_TAPS = {
    2: ("TwoTap", "Two Tap", "2Tap", "2"),
    4: ("FourTap", "Four Tap", "4Tap", "4"),
    8: ("EightTap", "Eight Tap", "8Tap", "8"),
    10: ("TenTap", "Ten Tap", "10Tap", "10"),
}


def bootstrap_ikap(root=None):
    """把 IKap 的 DLL/Python 目录接进运行环境，返回 ``(IKapBoard, IKapC, IKapCDef)``。"""
    root = str(root or os.environ.get(IKAP_ROOT_ENV) or IKAP_DEFAULT_ROOT)
    examples_dir = os.path.join(root, "Examples", "Python")
    lib_dir = os.path.join(examples_dir, "IKapLib")
    for dll_dir in (
        os.path.join(root, "Dll", "C", "x64"),
        os.path.join(root, "GenICam", "bin", "Win64_x64"),
    ):
        if os.path.isdir(dll_dir):
            os.environ["PATH"] = dll_dir + os.pathsep + os.environ.get("PATH", "")
            if hasattr(os, "add_dll_directory"):
                try:
                    os.add_dll_directory(dll_dir)
                except OSError:
                    pass
    for path in (lib_dir, examples_dir):
        if os.path.isdir(path) and path not in sys.path:
            sys.path.append(path)
    try:
        import IKapBoard
        import IKapC
        import IKapCDef
    except Exception as error:
        raise HardwareError(
            f"IKap Python SDK not found (root={root}): {error}. "
            f"Install the I-TEK IKap SDK or set {IKAP_ROOT_ENV}/CameraConfig.sdk_root."
        ) from error
    return IKapBoard, IKapC, IKapCDef


class AkCameraDevice(CameraDeviceBase):
    """IKap CameraLink 线扫相机。"""

    def __init__(self, config, status_callback=None):
        super().__init__(config, status_callback=status_callback, logger_name="camera.ak")
        self._board = None
        self._camera = None
        self._defs = None
        self._h_board = None
        self._h_camera = ctypes.c_void_p(None)
        self._frame_queue = queue.Queue(maxsize=256)
        self._vlcf_path = None
        self._vlcf_settings = {}
        self._ccf_path = None
        self._camera_name = None
        self._frame_index = ctypes.c_uint(0)
        self._frame_count = 1
        self._frame_size = 0
        self._grabbing = False
        self.effective = {}

    # -- 打开 / 关闭 ------------------------------------------------------- #
    def _open(self):
        board, camera, defs = bootstrap_ikap(getattr(self.config, "sdk_root", None))
        self._board, self._camera, self._defs = board, camera, defs
        result, count = camera.ItkManGetDeviceCount()
        if result != defs.ITKSTATUS_OK or int(count) < 1:
            raise HardwareError(f"no IKap camera found (status={result:#x}, count={count})")
        for index in range(int(count)):
            result, info = camera.ItkManGetDeviceInfo(index)
            if result != defs.ITKSTATUS_OK:
                continue
            if info.DeviceClass != b"CameraLink" or info.SerialNumber == b"":
                continue
            result, cl_info = camera.ItkManGetCLDeviceInfo(index)
            if result != defs.ITKSTATUS_OK:
                raise HardwareError(f"ItkManGetCLDeviceInfo failed: {result:#x}")
            result, handle = camera.ItkDevOpen(index, defs.ITKDEV_VAL_ACCESS_MODE_EXCLUSIVE)
            if result != defs.ITKSTATUS_OK or not handle:
                raise HardwareError(
                    f"ItkDevOpen failed: {result:#x} "
                    "(close IKapViewer/other camera software and retry)"
                )
            self._h_camera = handle
            self._h_board = board.IKapOpen(cl_info.HostInterface, cl_info.BoardIndex)
            if self._h_board is None or self._h_board == -1:
                camera.ItkDevClose(handle)
                self._h_camera = ctypes.c_void_p(None)
                raise HardwareError("IKapOpen board failed")
            name = info.FullName
            self._camera_name = name.decode("utf-8", "ignore") if isinstance(name, bytes) else str(name)
            break
        else:
            raise HardwareError("no CameraLink camera with a serial number found")
        self._load_configuration_files()
        return self.describe()

    def _close(self):
        self._stop_stream()
        if getattr(self._h_camera, "value", 0) and self._camera is not None:
            try:
                self._camera.ItkDevClose(self._h_camera)
            except Exception:
                self._log.exception("ItkDevClose failed")
        if self._h_board not in (None, -1) and self._board is not None:
            try:
                self._board.IKapClose(self._h_board)
            except Exception:
                self._log.exception("IKapClose failed")
        self._h_board = None
        self._h_camera = ctypes.c_void_p(None)

    # -- 配置文件与参数 ----------------------------------------------------- #
    def _load_configuration_files(self):
        vlcf_path, settings, ccf_path = load_vlcf(self.config)
        self._vlcf_path, self._vlcf_settings, self._ccf_path = vlcf_path, settings, ccf_path
        result = self._board.IKapLoadConfigurationFromFile(
            self._h_board, str(vlcf_path).encode("utf-8")
        )
        if result != self._board.IK_RTN_OK:
            raise HardwareError(
                f"IKapLoadConfigurationFromFile failed: {result:#x} ({vlcf_path})"
            )
        self._vlcf_effective = describe_vlcf(settings)
        if ccf_path is not None and getattr(self._h_camera, "value", 0):
            result = self._camera.ItkDevLoadConfigurationFromFile(
                self._h_camera, str(ccf_path).encode("utf-8")
            )
            if result != self._defs.ITKSTATUS_OK:
                raise HardwareError(
                    f"ItkDevLoadConfigurationFromFile failed: {result:#x} ({ccf_path})"
                )
        suffix = f" + {ccf_path.name}" if ccf_path is not None else ""
        self._status(f"configuration loaded: {vlcf_path.name}{suffix}")

    def _configure(self, rows_per_bline, spectral_samples):
        config = self.config
        vlcf = self._vlcf_settings
        width = vlcf_int(vlcf, "IMAGE_WIDTH", spectral_samples)
        offset_x = vlcf_int(vlcf, "IMAGE_OFFSET_X", config.offset_x_px)
        if not config.vlcf_authoritative:
            width = int(spectral_samples)
            offset_x = int(config.offset_x_px)
            self._set_board_info("IKP_IMAGE_WIDTH", width)
            self._set_board_info("IKP_IMAGE_OFFSET_X", offset_x)
            self._set_board_data_format(int(config.bit_depth))
        height = int(rows_per_bline)
        if config.vlcf_height_authority:
            height = vlcf_int(vlcf, "IMAGE_HEIGHT", rows_per_bline)
        self._set_board_info("IKP_IMAGE_HEIGHT", height)

        pixel_format = CAMERA_PIXEL_FORMATS[int(config.bit_depth)]
        self._set_camera_enum(("PixelFormat",), pixel_format)
        self._set_camera_taps(int(config.taps))
        self._set_camera_int(("OffsetX",), offset_x, required=False)
        self._set_camera_int(("OffsetY",), 0, required=False)
        self._set_camera_int(("Width",), width, required=False)
        self._set_camera_int(("Height",), height, required=False)
        # 先行频后曝光：曝光上限受行周期约束
        self._set_camera_float(("AcquisitionLineRate", "LineRate"), float(config.line_rate_hz))
        self._set_camera_enum(("ExposureAuto",), "Off", required=False)
        self._set_camera_float(("ExposureTime", "ExposureTimeAbs"), float(config.exposure_us))
        self._set_camera_float(
            ("DigitalGain", "Digital Gain"), float(config.digital_gain), required=False
        )
        self._set_camera_enum(("PGAGain", "PGA Gain"), str(config.gain), required=False)
        if self._ccf_path is None:
            # 有 .ccf 时触发模式以相机配置文件为准，这里不再覆盖
            self._set_camera_enum(
                ("SynchronizationMode", "Synchronization Mode"),
                str(config.trigger_selector),
                required=False,
            )
            self._set_camera_enum(
                ("TriggerSource", "Trigger Source"), str(config.trigger_source), required=False
            )
            self._set_camera_enum(
                ("TriggerActivation",), str(config.trigger_activation), required=False
            )
            self._set_camera_float(
                ("TriggerDelay", "TriggerDelayAbs"),
                float(config.trigger_delay_us),
                required=False,
            )
        self._frame_size = int(self._get_board_info("IKP_FRAME_SIZE"))
        self.effective = {
            "vlcf": str(self._vlcf_path),
            "ccf": None if self._ccf_path is None else str(self._ccf_path),
            "vlcf_settings": self._vlcf_effective,
            "image_width": int(width),
            "image_height": int(height),
            "offset_x": int(offset_x),
            "pixel_format": pixel_format,
            "taps": int(config.taps),
            "line_rate_hz": self._get_camera_float(("AcquisitionLineRate", "LineRate")),
            "exposure_us": self._get_camera_float(("ExposureTime", "ExposureTimeAbs")),
            "trigger_from_ccf": self._ccf_path is not None,
        }
        return dict(self.effective)

    # -- 板卡 / 相机 低层读写 ---------------------------------------------- #
    def _set_board_info(self, constant_name, value):
        param = getattr(self._board, constant_name, None)
        if param is None:
            raise HardwareError(f"IKapBoard has no constant {constant_name}")
        result = self._board.IKapSetInfo(self._h_board, param, int(value))
        if result != self._board.IK_RTN_OK:
            raise HardwareError(f"IKapSetInfo({constant_name}={value}) failed: {result:#x}")
        return True

    def _get_board_info(self, constant_name):
        param = getattr(self._board, constant_name, None)
        if param is None:
            raise HardwareError(f"IKapBoard has no constant {constant_name}")
        result, value = self._board.IKapGetInfo(self._h_board, param)
        if result != self._board.IK_RTN_OK:
            raise HardwareError(f"IKapGetInfo({constant_name}) failed: {result:#x}")
        return value

    def _set_board_data_format(self, bit_depth):
        param = getattr(self._board, "IKP_DATA_FORMAT", None)
        value = getattr(self._board, BOARD_DATA_FORMAT_CONSTANTS[int(bit_depth)], None)
        if param is None or value is None:
            raise HardwareError(f"IKapBoard lacks IKP_DATA_FORMAT for {bit_depth}-bit")
        result = self._board.IKapSetInfo(self._h_board, param, value)
        if result != self._board.IK_RTN_OK:
            raise HardwareError(
                f"IKapSetInfo(IKP_DATA_FORMAT={bit_depth}bit) failed: {result:#x}"
            )
        return True

    def _set_camera_enum(self, names, value, required=False):
        last = None
        for name in names:
            last = self._camera.ItkDevFromString(
                self._h_camera, _as_bytes(name), _as_bytes(value)
            )
            if last == self._defs.ITKSTATUS_OK:
                return True
        if required:
            raise HardwareError(f"camera rejected {names[0]}={value}: {last:#x}")
        return False

    def _set_camera_float(self, names, value, required=False):
        last = None
        for name in names:
            last = self._camera.ItkDevSetDouble(self._h_camera, _as_bytes(name), float(value))
            if last == self._defs.ITKSTATUS_OK:
                return True
        if required:
            raise HardwareError(f"camera rejected {names[0]}={value}: {last:#x}")
        return False

    def _get_camera_float(self, names):
        for name in names:
            result, value = self._camera.ItkDevGetDouble(self._h_camera, _as_bytes(name))
            if result == self._defs.ITKSTATUS_OK:
                return float(value)
        return None

    def _set_camera_int(self, names, value, required=False):
        last = None
        for name in names:
            last = self._camera.ItkDevSetInt64(self._h_camera, _as_bytes(name), int(value))
            if last == self._defs.ITKSTATUS_OK:
                return True
        if required:
            raise HardwareError(f"camera rejected {names[0]}={value}: {last:#x}")
        return False

    def _set_camera_taps(self, taps):
        candidates = CAMERA_TAPS.get(int(taps), (str(int(taps)),))
        return self._set_camera_enum(("SensorTaps", "Sensor Taps"), candidates[0], required=False)

    # -- 抓帧 --------------------------------------------------------------- #
    def _start_stream(self):
        board = self._board
        self._frame_index = ctypes.c_uint(0)
        self._set_board_info("IKP_FRAME_COUNT", int(self.config.frame_buffers))
        timeout_param = getattr(board, "IKP_TIME_OUT", None)
        if timeout_param is not None:
            board.IKapSetInfo(self._h_board, timeout_param, int(self.config.board_timeout_ms))
        mode_param = getattr(board, "IKP_GRAB_MODE", None)
        if mode_param is not None:
            board.IKapSetInfo(self._h_board, mode_param, board.IKP_GRAB_NON_BLOCK)
        transfer_param = getattr(board, "IKP_FRAME_TRANSFER_MODE", None)
        transfer_mode = getattr(
            board, "IKP_FRAME_TRANSFER_SYNCHRONOUS_NEXT_EMPTY_WITH_PROTECT", None
        )
        if transfer_param is not None and transfer_mode is not None:
            board.IKapSetInfo(self._h_board, transfer_param, transfer_mode)
        result = board.IKapStartGrab(self._h_board, 0)
        if result != board.IK_RTN_OK:
            raise HardwareError(f"IKapStartGrab failed: {result:#x}")
        self._frame_count = int(self._get_board_info("IKP_FRAME_COUNT"))
        self._grabbing = True

    def _stop_stream(self):
        if not getattr(self, "_grabbing", False):
            return
        self._grabbing = False
        try:
            self._board.IKapStopGrab(self._h_board)
        except Exception:
            self._log.exception("IKapStopGrab failed")

    def _next_frame(self):
        """轮询缓冲状态取一帧；超时返回 ``None``（由基类计入 ``zero_frames``）。"""
        if not self._grabbing:
            return None
        board = self._board
        timeout_s = max(1e-3, float(self.config.frame_timeout_ms) / 1000.0)
        deadline = time.perf_counter() + timeout_s
        while time.perf_counter() < deadline:
            result, status = board.IKapGetBufferStatus(self._h_board, self._frame_index)
            if result != board.IK_RTN_OK:
                raise HardwareError(f"IKapGetBufferStatus failed: {result:#x}")
            if status.uFull == 1:
                pointer = ctypes.c_void_p(None)
                board.IKapGetBufferAddress(self._h_board, self._frame_index, pointer)
                if not pointer.value:
                    raise HardwareError("IKapGetBufferAddress returned a null pointer")
                raw = (int(self._frame_size) * ctypes.c_ubyte).from_address(pointer.value)
                dtype = np.uint8 if int(self.config.bit_depth) == 8 else np.uint16
                array = np.frombuffer(raw, dtype=dtype).copy()
                expected = self._rows_per_bline * self._spectral_samples
                if array.size < expected:
                    raise HardwareError(
                        f"frame too small: {array.size} samples, expected {expected}"
                    )
                frame = array[:expected].reshape(
                    self._rows_per_bline, self._spectral_samples
                )
                self._frame_index.value = (
                    int(self._frame_index.value) + 1
                ) % max(1, int(self._frame_count))
                return frame
            time.sleep(0.0005)
        return None

    def _describe_extra(self):
        return {
            "camera_name": self._camera_name,
            "vlcf_path": None if self._vlcf_path is None else str(self._vlcf_path),
            "ccf_path": None if self._ccf_path is None else str(self._ccf_path),
            "vlcf": getattr(self, "_vlcf_effective", None),
            "effective": self.effective,
        }


def _as_bytes(value):
    if isinstance(value, bytes):
        return value
    return str(value).encode("utf-8")
