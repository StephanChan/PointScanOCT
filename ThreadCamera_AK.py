# -*- coding: utf-8 -*-
"""
AiKe/I-TEK IKap CameraLink camera worker for the OCT pipeline.

This thread follows the same queue and shared-memory contract as
ThreadCamera_HK.py. It configures the PL4KCL-200KT/IKap board, starts grab
before the galvo task, then stores triggered line frames into OCT raw-memory
slots for GPU/CPU processing.
"""

import ctypes
import os
import queue
import threading
import time
import traceback

import numpy as np
from PyQt5.QtCore import QThread

from ActionFields import DbackActionField
from CameraUi import downsample_spectral_axis, spectral_downsample, x_pixel_count
from IKapCamera_test import (
    IKapBoard,
    IKapC,
    IKapCDef,
    IKapLineScanGrabber,
    ENCODER_MULTIPLY_VALUES,
    GENERAL_INPUT_SAMPLE_MODE_VALUES,
    INTEGRATION_SOURCE_VALUES,
    check_itk,
    describe_itk_status,
    read_vlcf_settings,
    vlcf_int,
)


CONTINUOUS = 0x7FFFFFFF
AK_MEMORY_WRITE_METHOD = "direct"
AK_DEFAULT_LINE_RATE = 10000
AK_DEFAULT_PIXEL_FORMAT = "Mono8"
AK_DEFAULT_EXPOSURE_US = 10.0
AK_DEFAULT_GAIN = 1.0
AK_DEFAULT_DIGITAL_GAIN = 1.0
AK_DEFAULT_SAMPLES = 4096
AK_DEFAULT_HEIGHT = 1
AK_DEFAULT_FRAME_BUFFERS = 128
AK_DEFAULT_BOARD_TIMEOUT_MS = -1
AK_DEFAULT_FRAME_TIMEOUT_MS = 1000
AK_DEFAULT_TRIGGER_SELECTOR = "LineTrigger"
AK_DEFAULT_TRIGGER_SOURCE = "ShaftEncoder1"
AK_DEFAULT_TRIGGER_ACTIVATION = "RisingEdge"
AK_DEFAULT_OFFSET_W = 0
AK_DEFAULT_TAPS = 8
AK_CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "AKCameraConfig")
# 默认 .vlcf：软件启动时先按它设置一次相机（UI 的 VLCFPath_AK 可覆盖）。
AK_VLCF_DEFAULT = os.path.join(AK_CONFIG_DIR, "12bit4tap.vlcf")
# True: 板卡侧设置(ROI 宽度/偏移、位深、tap、CC1/积分触发源)一律以 .vlcf 为准，代码不再覆盖；
# False: 恢复旧行为（代码用 UI 值覆盖板卡）。用于一键回退对比。
AK_VLCF_AUTHORITATIVE = True
# True: 板卡 IMAGE_HEIGHT 也以 .vlcf 为准；False: 高度仍由扫描参数 AlinesPerBline 决定（推荐）。
AK_VLCF_HEIGHT_AUTHORITY = False
AK_PRINT_CAMERA_SETTINGS = False
AK_PRINT_GRAB_FORMAT = False
AK_PRINT_RAW_FRAME_SAMPLE = False


def ui_value(ui, names, default=None):
    for name in names:
        widget = getattr(ui, name, None)
        if widget is None:
            continue
        if hasattr(widget, "value"):
            return widget.value()
        if hasattr(widget, "currentText"):
            return widget.currentText()
        if hasattr(widget, "text"):
            return widget.text()
        if hasattr(widget, "isChecked"):
            return widget.isChecked()
    return default


def set_ui_value(ui, names, value):
    for name in names:
        widget = getattr(ui, name, None)
        if widget is None:
            continue
        if hasattr(widget, "setCurrentText"):
            widget.setCurrentText(str(value))
            return True
        if hasattr(widget, "setValue"):
            widget.setValue(value)
            return True
        if hasattr(widget, "setText"):
            widget.setText(str(value))
            return True
    return False


def normalize_trigger_source(value):
    text = str(value).replace(" ", "")
    mapping = {
        "ShaftEncoder1": "ShaftEncoder1",
        "GeneralInput1": "General Input1",
        "GeneralInput2": "General Input2",
        "Software": "Software",
    }
    return mapping.get(text, str(value))


def as_bytes(value):
    if isinstance(value, bytes):
        return value
    return str(value).encode("utf-8")


def decode_bytes(value):
    if isinstance(value, bytes):
        return value.split(b"\x00", 1)[0].decode("utf-8", errors="replace")
    return str(value)


def check_board(res, label):
    if res != IKapBoard.IK_RTN_OK:
        raise RuntimeError("%s failed: %#x" % (label, res))


def set_board_info(h_board, param, value, label, required=False):
    res = IKapBoard.IKapSetInfo(h_board, param, int(value))
    if res != IKapBoard.IK_RTN_OK and required:
        check_board(res, label)
    return res == IKapBoard.IK_RTN_OK


def get_board_info(h_board, param, label):
    res, value = IKapBoard.IKapGetInfo(h_board, param)
    check_board(res, label)
    return value


def set_camera_enum(h_camera, name, value, required=False):
    res = IKapC.ItkDevFromString(h_camera, as_bytes(name), as_bytes(value))
    if res != IKapCDef.ITKSTATUS_OK and required:
        raise RuntimeError("Set camera enum %s=%s failed: %#x" % (name, value, res))
    return res == IKapCDef.ITKSTATUS_OK


def set_camera_enum_any(h_camera, names, value, required=False):
    for name in names:
        if set_camera_enum(h_camera, name, value, required=False):
            return True
    if required:
        raise RuntimeError("Set camera enum %s=%s failed." % ("/".join(names), value))
    return False


def get_camera_enum(h_camera, names):
    for name in names:
        res, value = IKapC.ItkDevToString(h_camera, as_bytes(name))
        if res == IKapCDef.ITKSTATUS_OK:
            return True, decode_bytes(value)
    return False, None


def set_camera_float(h_camera, names, value, required=False):
    for name in names:
        res = IKapC.ItkDevSetDouble(h_camera, as_bytes(name), float(value))
        if res == IKapCDef.ITKSTATUS_OK:
            return True
    if required:
        raise RuntimeError("Set camera float %s=%g failed." % ("/".join(names), value))
    return False


def get_camera_float(h_camera, names):
    for name in names:
        res, value = IKapC.ItkDevGetDouble(h_camera, as_bytes(name))
        if res == IKapCDef.ITKSTATUS_OK:
            return True, value
    return False, None


def set_camera_int(h_camera, names, value, required=False):
    for name in names:
        res = IKapC.ItkDevSetInt64(h_camera, as_bytes(name), int(value))
        if res == IKapCDef.ITKSTATUS_OK:
            return True
    if required:
        raise RuntimeError("Set camera int %s=%d failed." % ("/".join(names), int(value)))
    return False


AK_PIXEL_FORMATS = {
    8: ("Mono8", IKapBoard.IKP_DATA_FORMAT_VAL_8Bit),
    10: ("Mono10", IKapBoard.IKP_DATA_FORMAT_VAL_10Bit),
    12: ("Mono12", IKapBoard.IKP_DATA_FORMAT_VAL_12Bit),
    14: ("Mono14", IKapBoard.IKP_DATA_FORMAT_VAL_14Bit),
    16: ("Mono16", IKapBoard.IKP_DATA_FORMAT_VAL_16Bit),
}
AK_PGA_GAIN_VALUES = {
    "x1": 1,
    "x2": 2,
    "x5": 5,
    "x10": 10,
}
AK_CAMERA_TAPS = {
    2: ("TwoTap", "Two Tap", "2Tap", "2"),
    4: ("FourTap", "Four Tap", "4Tap", "4"),
    8: ("EightTap", "Eight Tap", "8Tap", "8"),
    10: ("TenTap", "Ten Tap", "10Tap", "10"),
}
AK_BOARD_TAPS = (1, 2, 3, 4, 6, 8, 10)
AK_SENSOR_TAP_VALUES = tuple(AK_CAMERA_TAPS.keys())
AK_NO_10TAP_BIT_DEPTHS = (10, 12)
AK_VLCF_FILES = {
    (8, 8): "8bit8tap.vlcf",
    (8, 10): "8bit10tap.vlcf",
    (10, 8): "10bit8tap.vlcf",
    (12, 4): "12bit4tap.vlcf",
}

# .vlcf 的行触发源 -> UI 触发组合（以 .vlcf 为准回填 UI）
# INTEGRATION_TRIGGER_SOURCE: 0=Internal, 1/2=GeneralInput1/2, 5=ShaftEncoder1, 7/8=BoardSync1/2, 9=Software, 10=CameraSync
AK_VLCF_TRIGGER_SOURCE_NAMES = {
    0: "Internal",
    1: "GeneralInput1",
    2: "GeneralInput2",
    5: "ShaftEncoder1",
    7: "BoardSync1",
    8: "BoardSync2",
    9: "Software",
    10: "CameraSync",
}
AK_VLCF_TRIGGER_SOURCE_UI = {
    0: ("FreeRun", "Off", None),  # Internal：板卡内部行触发 = 自由运行
    1: ("LineTrigger", "On", "GeneralInput1"),
    2: ("LineTrigger", "On", "GeneralInput2"),
    5: ("LineTrigger", "On", "ShaftEncoder1"),
    7: ("LineTrigger", "On", None),
    8: ("LineTrigger", "On", None),
    9: ("SoftwareTrigger", "On", "Software"),
    10: ("LineTrigger", "On", None),
}


def ak_bit_depth(pixel_format):
    text = str(pixel_format).replace(" ", "").replace("-", "").lower()
    for bit_depth in (16, 14, 12, 10, 8):
        if str(bit_depth) in text:
            return bit_depth
    return 8


def ak_pixel_format_settings(pixel_format):
    bit_depth = ak_bit_depth(pixel_format)
    return AK_PIXEL_FORMATS.get(bit_depth, AK_PIXEL_FORMATS[8])


def normalize_taps(value, pixel_format=None):
    text = str(value).strip().lower().replace(" ", "")
    for suffix in ("taps", "tap"):
        if text.endswith(suffix):
            text = text[: -len(suffix)]
    word_values = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "six": 6,
        "eight": 8,
        "ten": 10,
    }
    try:
        taps = int(float(text))
    except ValueError:
        taps = word_values.get(text, AK_DEFAULT_TAPS)

    allowed = AK_SENSOR_TAP_VALUES
    if pixel_format is not None and ak_bit_depth(pixel_format) in AK_NO_10TAP_BIT_DEPTHS:
        allowed = tuple(tap for tap in allowed if tap != 10)
    return min(allowed, key=lambda item: abs(item - taps))


def set_camera_sensor_taps(h_camera, taps):
    for value in AK_CAMERA_TAPS.get(int(taps), (str(int(taps)),)):
        if set_camera_enum_any(
            h_camera,
            ("SensorTaps", "Sensor Taps", "SensorTap", "Sensor Tap"),
            value,
            required=False,
        ):
            return True
    return False


def ak_vlcf_path(pixel_format, taps):
    bit_depth = ak_bit_depth(pixel_format)
    taps = normalize_taps(taps, pixel_format)
    file_name = AK_VLCF_FILES.get((bit_depth, taps))
    if not file_name:
        return None
    return os.path.join(AK_CONFIG_DIR, file_name)


def normalize_pga_gain(value):
    text = str(value).strip().lower().replace(" ", "")
    if not text:
        return "x1"
    text = text.replace("gain", "")
    if text in AK_PGA_GAIN_VALUES:
        return text
    if text.startswith("x") and text in AK_PGA_GAIN_VALUES:
        return text
    try:
        gain_value = float(text)
    except ValueError:
        return "x1"
    nearest = min((1.0, 2.0, 5.0, 10.0), key=lambda item: abs(item - gain_value))
    return "x%d" % int(nearest)


def pga_gain_numeric(value):
    normalized = normalize_pga_gain(value)
    if normalized in AK_PGA_GAIN_VALUES:
        return AK_PGA_GAIN_VALUES[normalized]
    text = str(value).lower().replace("gain", "").replace("x", "").strip()
    try:
        return float(text)
    except ValueError:
        return AK_DEFAULT_GAIN


def set_pga_gain(h_camera, value):
    numeric = pga_gain_numeric(value)
    candidates = []
    for candidate in (
        normalize_pga_gain(value),
        str(int(numeric)) if float(numeric).is_integer() else str(numeric),
        "Gain%gx" % numeric,
    ):
        if candidate not in candidates:
            candidates.append(candidate)
    for candidate in candidates:
        if set_camera_enum_any(h_camera, ("PGAGain", "PGA Gain"), candidate, required=False):
            return True
    return False


class OCTIKapGrabber(IKapLineScanGrabber):
    """IKap grabber that pushes every callback frame into a FIFO."""

    def __init__(self, max_queue_frames=4096):
        super().__init__()
        self.frame_queue = queue.Queue(maxsize=max(8, int(max_queue_frames)))
        self._frame_lock = threading.Lock()

    def clear_frame_queue(self):
        while True:
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                return

    def _push_frame(self, frame):
        with self._frame_lock:
            self.last_frame = frame
            self.frame_ready_count += 1
        try:
            self.frame_queue.put_nowait(frame)
        except queue.Full:
            self.frame_lost_count += 1
            try:
                self.frame_queue.get_nowait()
                self.frame_queue.put_nowait(frame)
            except queue.Empty:
                pass

    def on_frame_ready(self, _):
        frame_index = self.current_frame_index.value
        res, status = IKapBoard.IKapGetBufferStatus(self.h_board, self.current_frame_index)
        if res != IKapBoard.IK_RTN_OK:
            print("IKapGetBufferStatus failed: %#x" % res)
            return
        if status.uFull == 1:
            res, frame_size = IKapBoard.IKapGetInfo(self.h_board, IKapBoard.IKP_FRAME_SIZE)
            if res != IKapBoard.IK_RTN_OK:
                print("IKP_FRAME_SIZE failed: %#x" % res)
                return
            p_user_buffer = ctypes.c_void_p(None)
            IKapBoard.IKapGetBufferAddress(self.h_board, self.current_frame_index, p_user_buffer)
            if p_user_buffer.value:
                raw = (frame_size * ctypes.c_ubyte).from_address(p_user_buffer.value)
                dtype = np.uint8 if self.depth == 8 else np.uint16
                array = np.frombuffer(raw, dtype=dtype).copy()
                expected = self.width * self.height * self.channel_count
                if array.size >= expected:
                    if self.channel_count == 1:
                        frame = array[:expected].reshape(self.height, self.width)
                    else:
                        frame = array[:expected].reshape(self.height, self.width, self.channel_count)
                else:
                    frame = array
                self.print_raw_frame_sample(frame)
                self._push_frame(frame)
        res, frame_count = IKapBoard.IKapGetInfo(self.h_board, IKapBoard.IKP_FRAME_COUNT)
        if res == IKapBoard.IK_RTN_OK and frame_count > 0:
            self.current_frame_index.value = (frame_index + 1) % int(frame_count)

    def start_grab(self, frame_count, timeout_ms=-1):
        if not self.is_board_open():
            raise RuntimeError("IKap board is not open.")
        self.current_frame_index = ctypes.c_uint(0)
        self.frame_ready_count = 0
        self.frame_lost_count = 0
        self.timeout_count = 0
        self.last_frame = None
        self.clear_frame_queue()

        set_board_info(self.h_board, IKapBoard.IKP_FRAME_COUNT, frame_count, "IKP_FRAME_COUNT", required=True)
        set_board_info(self.h_board, IKapBoard.IKP_TIME_OUT, timeout_ms, "IKP_TIME_OUT", required=True)
        set_board_info(
            self.h_board,
            IKapBoard.IKP_GRAB_MODE,
            IKapBoard.IKP_GRAB_NON_BLOCK,
            "IKP_GRAB_MODE",
            required=True,
        )
        set_board_info(
            self.h_board,
            IKapBoard.IKP_FRAME_TRANSFER_MODE,
            IKapBoard.IKP_FRAME_TRANSFER_SYNCHRONOUS_NEXT_EMPTY_WITH_PROTECT,
            "IKP_FRAME_TRANSFER_MODE",
            required=True,
        )

        frame_size = get_board_info(self.h_board, IKapBoard.IKP_FRAME_SIZE, "IKP_FRAME_SIZE")
        self.buffer_data = ctypes.create_string_buffer(frame_size)
        self.width = get_board_info(self.h_board, IKapBoard.IKP_IMAGE_WIDTH, "IKP_IMAGE_WIDTH")
        self.height = get_board_info(self.h_board, IKapBoard.IKP_IMAGE_HEIGHT, "IKP_IMAGE_HEIGHT")
        data_format = get_board_info(self.h_board, IKapBoard.IKP_DATA_FORMAT, "IKP_DATA_FORMAT")
        image_type = get_board_info(self.h_board, IKapBoard.IKP_IMAGE_TYPE, "IKP_IMAGE_TYPE")

        self.depth = 8 if data_format == 8 else 16
        if image_type == IKapBoard.IKP_IMAGE_TYPE_VAL_MONOCHROME:
            self.channel_count = 1
        elif image_type in (IKapBoard.IKP_IMAGE_TYPE_VAL_RGB, IKapBoard.IKP_IMAGE_TYPE_VAL_BGR):
            self.channel_count = 3
        else:
            self.channel_count = 4
        if AK_PRINT_GRAB_FORMAT:
            print(
                "AK grab format: data_format=%s, depth=%d, frame_size=%d, width=%d, height=%d"
                % (str(data_format), self.depth, int(frame_size), int(self.width), int(self.height))
            )

        self._register_callbacks()
        res = IKapBoard.IKapStartGrab(self.h_board, 0)
        check_board(res, "IKapStartGrab")
        self.grab_started = True

    def stop_grab(self):
        if not self.is_board_open():
            return
        for event_type in (
            IKapBoard.IKEvent_GrabStart,
            IKapBoard.IKEvent_GrabStop,
            IKapBoard.IKEvent_FrameReady,
            IKapBoard.IKEvent_TimeOut,
            IKapBoard.IKEvent_FrameLost,
        ):
            try:
                IKapBoard.IKapUnRegisterCallback(self.h_board, event_type)
            except Exception:
                pass
        try:
            if self.grab_started:
                IKapBoard.IKapStopGrab(self.h_board)
        except Exception:
            pass
        self.grab_started = False

    def on_grab_start(self, _):
        return None

    def on_grab_stop(self, _):
        return None

    def on_timeout(self, _):
        self.timeout_count += 1

    def on_frame_lost(self, _):
        self.frame_lost_count += 1

    def print_raw_frame_sample(self, frame):
        if not AK_PRINT_RAW_FRAME_SAMPLE:
            return
        frame = np.asarray(frame)
        if frame.ndim == 1:
            values = frame[2048:2050] if frame.size >= 2050 else frame[-min(2, frame.size):]
            row_index = 0
            col_start = 2048 if frame.size >= 2050 else max(0, frame.size - len(values))
        else:
            row_index = min(max(0, int(frame.shape[0]) // 2), int(frame.shape[0]) - 1)
            if frame.ndim == 3:
                line = frame[row_index, :, 0]
            else:
                line = frame[row_index, :]
            if line.size >= 2050:
                col_start = 2048
                values = line[2048:2050]
            else:
                col_start = max(0, line.size - 2)
                values = line[col_start:]
        print(
            "AK raw frame #%d sample: shape=%s, dtype=%s, row=%d, cols=%d:%d, values=%s"
            % (
                self.frame_ready_count + 1,
                str(frame.shape),
                str(frame.dtype),
                row_index,
                col_start,
                col_start + len(values),
                np.asarray(values).tolist(),
            )
        )


class Camera(QThread):
    def __init__(self):
        super().__init__()
        self.MemoryLoc = 0
        self.exit_message = "AiKe IKap camera thread exited."
        self.grabber = None
        self.is_open = False
        self.is_grabbing = False
        self.memory_write_method = AK_MEMORY_WRITE_METHOD
        self._ikap_initialized = False
        self.SIM = False
        self.AlinesPerBline = AK_DEFAULT_HEIGHT
        self.NSamples_AK = AK_DEFAULT_SAMPLES
        self.SpectralDS = 1
        self.ProcessedSamples = AK_DEFAULT_SAMPLES
        self.BlinesPerAcq = 1
        self.loaded_vlcf_path = None
        # .vlcf 生效状态：解析出的设置 + 供相机侧对齐的取值
        self.vlcf_path = None
        self.vlcf_settings = {}
        self.effective_width = AK_DEFAULT_SAMPLES
        self.effective_height = AK_DEFAULT_HEIGHT
        self.effective_offset_x = AK_DEFAULT_OFFSET_W
        self.effective_pixel_depth = ak_bit_depth(AK_DEFAULT_PIXEL_FORMAT)
        self.effective_taps = AK_DEFAULT_TAPS
        self.zero_frames = 0
        # 相机侧配置文件（.ccf，触发模式 SynchronizationMode 由它决定）
        self.ccf_path = None
        self.camera_ccf_loaded = False
        self.camera_sync_mode = None

    def run(self):
        print("initializing AiKe IKap camera...")
        try:
            self.initCamera()
            self.GetExposure()
            self.GetGain()
            self.GetPixelDepth()
            self.GetTaps()
        except Exception as error:
            print("AiKe camera init failed, using camera simulation. Error:", error)
            setattr(self.ui, "_ak_camera_found", False)
            setattr(self.ui, "_hik_camera_found", False)
            self.grabber = None
            self.SIM = True
        self.QueueOut()

    def QueueOut(self):
        self.item = self.queue.get()
        while self.item.action != "exit":
            try:
                if self.item.action == "ConfigureBoard":
                    self.ConfigureBoard()
                elif self.item.action == "Acquire":
                    if self.grabber is not None and self.grabber.is_board_open():
                        self.Stream_on()
                        self.Acquire()
                        self.Stream_off()
                    else:
                        self.simData()
                elif self.item.action == "GetTemp":
                    self.GetTemp()
                elif self.item.action == "ApplyRuntimeParams":
                    self.ApplyRuntimeParameters()
                else:
                    self.emit_status(f"Unknown AiKe camera command: {self.item.action}")
            except Exception as error:
                message = "AiKe camera command failed. This action was skipped: " + str(error)
                self.emit_status(message)
                print(message)
                print(traceback.format_exc())
                try:
                    self.DbackQueue.put(DbackActionField(0, error=message))
                except Exception:
                    pass
            self.item = self.queue.get()
        self.Close()
        print(self.exit_message)
        self.emit_status(self.exit_message)

    def emit_status(self, message):
        if message is not None and hasattr(self, "ui_bridge"):
            self.ui_bridge.status_message.emit(str(message))

    def initCamera(self):
        res = IKapC.ItkManInitialize()
        check_itk(res, "ItkManInitialize")
        self._ikap_initialized = True

        frame_buffers = int(ui_value(self.ui, ["FrameBuf_AK"], AK_DEFAULT_FRAME_BUFFERS))
        self.grabber = OCTIKapGrabber(max_queue_frames=max(frame_buffers * 4, AK_DEFAULT_FRAME_BUFFERS))
        self.grabber.open_first_cameralink()
        # 启动即按默认/UI 选定的 .vlcf 设置一次相机（原来的 load_configuration("") 等于不加载）
        self.ApplyStartupConfiguration()
        self.is_open = True
        setattr(self.ui, "_ak_camera_found", True)
        # Existing AODO code uses this legacy flag to decide whether to emit
        # synchronized camera DO triggers. Keep it true when the AK camera exists.
        setattr(self.ui, "_hik_camera_found", True)

    def ConfigureBoard(self):
        # In OCT mode, one IKap frame is one B-line. Its height is the number
        # of triggered A-lines along X, so it follows the scan X pixel count.
        self.AlinesPerBline = max(1, x_pixel_count(self.ui))
        self.NSamples_AK = int(ui_value(self.ui, ["NSamples_AK"], AK_DEFAULT_SAMPLES))
        self.SpectralDS = spectral_downsample(self.ui)
        if self.NSamples_AK % self.SpectralDS != 0:
            raise ValueError(
                "SpectralDS_AK must divide NSamples_AK: "
                f"NSamples_AK={self.NSamples_AK}, SpectralDS_AK={self.SpectralDS}"
            )
        self.ProcessedSamples = self.NSamples_AK // self.SpectralDS
        x_pixels = x_pixel_count(self.ui)
        bline_avg = max(1, int(self.ui.BlineAVG.value()))

        acq_mode = self.ui.ACQMode.currentText()
        if acq_mode in ["FiniteBscan", "Xfrequency"]:
            self.BlinesPerAcq = x_pixels * bline_avg
        elif acq_mode in ["ContinuousBline", "ContinuousCscan"]:
            self.BlinesPerAcq = CONTINUOUS
        elif acq_mode in ["FiniteCscan"]:
            self.BlinesPerAcq = self.ui.Ypixels.value() * x_pixels * bline_avg
        else:
            self.BlinesPerAcq = x_pixels * bline_avg

        if self.grabber is not None and self.grabber.is_board_open():
            # UI 换了 .vlcf 时强制重载（UI 只置标志，真正加载在相机线程里做）
            vlcf_dirty = bool(getattr(self.ui, "_ak_vlcf_dirty", False))
            self.LoadAutoConfiguration(force=vlcf_dirty)
            if vlcf_dirty:
                setattr(self.ui, "_ak_vlcf_dirty", False)
            # 先设行频再设曝光：曝光受行周期上限约束
            self.SetLineRate()
            self.SetExposure()
            self.SetGain()
            self.SetPixelDepth()
            self.SetTaps()
            self.SetLineRate()
            self.SetROI()
            self.SetTrigger()
            if AK_PRINT_CAMERA_SETTINGS:
                self.print_configuration_readback()
        self.DbackQueue.put(0)

    def ResolveVlcfPath(self):
        """决定要用的 .vlcf：UI 浏览框 > 默认 12bit4tap.vlcf > 按位深/tap 的预设。"""
        ui_path = str(ui_value(self.ui, ["VLCFPath_AK", "VlcfPath_AK"], "") or "").strip()
        if ui_path:
            return os.path.abspath(ui_path)
        if os.path.isfile(AK_VLCF_DEFAULT):
            return AK_VLCF_DEFAULT
        pixel_format = str(ui_value(self.ui, ["PixelFormat_AK"], AK_DEFAULT_PIXEL_FORMAT))
        taps = normalize_taps(ui_value(self.ui, ["Taps", "Tap"], AK_DEFAULT_TAPS), pixel_format)
        vlcf_path = ak_vlcf_path(pixel_format, taps)
        return os.path.abspath(vlcf_path) if vlcf_path else None

    def LoadAutoConfiguration(self, force=False):
        """加载 .vlcf 并解析出相机侧要对齐的取值（force=True 时忽略缓存强制重载）。"""
        if self.grabber is None or not self.grabber.is_board_open():
            return False
        vlcf_path = self.ResolveVlcfPath()
        if not vlcf_path:
            message = "No AiKe .vlcf available; using current board configuration."
            print(message)
            self.emit_status(message)
            self.loaded_vlcf_path = None
            return False
        if not os.path.isfile(vlcf_path):
            message = "AiKe .vlcf not found: %s" % vlcf_path
            print(message)
            self.emit_status(message)
            self.loaded_vlcf_path = None
            return False
        if self.loaded_vlcf_path == vlcf_path and not force:
            return True
        self.grabber.load_configuration(vlcf_path)
        self.loaded_vlcf_path = vlcf_path
        self.vlcf_path = vlcf_path
        # 相机侧 .ccf（与 .vlcf 同目录同名）：决定触发模式 SynchronizationMode；
        # 其中与 .vlcf 冲突的 ROI/位深/tap 等随后由 SetROI/SetPixelDepth/SetTaps 按 .vlcf 覆盖。
        self.LoadCameraConfigurationFile()
        self.ApplyVlcfCameraDefaults()
        message = "Loaded AiKe .vlcf: %s (%s)" % (
            os.path.basename(vlcf_path),
            self.VlcfEffectiveText(),
        )
        print(message)
        self.emit_status(message)
        return True

    def ApplyVlcfCameraDefaults(self):
        """解析 .vlcf：板卡侧由文件决定，相机侧沿用同一批取值以保持两侧一致。"""
        settings = read_vlcf_settings(self.vlcf_path) if self.vlcf_path else {}
        self.vlcf_settings = settings
        depth = vlcf_int(settings, "PIXEL_DEPTH")
        width = vlcf_int(settings, "IMAGE_WIDTH")
        height = vlcf_int(settings, "IMAGE_HEIGHT")
        offset_x = vlcf_int(settings, "IMAGE_OFFSET_X")
        taps = vlcf_int(settings, "TAP_NUMBER")
        if depth:
            self.effective_pixel_depth = depth
        if width:
            self.effective_width = width
            # 光谱采样数跟随 .vlcf（UI 的 NSamples_AK 已锁定为只读显示）
            self.NSamples_AK = int(width)
            set_ui_value(self.ui, ["NSamples_AK"], int(width))
        if offset_x is not None:
            self.effective_offset_x = int(offset_x)
        if taps:
            self.effective_taps = normalize_taps(taps, self.effective_pixel_depth)
        if height and AK_VLCF_HEIGHT_AUTHORITY:
            self.effective_height = int(height)
        self.UpdateVlcfUiReadback()

    def VlcfEffectiveText(self):
        camera_format = ak_pixel_format_settings(self.effective_pixel_depth)[0]
        return "生效: %s / %d tap / W=%d / OFFSET_X=%d%s%s" % (
            camera_format,
            self.effective_taps,
            self.effective_width,
            self.effective_offset_x,
            self.VlcfTriggerText(),
            (" / 相机触发=%s" % self.camera_sync_mode) if self.camera_sync_mode else "",
        )

    def VlcfTriggerText(self):
        source_value = vlcf_int(self.vlcf_settings, "INTEGRATION_TRIGGER_SOURCE")
        if source_value is None:
            return ""
        return " / 行触发=%s(%s)" % (
            AK_VLCF_TRIGGER_SOURCE_NAMES.get(int(source_value), "?"),
            source_value,
        )

    def ReadCcfValue(self, key):
        """从相机 .ccf（INI 风格文本）里读一个字段值。"""
        if not self.ccf_path or not os.path.isfile(self.ccf_path):
            return None
        try:
            with open(self.ccf_path, "r", encoding="latin-1") as handle:
                for line in handle:
                    if "=" not in line:
                        continue
                    name, _, value = line.partition("=")
                    if name.strip() == key:
                        return value.strip()
        except Exception as error:
            print("Read .ccf failed:", error)
        return None

    def ReadCameraSynchronizationMode(self):
        """从相机读回 SynchronizationMode（.ccf 生效后的实际触发模式）。"""
        if self.grabber is None or not self.grabber.is_camera_open():
            return None
        ok, value = get_camera_enum(
            self.grabber.h_camera,
            ("SynchronizationMode", "Synchronization Mode"),
        )
        return str(value) if ok else None

    def LoadCameraConfigurationFile(self, ccf_path=None):
        """加载相机侧 .ccf（触发模式等相机参数以它为准；ROI 等随后由 .vlcf 覆盖）。"""
        self.camera_ccf_loaded = False
        path = ccf_path or (os.path.splitext(self.vlcf_path or "")[0] + ".ccf")
        if not path or not os.path.isfile(path):
            message = "Camera .ccf not found (%s); camera trigger keeps its current setting." % (
                path or "n/a"
            )
            print(message)
            self.emit_status(message)
            return False
        if self.grabber is None or not self.grabber.is_camera_open():
            message = "Camera control handle is not open; .ccf not loaded (board-only mode)."
            print(message)
            self.emit_status(message)
            return False
        res = IKapC.ItkDevLoadConfigurationFromFile(self.grabber.h_camera, path.encode("utf-8"))
        if res != IKapCDef.ITKSTATUS_OK:
            message = "ItkDevLoadConfigurationFromFile failed: %#x (%s)" % (
                res,
                describe_itk_status(res),
            )
            print(message)
            self.emit_status(message)
            return False
        self.ccf_path = path
        self.camera_ccf_loaded = True
        message = "Loaded camera .ccf: %s (SynchronizationMode=%s)" % (
            os.path.basename(path),
            self.ReadCameraSynchronizationMode() or self.ReadCcfValue("SynchronizationMode") or "?",
        )
        print(message)
        self.emit_status(message)
        return True

    def ApplyCameraTriggerUi(self):
        """触发回填：优先用相机 .ccf 的 SynchronizationMode（IKapCViewer 同一语义）。"""
        sync_mode = self.ReadCameraSynchronizationMode() or self.ReadCcfValue("SynchronizationMode")
        if sync_mode:
            self.camera_sync_mode = sync_mode
            set_ui_value(self.ui, ["TriggerSelector_AK"], sync_mode)
            set_ui_value(
                self.ui,
                ["TriggerON_AK"],
                "Off" if "internal" in sync_mode.lower() else "On",
            )
            message = "相机触发模式(.ccf): %s → UI %s" % (sync_mode, "Off" if "internal" in sync_mode.lower() else "On")
            print(message)
            self.emit_status(message)
            return
        # 没有 .ccf 时退化为按 .vlcf 的行触发源回填
        source_value = vlcf_int(self.vlcf_settings, "INTEGRATION_TRIGGER_SOURCE")
        if source_value is None:
            return
        combo = AK_VLCF_TRIGGER_SOURCE_UI.get(int(source_value))
        if combo is None:
            message = "VLCF 行触发源=%s 在 UI 无对应组合，触发控件保持原值。" % source_value
            print(message)
            self.emit_status(message)
            return
        selector, trigger_on, trigger_source = combo
        set_ui_value(self.ui, ["TriggerSelector_AK"], selector)
        set_ui_value(self.ui, ["TriggerON_AK"], trigger_on)
        if trigger_source:
            set_ui_value(self.ui, ["TriggerSource_AK"], trigger_source)
        message = "VLCF 生效触发: %s / %s%s" % (
            selector,
            trigger_on,
            (" / " + trigger_source) if trigger_source else "",
        )
        print(message)
        self.emit_status(message)

    def UpdateVlcfUiReadback(self):
        """把 .vlcf 的生效值回填到只读显示字段（输入控件本身已由 UI 锁定）。"""
        camera_format = ak_pixel_format_settings(self.effective_pixel_depth)[0]
        set_ui_value(self.ui, ["PixelFormatDisplay_AK"], camera_format)
        set_ui_value(self.ui, ["Tapread", "TapRead", "TapsRead", "Tap_read"], int(self.effective_taps))
        # 触发控件：以相机 .ccf 的 SynchronizationMode 为准回填（没有 .ccf 时退回 .vlcf）
        self.ApplyCameraTriggerUi()
        text = self.VlcfEffectiveText()
        label = getattr(self.ui, "VLCFInfo_AK", None)
        if label is not None:
            try:
                # 只显示短状态，完整生效值放 tooltip，避免把界面拉长
                label.setText("已加载")
                label.setToolTip(text)
            except Exception:
                pass
        setattr(self.ui, "_ak_vlcf_effective", text)

    def warn_vlcf_mismatch(self, label, ui_value_raw, vlcf_value):
        """UI 输入与 .vlcf 生效值不一致时提示（以 .vlcf 为准）。"""
        if ui_value_raw is None or vlcf_value is None:
            return
        if str(ui_value_raw).strip() != str(vlcf_value).strip():
            message = "VLCF 生效值 %s=%s 与 UI 输入 %s 不一致，当前以 .vlcf 为准。" % (
                label,
                vlcf_value,
                ui_value_raw,
            )
            print(message)
            self.emit_status(message)

    def ApplyStartupConfiguration(self):
        """启动时先按（UI 选定的 / 默认的）.vlcf 把板卡与相机设置一次。"""
        if self.grabber is None or not self.grabber.is_board_open():
            return False
        if not self.LoadAutoConfiguration(force=True):
            message = "AiKe camera started without a .vlcf; board keeps its current configuration."
            print(message)
            self.emit_status(message)
            return False
        # 先设行频再设曝光：曝光受行周期上限约束
        self.SetLineRate()
        self.SetExposure()
        self.SetGain()
        self.SetPixelDepth()
        self.SetTaps()
        self.SetROI()
        self.SetTrigger()
        message = "AiKe camera configured at startup from %s (%s)" % (
            os.path.basename(self.vlcf_path or ""),
            self.VlcfEffectiveText(),
        )
        print(message)
        self.emit_status(message)
        return True

    def SetROI(self):
        if self.grabber is None or not self.grabber.is_board_open():
            return
        height = int(self.effective_height) if AK_VLCF_HEIGHT_AUTHORITY else self.AlinesPerBline
        if AK_VLCF_AUTHORITATIVE:
            # 宽度/偏移由 .vlcf 决定（12bit4tap.vlcf: W=1440, OFFSET_X=0），代码不再覆盖；
            # 高度是扫描参数（每条 B-line 的 A-line 数），是本模式下唯一的例外项。
            width = int(self.effective_width)
            offset_w = int(self.effective_offset_x)
            if not AK_VLCF_HEIGHT_AUTHORITY:
                set_board_info(self.grabber.h_board, IKapBoard.IKP_IMAGE_HEIGHT, height, "IKP_IMAGE_HEIGHT")
        else:
            width = int(self.NSamples_AK)
            offset_w = int(ui_value(self.ui, ["offsetW_AK"], AK_DEFAULT_OFFSET_W))
            set_board_info(self.grabber.h_board, IKapBoard.IKP_IMAGE_WIDTH, width, "IKP_IMAGE_WIDTH")
            set_board_info(self.grabber.h_board, IKapBoard.IKP_IMAGE_HEIGHT, height, "IKP_IMAGE_HEIGHT")
            set_board_info(self.grabber.h_board, IKapBoard.IKP_IMAGE_OFFSET_X, offset_w, "IKP_IMAGE_OFFSET_X")
        if self.grabber.is_camera_open():
            set_camera_int(self.grabber.h_camera, ("OffsetX",), offset_w, required=False)
            set_camera_int(self.grabber.h_camera, ("OffsetY",), 0, required=False)
            set_camera_int(self.grabber.h_camera, ("Width",), width, required=False)
            set_camera_int(self.grabber.h_camera, ("Height",), height, required=False)

    def SetTrigger(self):
        if self.camera_ccf_loaded:
            # 触发模式由相机 .ccf 决定、板卡侧由 .vlcf 决定：不再从 UI 写任何触发参数，只报告
            message = "Trigger comes from camera .ccf (%s); board side from .vlcf (%s)" % (
                self.camera_sync_mode or "?",
                os.path.basename(self.vlcf_path or ""),
            )
            print(message)
            self.emit_status(message)
            return
        selector = str(ui_value(self.ui, ["TriggerSelector_AK"], AK_DEFAULT_TRIGGER_SELECTOR))
        trigger_on = str(ui_value(self.ui, ["TriggerON_AK"], "On"))
        source = normalize_trigger_source(ui_value(self.ui, ["TriggerSource_AK"], AK_DEFAULT_TRIGGER_SOURCE))
        activation = str(ui_value(self.ui, ["TriggerActivation_AK"], AK_DEFAULT_TRIGGER_ACTIVATION))
        delay_ms = float(ui_value(self.ui, ["TriggerDelay_AK"], 0.0))

        if trigger_on == "Off":
            selector = "FreeRun"

        if selector == "FreeRun":
            self.configure_freerun()
        elif selector == "HardwareTrigger":
            self.configure_hardware_trigger(source, activation, delay_ms)
        elif selector == "SoftwareTrigger":
            self.configure_software_trigger(activation, delay_ms)
        else:
            self.configure_line_trigger(source, activation, delay_ms)

    def configure_freerun(self):
        if self.grabber.is_camera_open():
            set_camera_enum_any(
                self.grabber.h_camera,
                ("SynchronizationMode", "Synchronization Mode"),
                "InternalFreeRun",
                required=False,
            )
            set_camera_enum_any(
                self.grabber.h_camera,
                ("WorkingMode", "Working Mode"),
                "Free Run",
                required=False,
            )
            set_camera_enum(self.grabber.h_camera, "TriggerMode", "Off", required=False)

    def configure_hardware_trigger(self, source, activation, delay_ms):
        if self.grabber.is_camera_open():
            set_camera_enum_any(
                self.grabber.h_camera,
                ("SynchronizationMode", "Synchronization Mode"),
                "InternalFreeRun",
                required=False,
            )
            set_camera_enum_any(
                self.grabber.h_camera,
                ("WorkingMode", "Working Mode"),
                "Hardware Trigger",
                required=False,
            )
            set_camera_enum_any(
                self.grabber.h_camera,
                ("TriggerSource", "Trigger Source"),
                source,
                required=False,
            )
            set_camera_enum(self.grabber.h_camera, "TriggerActivation", activation, required=False)
            set_camera_float(self.grabber.h_camera, ("TriggerDelay", "TriggerDelayAbs"), delay_ms * 1000.0, required=False)

    def configure_software_trigger(self, activation, delay_ms):
        self.configure_board_line_trigger("Software", activation, "1")
        if self.grabber.is_camera_open():
            self.configure_camera_line_trigger(activation)
            set_camera_float(self.grabber.h_camera, ("TriggerDelay", "TriggerDelayAbs"), delay_ms * 1000.0, required=False)

    def configure_line_trigger(self, source, activation, delay_ms):
        integration_source = "ShaftEncoder1"
        if source in ("General Input1", "General Input2", "Software"):
            integration_source = source.replace(" ", "")
        # 板卡侧触发源（CC1_SOURCE / INTEGRATION_TRIGGER_SOURCE / INTEGRATION_METHOD / 编码器）
        # 由 .vlcf 决定；AK_VLCF_AUTHORITATIVE=False 时恢复成用代码覆盖。
        if AK_VLCF_AUTHORITATIVE:
            self.emit_status(
                "Board line trigger comes from .vlcf (%s)" % os.path.basename(self.vlcf_path or "")
            )
        else:
            self.configure_board_line_trigger(integration_source, activation, "1")
        if self.grabber.is_camera_open():
            self.configure_camera_line_trigger(activation)
            set_camera_float(self.grabber.h_camera, ("TriggerDelay", "TriggerDelayAbs"), delay_ms * 1000.0, required=False)

    def configure_board_line_trigger(self, integration_source, activation, encoder_multiply):
        if self.grabber is None or not self.grabber.is_board_open():
            return
        integration_value = INTEGRATION_SOURCE_VALUES.get(
            integration_source,
            INTEGRATION_SOURCE_VALUES["ShaftEncoder1"],
        )
        set_board_info(
            self.grabber.h_board,
            IKapBoard.IKP_CC1_SOURCE,
            IKapBoard.IKP_CC_SOURCE_VAL_INTEGRATION_SIGNAL1,
            "IKP_CC1_SOURCE",
            required=True,
        )
        set_board_info(
            self.grabber.h_board,
            IKapBoard.IKP_INTEGRATION_TRIGGER_SOURCE,
            integration_value,
            "IKP_INTEGRATION_TRIGGER_SOURCE",
            required=True,
        )
        set_board_info(
            self.grabber.h_board,
            IKapBoard.IKP_INTEGRATION_METHOD,
            IKapBoard.IKP_INTEGRATION_METHOD_VAL_1,
            "IKP_INTEGRATION_METHOD",
            required=True,
        )

        if integration_value == IKapBoard.IKP_INTEGRATION_TRIGGER_SOURCE_VAL_GENERAL_INPUT1:
            sample_mode = GENERAL_INPUT_SAMPLE_MODE_VALUES.get(
                activation,
                IKapBoard.IKP_GENERAL_INPUT_SAMPLE_MODE_VAL_RISING_EDGE,
            )
            set_board_info(
                self.grabber.h_board,
                IKapBoard.IKP_GENERAL_INPUT1_SAMPLE_MODE,
                sample_mode,
                "IKP_GENERAL_INPUT1_SAMPLE_MODE",
                required=True,
            )
            set_board_info(
                self.grabber.h_board,
                IKapBoard.IKP_GENERAL_INPUT1_TRIGGER_MODE,
                IKapBoard.IKP_GENERAL_INPUT_TRIGGER_MODE_VAL_EDGE,
                "IKP_GENERAL_INPUT1_TRIGGER_MODE",
                required=True,
            )

        if integration_value == IKapBoard.IKP_INTEGRATION_TRIGGER_SOURCE_VAL_GENERAL_INPUT2:
            sample_mode = GENERAL_INPUT_SAMPLE_MODE_VALUES.get(
                activation,
                IKapBoard.IKP_GENERAL_INPUT_SAMPLE_MODE_VAL_RISING_EDGE,
            )
            set_board_info(
                self.grabber.h_board,
                IKapBoard.IKP_GENERAL_INPUT2_SAMPLE_MODE,
                sample_mode,
                "IKP_GENERAL_INPUT2_SAMPLE_MODE",
                required=True,
            )
            set_board_info(
                self.grabber.h_board,
                IKapBoard.IKP_GENERAL_INPUT2_TRIGGER_MODE,
                IKapBoard.IKP_GENERAL_INPUT_TRIGGER_MODE_VAL_EDGE,
                "IKP_GENERAL_INPUT2_TRIGGER_MODE",
                required=True,
            )

        if integration_value == IKapBoard.IKP_INTEGRATION_TRIGGER_SOURCE_VAL_SHAFT_ENCODER1:
            multiply_value = ENCODER_MULTIPLY_VALUES.get(str(encoder_multiply), ENCODER_MULTIPLY_VALUES["1"])
            set_board_info(
                self.grabber.h_board,
                IKapBoard.IKP_SHAFT_ENCODER1_CHANNEL,
                IKapBoard.IKP_SHAFT_ENCODER_CHANNEL_VAL_A,
                "IKP_SHAFT_ENCODER1_CHANNEL",
                required=True,
            )
            set_board_info(
                self.grabber.h_board,
                IKapBoard.IKP_SHAFT_ENCODER1_MULTIPlY_FACTOR,
                multiply_value,
                "IKP_SHAFT_ENCODER1_MULTIPlY_FACTOR",
                required=True,
            )

    def configure_camera_line_trigger(self, activation):
        set_camera_enum_any(
            self.grabber.h_camera,
            ("SynchronizationMode", "Synchronization Mode"),
            "ExternalPulse",
            required=False,
        )
        set_camera_enum_any(
            self.grabber.h_camera,
            ("ExternalTriggerSource", "External Trigger Source"),
            "CC1",
            required=False,
        )
        set_camera_enum_any(
            self.grabber.h_camera,
            ("WorkingMode", "Working Mode"),
            "Free Run",
            required=False,
        )
        set_camera_enum(self.grabber.h_camera, "TriggerSelector", "FrameStart", required=False)
        set_camera_enum(self.grabber.h_camera, "TriggerMode", "Off", required=False)
        set_camera_enum(self.grabber.h_camera, "TriggerSelector", "LineStart", required=False)
        set_camera_enum(self.grabber.h_camera, "TriggerMode", "On", required=False)
        set_camera_enum(self.grabber.h_camera, "TriggerSource", "Line1", required=False)
        set_camera_enum(self.grabber.h_camera, "TriggerActivation", activation, required=False)

    def SetLineRate(self):
        line_rate = int(ui_value(self.ui, ["LineRate_AK"], AK_DEFAULT_LINE_RATE))
        if self.grabber is not None and self.grabber.is_camera_open():
            ok = set_camera_float(
                self.grabber.h_camera,
                ("AcquisitionLineRate", "LineRate", "AcquisitionFrameRate"),
                line_rate,
                required=False,
            )
            if not ok:
                self.emit_status("行频写入失败：相机不接受 AcquisitionLineRate=%s（详见控制台 Warning）" % line_rate)
            return ok
        return False

    def GetLineRate(self):
        if self.grabber is None or not self.grabber.is_camera_open():
            return None
        ok, value = get_camera_float(
            self.grabber.h_camera,
            ("AcquisitionLineRate", "LineRate", "AcquisitionFrameRate"),
        )
        return value if ok else None

    def SetPixelDepth(self):
        ui_pixel_format = str(ui_value(self.ui, ["PixelFormat_AK"], AK_DEFAULT_PIXEL_FORMAT))
        if AK_VLCF_AUTHORITATIVE:
            # 位深由 .vlcf 的 PIXEL_DEPTH 决定，代码不再写板卡 IKP_DATA_FORMAT
            camera_pixel_format, board_data_format = ak_pixel_format_settings(self.effective_pixel_depth)
            self.warn_vlcf_mismatch("PIXEL_DEPTH", ak_bit_depth(ui_pixel_format), self.effective_pixel_depth)
        else:
            camera_pixel_format, board_data_format = ak_pixel_format_settings(ui_pixel_format)
            if self.grabber is not None and self.grabber.is_board_open():
                set_board_info(
                    self.grabber.h_board,
                    IKapBoard.IKP_DATA_FORMAT,
                    board_data_format,
                    "IKP_DATA_FORMAT",
                    required=False,
                )
        if self.grabber is not None and self.grabber.is_camera_open():
            set_camera_enum(self.grabber.h_camera, "PixelFormat", camera_pixel_format, required=False)
        set_ui_value(self.ui, ["PixelFormatDisplay_AK"], camera_pixel_format)

    def GetPixelDepth(self):
        ui_pixel_format = str(ui_value(self.ui, ["PixelFormat_AK"], AK_DEFAULT_PIXEL_FORMAT))
        camera_pixel_format, _board_data_format = ak_pixel_format_settings(ui_pixel_format)
        set_ui_value(self.ui, ["PixelFormatDisplay_AK"], camera_pixel_format)

    def SetTaps(self):
        pixel_format = str(ui_value(self.ui, ["PixelFormat_AK"], AK_DEFAULT_PIXEL_FORMAT))
        requested_taps = normalize_taps(ui_value(self.ui, ["Taps", "Tap"], AK_DEFAULT_TAPS))
        if AK_VLCF_AUTHORITATIVE:
            # tap 数由 .vlcf 的 TAP_NUMBER 决定，代码不再写板卡 IKP_TAP_NUMBER
            taps = normalize_taps(self.effective_taps, self.effective_pixel_depth)
            self.warn_vlcf_mismatch("TAP_NUMBER", requested_taps, taps)
        else:
            taps = normalize_taps(requested_taps, pixel_format)
            if requested_taps != taps:
                message = (
                    "AiKe %d-bit mode does not use %d taps; using %d taps."
                    % (ak_bit_depth(pixel_format), requested_taps, taps)
                )
                print(message)
                self.emit_status(message)
                set_ui_value(self.ui, ["Taps", "Tap"], taps)
            if self.grabber is not None and self.grabber.is_board_open():
                set_board_info(
                    self.grabber.h_board,
                    IKapBoard.IKP_TAP_NUMBER,
                    taps,
                    "IKP_TAP_NUMBER",
                    required=False,
                )
        if self.grabber is not None and self.grabber.is_camera_open():
            set_camera_sensor_taps(self.grabber.h_camera, taps)
        self.GetTaps(default_taps=taps)

    def GetTaps(self, default_taps=None):
        taps = default_taps
        if self.grabber is not None and self.grabber.is_board_open():
            try:
                board_taps = get_board_info(self.grabber.h_board, IKapBoard.IKP_TAP_NUMBER, "IKP_TAP_NUMBER")
                if int(board_taps) in AK_BOARD_TAPS:
                    taps = int(board_taps)
            except Exception:
                pass
        if taps is None and self.grabber is not None and self.grabber.is_camera_open():
            ok, camera_taps = get_camera_enum(
                self.grabber.h_camera,
                ("SensorTaps", "Sensor Taps", "SensorTap", "Sensor Tap"),
            )
            if ok:
                taps = normalize_taps(camera_taps)
        if taps is None:
            taps = normalize_taps(ui_value(self.ui, ["Taps", "Tap"], AK_DEFAULT_TAPS))
        set_ui_value(self.ui, ["Tapread", "TapRead", "TapsRead", "Tap_read"], int(taps))

    def SetExposure(self):
        exposure_us = float(ui_value(self.ui, ["Exposure_AK"], AK_DEFAULT_EXPOSURE_US))
        ok = False
        if self.grabber is not None and self.grabber.is_camera_open():
            set_camera_enum(self.grabber.h_camera, "ExposureAuto", "Off", required=False)
            ok = set_camera_float(
                self.grabber.h_camera,
                ("ExposureTime", "ExposureTimeAbs", "ExposureTimeRaw"),
                exposure_us,
                required=False,
            )
            if not ok:
                self.emit_status("曝光写入失败：相机不接受 ExposureTime=%s us（详见控制台 Warning）" % exposure_us)
        self.GetExposure()
        return ok

    def GetExposure(self):
        if self.grabber is None or not self.grabber.is_camera_open():
            return None
        ok, value = get_camera_float(self.grabber.h_camera, ("ExposureTime", "ExposureTimeAbs", "ExposureTimeRaw"))
        if ok:
            set_ui_value(self.ui, ["ExposureDisplay_AK"], value)
            return value
        return None

    def SetGain(self):
        pga_gain = normalize_pga_gain(
            ui_value(
                self.ui,
                ["AnalogGain_AK", "PGAGain_AK", "PgaGain_AK", "AGain_AK", "DGain_AK"],
                AK_DEFAULT_GAIN,
            )
        )
        set_ui_value(
            self.ui,
            ["AnalogGainDisplay_AK", "PGAGainDisplay_AK", "PgaGainDisplay_AK", "AGainDisplay_AK", "DGainDisplay_AK"],
            pga_gain_numeric(pga_gain),
        )
        ok = False
        if self.grabber is not None and self.grabber.is_camera_open():
            set_camera_float(
                self.grabber.h_camera,
                ("DigitalGain", "Digital Gain"),
                AK_DEFAULT_DIGITAL_GAIN,
                required=False,
            )
            ok = set_pga_gain(self.grabber.h_camera, pga_gain)
            if not ok:
                self.emit_status("增益写入失败：相机不接受 PGAGain=%s（详见控制台 Warning）" % pga_gain)
        self.GetGain()
        return ok

    def GetGain(self):
        if self.grabber is None or not self.grabber.is_camera_open():
            return None
        ok, value = get_camera_enum(self.grabber.h_camera, ("PGAGain", "PGA Gain"))
        if ok:
            display_value = pga_gain_numeric(value)
            set_ui_value(
                self.ui,
                ["AnalogGainDisplay_AK", "PGAGainDisplay_AK", "PgaGainDisplay_AK", "AGainDisplay_AK", "DGainDisplay_AK"],
                display_value,
            )
            return display_value
        return None

    def ApplyRuntimeParameters(self):
        """UI 改动曝光/行频/增益后立即重设相机对应参数，并报告"请求值 vs 相机读回值"。

        .vlcf/.ccf 里的曝光与行频只作为初始值；这里始终以 UI 当前值为准。
        先写行频再写曝光：曝光受行周期上限约束，顺序反了会被相机钳位。
        正在采集时不打断采集，等下一次 ConfigureBoard（扫描开始）时自然生效。
        """
        if self.grabber is None or not self.grabber.is_camera_open():
            self.emit_status("相机未打开，曝光/行频/增益未下发。")
            return False
        if self.is_grabbing:
            self.emit_status("正在采集，曝光/行频/增益将在本次采集结束后生效。")
            return False
        line_ok = self.SetLineRate()
        exposure_ok = self.SetExposure()
        gain_ok = self.SetGain()
        exposure_actual = self.GetExposure()
        line_actual = self.GetLineRate()
        gain_actual = self.GetGain()
        message = (
            "相机参数 请求: exposure=%s us / line_rate=%s Hz / gain=%s  →  实际: exposure=%s us / "
            "line_rate=%s Hz / gain=%s%s"
            % (
                ui_value(self.ui, ["Exposure_AK"], AK_DEFAULT_EXPOSURE_US),
                ui_value(self.ui, ["LineRate_AK"], AK_DEFAULT_LINE_RATE),
                ui_value(
                    self.ui,
                    ["AnalogGain_AK", "PGAGain_AK", "PgaGain_AK", "AGain_AK", "DGain_AK"],
                    AK_DEFAULT_GAIN,
                ),
                "读回失败" if exposure_actual is None else exposure_actual,
                "读回失败" if line_actual is None else line_actual,
                "读回失败" if gain_actual is None else gain_actual,
                "" if (line_ok and exposure_ok and gain_ok) else "  [有写入失败，见上方 Warning]",
            )
        )
        print(message)
        self.emit_status(message)
        return bool(line_ok and exposure_ok and gain_ok)

    def GetTemp(self):
        return None

    def print_configuration_readback(self):
        """打印 .vlcf 生效值与板卡读回值（AK_PRINT_CAMERA_SETTINGS=True 时调用）。"""
        if self.grabber is None:
            return
        print("IKap .vlcf:", self.vlcf_path)
        for key in sorted(self.vlcf_settings):
            print("  %-32s = %s" % (key, self.vlcf_settings[key]))
        if not self.grabber.is_board_open():
            return
        for name, param in (
            ("IKP_DATA_FORMAT", IKapBoard.IKP_DATA_FORMAT),
            ("IKP_TAP_NUMBER", IKapBoard.IKP_TAP_NUMBER),
            ("IKP_IMAGE_WIDTH", IKapBoard.IKP_IMAGE_WIDTH),
            ("IKP_IMAGE_HEIGHT", IKapBoard.IKP_IMAGE_HEIGHT),
            ("IKP_IMAGE_OFFSET_X", IKapBoard.IKP_IMAGE_OFFSET_X),
            ("IKP_FRAME_SIZE", IKapBoard.IKP_FRAME_SIZE),
            ("IKP_CC1_SOURCE", IKapBoard.IKP_CC1_SOURCE),
            ("IKP_INTEGRATION_TRIGGER_SOURCE", IKapBoard.IKP_INTEGRATION_TRIGGER_SOURCE),
        ):
            res, value = IKapBoard.IKapGetInfo(self.grabber.h_board, param)
            if res == IKapBoard.IK_RTN_OK:
                print("  %-32s = %s" % (name, value))
            else:
                print("  %-32s = <read failed %#x>" % (name, res))

    def Stream_on(self):
        if self.grabber is not None and not self.is_grabbing:
            frame_buffers = max(1, int(ui_value(self.ui, ["FrameBuf_AK"], AK_DEFAULT_FRAME_BUFFERS)))
            timeout_ms = int(ui_value(self.ui, ["Timeout_AK"], AK_DEFAULT_BOARD_TIMEOUT_MS))
            self.grabber.clear_frame_queue()
            self.grabber.start_grab(frame_buffers, timeout_ms=timeout_ms)
            self.is_grabbing = True

    def Stream_off(self):
        if self.grabber is not None and self.is_grabbing:
            self.grabber.stop_grab()
            self.is_grabbing = False
            if self.zero_frames:
                message = (
                    "AiKe acquisition used %d zero (timeout) frames; check the .vlcf line-trigger source "
                    "(ShaftEncoder1/CC1) and the external trigger wiring." % self.zero_frames
                )
                print(message)
                self.emit_status(message)

    def Acquire(self):
        NBlines = self.Memory[0].shape[0]
        BlinesCount = 0
        self.zero_frames = 0
        frame_timeout_s = max(
            0.001,
            float(ui_value(self.ui, ["FrameTimeout_AK"], AK_DEFAULT_FRAME_TIMEOUT_MS)) / 1000.0,
        )
        current_slot = self.MemoryFreeQueue.get() if hasattr(self, "MemoryFreeQueue") else self.MemoryLoc
        package_start_time = time.perf_counter()
        self.DbackQueue.put(0)

        while BlinesCount < self.BlinesPerAcq and self.ui.RunButton.isChecked():
            try:
                frame = self.grabber.frame_queue.get(timeout=frame_timeout_s)
            except queue.Empty:
                self.zero_frames += 1
                print(
                    "AiKe camera timeout %.3f s (zero frames so far: %d)"
                    % (frame_timeout_s, self.zero_frames)
                )
                frame = np.zeros(
                    [self.AlinesPerBline, int(self.effective_width)],
                    dtype=self.Memory[current_slot].dtype,
                )

            frame = self.prepare_frame(frame, self.Memory[current_slot].dtype)
            frame = downsample_spectral_axis(frame, self.SpectralDS, axis=1)
            row_start = 0
            while row_start < frame.shape[0] and BlinesCount < self.BlinesPerAcq:
                slot_index = BlinesCount % NBlines
                rows_available = NBlines - slot_index
                rows_needed = self.BlinesPerAcq - BlinesCount
                rows_to_write = min(rows_available, frame.shape[0] - row_start, rows_needed)
                self.write_frame_rows_to_memory(
                    frame[row_start : row_start + rows_to_write],
                    current_slot,
                    slot_index,
                )
                row_start += rows_to_write
                BlinesCount += rows_to_write

                if BlinesCount % NBlines == 0:
                    package_start = BlinesCount - NBlines
                    self.DatabackQueue.put(
                        DbackActionField(
                            current_slot,
                            package_start=package_start,
                            package_frame_count=NBlines,
                            package_total_frames=self.BlinesPerAcq,
                            is_final=BlinesCount >= self.BlinesPerAcq,
                            profile={
                                "camera_package_start_s": package_start_time,
                                "camera_package_ready_s": time.perf_counter(),
                                "camera_package_fill_s": time.perf_counter() - package_start_time,
                            },
                        )
                    )
                    if hasattr(self, "MemoryFreeQueue"):
                        current_slot = self.MemoryFreeQueue.get()
                    else:
                        self.MemoryLoc = (self.MemoryLoc + 1) % self.memoryCount
                        current_slot = self.MemoryLoc
                    package_start_time = time.perf_counter()

            if self.ui.PauseButton.isChecked():
                self.Stream_off()
                while self.ui.PauseButton.isChecked() and self.ui.RunButton.isChecked():
                    time.sleep(0.5)
                self.Stream_on()

        partial_count = BlinesCount % NBlines
        if partial_count:
            package_start = BlinesCount - partial_count
            self.DatabackQueue.put(
                DbackActionField(
                    current_slot,
                    package_start=package_start,
                    package_frame_count=partial_count,
                    package_total_frames=self.BlinesPerAcq,
                    is_final=True,
                    profile={
                        "camera_package_start_s": package_start_time,
                        "camera_package_ready_s": time.perf_counter(),
                        "camera_package_fill_s": time.perf_counter() - package_start_time,
                    },
                )
            )

    def prepare_frame(self, frame, dtype):
        frame = np.asarray(frame)
        if frame.ndim == 3:
            frame = frame[:, :, 0]
        if frame.ndim == 1:
            frame = frame.reshape(1, -1)
        if frame.shape[0] != self.AlinesPerBline and frame.shape[1] == self.AlinesPerBline:
            frame = frame.T
        if frame.shape[1] != self.NSamples_AK:
            if frame.shape[0] == self.NSamples_AK:
                frame = frame.T
            else:
                raise ValueError(
                    "AiKe frame shape %s does not match NSamples_AK=%d"
                    % (str(frame.shape), self.NSamples_AK)
                )
        return frame.astype(dtype, copy=False)

    def write_frame_rows_to_memory(self, rows, memory_slot, frame_index):
        rows = np.asarray(rows)
        dest = self.Memory[memory_slot][frame_index : frame_index + rows.shape[0]]
        if dest.ndim == 2 and rows.ndim == 2 and rows.shape == dest.shape:
            np.copyto(dest, rows)
        elif dest.ndim == 3 and dest.shape[1] == 1 and rows.ndim == 2:
            np.copyto(dest[:, 0, :], rows)
        elif rows.shape == dest.shape:
            np.copyto(dest, rows)
        elif rows.T.shape == dest.shape:
            np.copyto(dest, rows.T)
        else:
            raise ValueError(
                "AiKe frame shape %s does not match destination %s"
                % (str(rows.shape), str(dest.shape))
            )

    def Close(self):
        if self.grabber is not None:
            self.Stream_off()
            self.grabber.close()
            self.grabber = None
        if self._ikap_initialized:
            IKapC.ItkManTerminate()
            self._ikap_initialized = False

    def simData(self):
        NBlines = self.Memory[0].shape[0]
        BlinesCount = 0
        current_slot = self.MemoryFreeQueue.get() if hasattr(self, "MemoryFreeQueue") else self.MemoryLoc
        package_start_time = time.perf_counter()
        sim_start_time = time.perf_counter()
        sim_line_rate = max(1.0, float(ui_value(self.ui, ["LineRate_AK"], AK_DEFAULT_LINE_RATE)))
        self.DbackQueue.put(0)

        while BlinesCount < self.BlinesPerAcq and self.ui.RunButton.isChecked():
            pixel_format = str(ui_value(self.ui, ["PixelFormat_AK"], AK_DEFAULT_PIXEL_FORMAT))
            dtype = np.uint8 if pixel_format == "Mono8" else np.uint16
            frame = np.zeros([self.AlinesPerBline, self.ProcessedSamples], dtype=dtype)
            row_start = 0
            while row_start < frame.shape[0] and BlinesCount < self.BlinesPerAcq:
                slot_index = BlinesCount % NBlines
                rows_available = NBlines - slot_index
                rows_needed = self.BlinesPerAcq - BlinesCount
                rows_to_write = min(rows_available, frame.shape[0] - row_start, rows_needed)
                self.write_frame_rows_to_memory(
                    frame[row_start : row_start + rows_to_write],
                    current_slot,
                    slot_index,
                )
                row_start += rows_to_write
                BlinesCount += rows_to_write
            if BlinesCount % 16 == 0:
                target_elapsed = BlinesCount / sim_line_rate
                sleep_s = target_elapsed - (time.perf_counter() - sim_start_time)
                if sleep_s > 0:
                    time.sleep(min(sleep_s, 0.05))
            if BlinesCount % NBlines == 0:
                package_start = BlinesCount - NBlines
                self.DatabackQueue.put(
                    DbackActionField(
                        current_slot,
                        package_start=package_start,
                        package_frame_count=NBlines,
                        package_total_frames=self.BlinesPerAcq,
                        is_final=BlinesCount >= self.BlinesPerAcq,
                        profile={
                            "camera_package_start_s": package_start_time,
                            "camera_package_ready_s": time.perf_counter(),
                            "camera_package_fill_s": time.perf_counter() - package_start_time,
                        },
                    )
                )
                if hasattr(self, "MemoryFreeQueue"):
                    current_slot = self.MemoryFreeQueue.get()
                else:
                    self.MemoryLoc = (self.MemoryLoc + 1) % self.memoryCount
                    current_slot = self.MemoryLoc
                package_start_time = time.perf_counter()

            if self.ui.PauseButton.isChecked():
                while self.ui.PauseButton.isChecked() and self.ui.RunButton.isChecked():
                    time.sleep(0.5)

        partial_count = BlinesCount % NBlines
        if partial_count:
            package_start = BlinesCount - partial_count
            self.DatabackQueue.put(
                DbackActionField(
                    current_slot,
                    package_start=package_start,
                    package_frame_count=partial_count,
                    package_total_frames=self.BlinesPerAcq,
                    is_final=True,
                    profile={
                        "camera_package_start_s": package_start_time,
                        "camera_package_ready_s": time.perf_counter(),
                        "camera_package_fill_s": time.perf_counter() - package_start_time,
                    },
                )
            )
