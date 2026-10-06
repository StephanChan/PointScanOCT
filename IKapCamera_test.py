# -*- coding: utf-8 -*-
"""I-TEK IKap CameraLink line-scan camera test with ART-DAQ line triggers.

This follows the I-TEK CameraLink LineScanCamera examples:
- Open a CameraLink camera through IKapC.
- Open the matching IKap PCIe board through IKapBoard.
- Configure external line trigger.
- Start non-blocking grab and generate line-trigger pulses with an ART-DAQ DO.
"""

import argparse
import ctypes
import os
import sys
import time

import numpy as np


IKAP_ROOT = r"C:\Program Files\I-TEK OptoElectronics\IKapLibrary"
IKAP_EXAMPLES_PYTHON = os.path.join(IKAP_ROOT, "Examples", "Python")
IKAP_LIB_DIR = os.path.join(IKAP_EXAMPLES_PYTHON, "IKapLib")
IKAP_DLL_DIR = os.path.join(IKAP_ROOT, "Dll", "C", "x64")
IKAP_GENICAM_DIR = os.path.join(IKAP_ROOT, "GenICam", "bin", "Win64_x64")

# 项目内的 .vlcf 配置文件目录。
AK_CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "AKCameraConfig")
# 默认使用 12bit4tap.vlcf 配置采集卡：板卡侧设置（几何尺寸、像素位深、tap 数、
# CC1/积分触发源、编码器通道与倍频、通用输入采样模式）以该文件为准。
DEFAULT_VLCF_PATH = os.path.join(AK_CONFIG_DIR, "12bit4tap.vlcf")

# .vlcf 中决定板卡行为的字段（板卡设置的权威来源）。
VLCF_KEYS_OF_INTEREST = (
    "PIXEL_DEPTH",
    "TAP_NUMBER",
    "TAP_ARRANGEMENT",
    "PIXEL_CLOCK",
    "IMAGE_WIDTH",
    "IMAGE_HEIGHT",
    "IMAGE_TYPE",
    "IMAGE_OFFSET_X",
    "CC1_SOURCE",
    "INTEGRATION_TRIGGER_SOURCE",
    "INTEGRATION_METHOD",
    "SHAFT_ENCODER_CHANNEL",
    "SHAFT_ENCODER_MULTIPLY_FACTOR",
    "GENERAL_INPUT1_SAMPLE_MODE",
    "GENERALINPUT1_TRIGGER_MODE",
    "GRAB_MODE",
    "TIMEOUT",
    "INTERNAL_TRIGGER_FREQUENCY",
    "PIXEL_FORMAT_PFNC",
)

# GenICam PFNC 编码 -> 位深，用于校验 .vlcf 的 Pixel_Format_PFNC 与 PIXEL_DEPTH 是否一致。
# 例：Mono8=17301505(0x01080001)，Mono10=17825795，Mono12=17825797，Mono16=17825799。
PFNC_MONO_BIT_DEPTH = {
    17301505: 8,
    17825795: 10,
    17825797: 12,
    17825799: 16,
}

ARTDAQ_PYTHON_LIB_DIR = (
    r"C:\Program Files (x86)\ART Technology\ART-DAQ\Samples\Python\LIB"
)


def bootstrap_paths():
    for dll_dir in (IKAP_DLL_DIR, IKAP_GENICAM_DIR):
        if os.path.isdir(dll_dir):
            os.environ["PATH"] = dll_dir + os.pathsep + os.environ.get("PATH", "")
            if hasattr(os, "add_dll_directory"):
                os.add_dll_directory(dll_dir)
    if IKAP_LIB_DIR not in sys.path:
        sys.path.append(IKAP_LIB_DIR)
    if ARTDAQ_PYTHON_LIB_DIR not in sys.path:
        sys.path.append(ARTDAQ_PYTHON_LIB_DIR)


bootstrap_paths()

import IKapBoard
import IKapC
import IKapCDef
import artdaq as ni
from artdaq.constants import AcquisitionType, Edge
from artdaq.stream_writers import DigitalSingleChannelWriter


def as_bytes(value):
    if isinstance(value, bytes):
        return value
    return str(value).encode("utf-8")


def decode_bytes(value):
    if isinstance(value, bytes):
        return value.split(b"\x00", 1)[0].decode("utf-8", errors="replace")
    return str(value)


def read_vlcf_settings(vlcf_path):
    """读取 .vlcf（INI 风格文本）中与板卡/相机设置相关的字段。

    返回 {大写字段名: 去掉首尾空白的字符串}。文件不存在时返回空字典。
    """
    settings = {}
    if not vlcf_path or not os.path.isfile(vlcf_path):
        return settings
    wanted = {key.upper() for key in VLCF_KEYS_OF_INTEREST}
    with open(vlcf_path, "r", encoding="latin-1") as handle:
        for line in handle:
            if "=" not in line:
                continue
            name, _, value = line.partition("=")
            name = name.strip().upper()
            if name in wanted:
                settings[name] = value.strip()
    return settings


def vlcf_int(settings, key, default=None):
    """从 read_vlcf_settings() 的结果里取整数值（兼容 9000.0 这类写法）。"""
    value = settings.get(key.upper())
    if value is None:
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


# GenICam 特征类型（ITK_FEATURE_TYPE），用于特征自检时打印可读类型。
ITK_FEATURE_TYPE_NAMES = {
    0: "Unknown",
    1: "Int64",
    2: "Double",
    3: "String",
    4: "Bool",
    5: "Command",
    6: "Enum",
    7: "Category",
}

# 相机侧关键特征名：用于 --dump-camera-features 与失败诊断。
CAMERA_FEATURE_PROBE_NAMES = (
    "Width",
    "Height",
    "OffsetX",
    "OffsetY",
    "PixelFormat",
    "SensorTaps",
    "SensorTap",
    "ExposureTime",
    "ExposureAuto",
    "AcquisitionLineRate",
    "SynchronizationMode",
    "WorkingMode",
    "TriggerSelector",
    "TriggerMode",
    "TriggerSource",
    "TriggerActivation",
    "ExternalTriggerSource",
    "FrameStart",
    "LineStart",
    "PGAGain",
    "DigitalGain",
    "DigitalOffset",
)


def describe_itk_status(res):
    """把 IKapC 的状态码翻译成可读名称（低 16 位是 ITKSTATUS_* 的 id）。"""
    low_status = int(res) & 0xFFFF
    for name in dir(IKapCDef):
        if not name.startswith("ITKSTATUS_"):
            continue
        value = getattr(IKapCDef, name)
        if isinstance(value, int) and value == low_status:
            return "%s(id=%#x)" % (name, low_status)
    return "id=%#x" % low_status


def print_frame_stats(frames):
    """打印每帧的基本统计，用于快速判断图像是否正常（全黑/饱和/错位）。"""
    for index, frame in enumerate(frames):
        data = np.asarray(frame)
        if data.size == 0:
            print("  frame #%d: empty" % index)
            continue
        print(
            "  frame #%d: shape=%s dtype=%s min=%s max=%s mean=%.2f std=%.2f"
            % (
                index,
                data.shape,
                data.dtype,
                int(data.min()),
                int(data.max()),
                float(data.mean()),
                float(data.std()),
            )
        )


def save_frames(frames, save_dir):
    """把抓到的每一帧存成 .npy（可选）。"""
    if not save_dir or not frames:
        return
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    for index, frame in enumerate(frames):
        path = os.path.join(save_dir, "ikap_frame_%02d.npy" % index)
        np.save(path, np.asarray(frame))
        print("Saved frame:", os.path.abspath(path))


def show_frames(frames, title="IKap frames", save_path=None, max_cols=5):
    """把抓到的多帧显示在同一个 matplotlib 窗口里（每帧一个子图）。

    用 0.5%/99.5% 分位数做对比度拉伸，避免 12bit 图像在小动态范围下看不见结构。
    """
    if not frames:
        print("No frame to display.")
        return False
    import matplotlib.pyplot as plt

    count = len(frames)
    cols = min(int(max_cols), count)
    rows = (count + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 3.0 * rows), squeeze=False)
    for index, frame in enumerate(frames):
        data = np.asarray(frame)
        axis = axes[index // cols][index % cols]
        if data.ndim == 3:
            data = data[:, :, 0]
        low = float(np.percentile(data, 0.5))
        high = float(np.percentile(data, 99.5))
        if high <= low:
            low, high = float(data.min()), max(float(data.max()), float(data.min()) + 1.0)
        axis.imshow(data, cmap="gray", aspect="auto", vmin=low, vmax=high)
        axis.set_title(
            "#%d %dx%d min=%d max=%d mean=%.1f"
            % (index, data.shape[1], data.shape[0], int(data.min()), int(data.max()), float(data.mean())),
            fontsize=8,
        )
        axis.set_xticks([])
        axis.set_yticks([])
    for index in range(count, rows * cols):
        axes[index // cols][index % cols].axis("off")
    fig.suptitle(title)
    # 注意：本机 matplotlib 在 fig.tight_layout() 与 savefig(PNG) 路径上会触发原生崩溃(0xC06D007E)，
    # 所以这里只用纯计算的 subplots_adjust，并且把 PNG 保存放到 plt.show() 之后、并用 try 包住。
    fig.subplots_adjust(left=0.02, right=0.99, top=0.93, bottom=0.02, wspace=0.06, hspace=0.14)
    print("Showing %d frames in a matplotlib window (close the window to finish)." % count)
    plt.show()
    if save_path:
        try:
            fig.savefig(save_path, dpi=120)
            print("Saved frame grid image:", os.path.abspath(save_path))
        except Exception as error:
            print("Warning: saving the frame grid image failed:", error)
    return True


def check_itk(res, label):
    if res != IKapCDef.ITKSTATUS_OK:
        low_status = int(res) & 0xFFFF
        suffix = ""
        if low_status == getattr(IKapCDef, "ITKSTATUS_SERAIL_PORT_NOT_AVAILABLE", -1):
            suffix = " (CameraLink serial port is unavailable; close IKapViewer/other camera software or reopen the Python console.)"
        elif low_status == getattr(IKapCDef, "ITKSTATUS_DEVICE_BUSY", -1):
            suffix = " (device is busy; another process may be using it.)"
        raise RuntimeError("%s failed: %#x%s" % (label, res, suffix))


def check_board(res, label):
    if res != IKapBoard.IK_RTN_OK:
        raise RuntimeError("%s failed: %#x" % (label, res))


def warn_board(res, label):
    if res != IKapBoard.IK_RTN_OK:
        print("Warning: %s failed: %#x" % (label, res))
        return False
    return True


def set_feature_enum(h_camera, name, value, required=False):
    res = IKapC.ItkDevFromString(h_camera, as_bytes(name), as_bytes(value))
    if res != IKapCDef.ITKSTATUS_OK:
        message = "Set camera enum %s=%s failed: %#x (%s)" % (name, value, res, describe_itk_status(res))
        if required:
            raise RuntimeError(message)
        print("Warning:", message)
        return False
    print("Camera enum %s=%s" % (name, value))
    return True


def set_feature_enum_any(h_camera, names, value, required=False):
    for name in names:
        if set_feature_enum(h_camera, name, value, required=False):
            return True
    message = "Set camera enum %s=%s failed." % ("/".join(names), value)
    if required:
        raise RuntimeError(message)
    return False


def set_feature_float(h_camera, names, value, required=False):
    last_res = None
    for name in names:
        res = IKapC.ItkDevSetDouble(h_camera, as_bytes(name), float(value))
        last_res = res
        if res == IKapCDef.ITKSTATUS_OK:
            print("Camera float %s=%g" % (name, value))
            return True
    message = "Set camera float %s=%g failed: %#x (%s)" % (
        "/".join(names),
        value,
        last_res,
        describe_itk_status(last_res),
    )
    if required:
        raise RuntimeError(message)
    print("Warning:", message)
    return False


def read_feature_float(h_camera, names):
    for name in names:
        res, value = IKapC.ItkDevGetDouble(h_camera, as_bytes(name))
        if res == IKapCDef.ITKSTATUS_OK:
            print("Camera float readback %s=%g" % (name, value))
            return True, name, value
    print("Warning: read camera float %s failed." % "/".join(names))
    return False, None, None


def read_feature_int(h_camera, names):
    for name in names:
        res, value = IKapC.ItkDevGetInt64(h_camera, as_bytes(name))
        if res == IKapCDef.ITKSTATUS_OK:
            print("Camera int readback %s=%d" % (name, value))
            return True, name, value
    print("Warning: read camera int %s failed." % "/".join(names))
    return False, None, None


def read_feature_enum(h_camera, names):
    for name in names:
        res, value = IKapC.ItkDevToString(h_camera, as_bytes(name))
        if res == IKapCDef.ITKSTATUS_OK:
            print("Camera enum readback %s=%s" % (name, decode_bytes(value)))
            return True, name, value
    print("Warning: read camera enum %s failed." % "/".join(names))
    return False, None, None


def set_feature_int(h_camera, names, value, required=False):
    last_res = None
    for name in names:
        res = IKapC.ItkDevSetInt64(h_camera, as_bytes(name), int(value))
        last_res = res
        if res == IKapCDef.ITKSTATUS_OK:
            print("Camera int %s=%d" % (name, value))
            return True
    message = "Set camera int %s=%d failed: %#x (%s)" % (
        "/".join(names),
        value,
        last_res,
        describe_itk_status(last_res),
    )
    if required:
        raise RuntimeError(message)
    print("Warning:", message)
    return False


def camera_feature_type_name(feature_type):
    return ITK_FEATURE_TYPE_NAMES.get(feature_type, "type=%s" % feature_type)


def digital_line_mask(line_name):
    try:
        line_number = int(str(line_name).split("line")[-1])
    except (TypeError, ValueError) as error:
        raise ValueError("Unsupported ART-DAQ line name: %s" % line_name) from error
    if line_number < 0 or line_number > 31:
        raise ValueError("Unsupported ART-DAQ line name: %s" % line_name)
    return np.uint32(1 << line_number)


def build_trigger_train(line_name, trigger_rate_hz, duration_s):
    sample_rate_hz = float(trigger_rate_hz) * 2.0
    sample_count = max(2, int(round(sample_rate_hz * float(duration_s))))
    if sample_count % 2:
        sample_count += 1
    waveform = np.zeros(sample_count, dtype=np.uint32)
    waveform[0::2] = digital_line_mask(line_name)
    return waveform, sample_rate_hz


def build_trigger_train_count(line_name, trigger_rate_hz, trigger_count):
    sample_rate_hz = float(trigger_rate_hz) * 2.0
    trigger_count = max(1, int(trigger_count))
    waveform = np.zeros(trigger_count * 2, dtype=np.uint32)
    waveform[0::2] = digital_line_mask(line_name)
    duration_s = float(trigger_count) / float(trigger_rate_hz)
    return waveform, sample_rate_hz, duration_s


def output_artdaq_line_triggers(device, line, trigger_rate_hz, duration_s):
    physical_line = "%s/%s" % (device, line)
    waveform, sample_rate_hz = build_trigger_train(line, trigger_rate_hz, duration_s)
    print(
        "ART-DAQ trigger output: line=%s, trigger_rate=%g Hz, sample_rate=%g Hz, "
        "samples=%d, duration=%.3f s"
        % (physical_line, trigger_rate_hz, sample_rate_hz, waveform.size, duration_s)
    )
    with ni.Task("ikap_line_trigger_do") as task:
        task.do_channels.add_do_chan(lines=physical_line)
        task.timing.cfg_samp_clk_timing(
            rate=sample_rate_hz,
            active_edge=Edge.RISING,
            sample_mode=AcquisitionType.FINITE,
            samps_per_chan=waveform.size,
        )
        writer = DigitalSingleChannelWriter(task.out_stream, auto_start=False)
        writer.verify_array_shape = False
        written = writer.write_many_sample_port_u_32(np.ascontiguousarray(waveform))
        print("ART-DAQ trigger samples written:", written)
        task.start()
        task.wait_until_done(timeout=float(duration_s) + 5.0)
        task.stop()


def output_artdaq_line_triggers_count(device, line, trigger_rate_hz, trigger_count, start_callback=None):
    wall_start = time.perf_counter()
    physical_line = "%s/%s" % (device, line)
    waveform, sample_rate_hz, duration_s = build_trigger_train_count(
        line,
        trigger_rate_hz,
        trigger_count,
    )
    print(
        "ART-DAQ trigger output: line=%s, trigger_rate=%g Hz, trigger_count=%d, "
        "sample_rate=%g Hz, samples=%d, duration=%.6f s"
        % (
            physical_line,
            trigger_rate_hz,
            int(trigger_count),
            sample_rate_hz,
            waveform.size,
            duration_s,
        )
    )
    with ni.Task("ikap_line_trigger_do") as task:
        task.do_channels.add_do_chan(lines=physical_line)
        task.timing.cfg_samp_clk_timing(
            rate=sample_rate_hz,
            active_edge=Edge.RISING,
            sample_mode=AcquisitionType.FINITE,
            samps_per_chan=waveform.size,
        )
        writer = DigitalSingleChannelWriter(task.out_stream, auto_start=False)
        writer.verify_array_shape = False
        written = writer.write_many_sample_port_u_32(np.ascontiguousarray(waveform))
        print("ART-DAQ trigger samples written:", written)
        task.start()
        start_wait = time.perf_counter()
        if start_callback is not None:
            start_callback(start_wait)
        task.wait_until_done(timeout=float(duration_s) + 5.0)
        wait_elapsed = time.perf_counter() - start_wait
        task.stop()
    total_elapsed = time.perf_counter() - wall_start
    print(
        "ART-DAQ trigger timing: expected_output=%.6f s, wait_elapsed=%.6f s, total_elapsed=%.6f s"
        % (duration_s, wait_elapsed, total_elapsed)
    )


INTEGRATION_SOURCE_VALUES = {
    "Internal": IKapBoard.IKP_INTEGRATION_TRIGGER_SOURCE_VAL_INTERNAL,
    "GeneralInput1": IKapBoard.IKP_INTEGRATION_TRIGGER_SOURCE_VAL_GENERAL_INPUT1,
    "General Input1": IKapBoard.IKP_INTEGRATION_TRIGGER_SOURCE_VAL_GENERAL_INPUT1,
    "GeneralInput2": IKapBoard.IKP_INTEGRATION_TRIGGER_SOURCE_VAL_GENERAL_INPUT2,
    "General Input2": IKapBoard.IKP_INTEGRATION_TRIGGER_SOURCE_VAL_GENERAL_INPUT2,
    "ShaftEncoder1": IKapBoard.IKP_INTEGRATION_TRIGGER_SOURCE_VAL_SHAFT_ENCODER1,
    "Shaft Encoder1": IKapBoard.IKP_INTEGRATION_TRIGGER_SOURCE_VAL_SHAFT_ENCODER1,
    "Software": IKapBoard.IKP_INTEGRATION_TRIGGER_SOURCE_VAL_SOFTWARE,
}

GENERAL_INPUT_SAMPLE_MODE_VALUES = {
    "ActiveHigh": IKapBoard.IKP_GENERAL_INPUT_SAMPLE_MODE_VAL_ACTIVE_HIGH,
    "ActiveLow": IKapBoard.IKP_GENERAL_INPUT_SAMPLE_MODE_VAL_ACTIVE_LOW,
    "RisingEdge": IKapBoard.IKP_GENERAL_INPUT_SAMPLE_MODE_VAL_RISING_EDGE,
    "FallingEdge": IKapBoard.IKP_GENERAL_INPUT_SAMPLE_MODE_VAL_FALLING_EDGE,
}

ENCODER_MULTIPLY_VALUES = {
    "1": IKapBoard.IKP_SHAFT_ENCODER_MULTIPLY_FACTOR_VAL_1,
    "2": IKapBoard.IKP_SHAFT_ENCODER_MULTIPLY_FACTOR_VAL_2,
    "4": IKapBoard.IKP_SHAFT_ENCODER_MULTIPLY_FACTOR_VAL_4,
    "8": IKapBoard.IKP_SHAFT_ENCODER_MULTIPLY_FACTOR_VAL_8,
    "16": IKapBoard.IKP_SHAFT_ENCODER_MULTIPLY_FACTOR_VAL_16,
    "32": IKapBoard.IKP_SHAFT_ENCODER_MULTIPLY_FACTOR_VAL_32,
}


IKAP_PIXEL_FORMATS = {
    8: ("Mono8", IKapBoard.IKP_DATA_FORMAT_VAL_8Bit),
    10: ("Mono10", IKapBoard.IKP_DATA_FORMAT_VAL_10Bit),
    12: ("Mono12", IKapBoard.IKP_DATA_FORMAT_VAL_12Bit),
    14: ("Mono14", IKapBoard.IKP_DATA_FORMAT_VAL_14Bit),
    16: ("Mono16", IKapBoard.IKP_DATA_FORMAT_VAL_16Bit),
}

# .vlcf 的 TAP_NUMBER 需要与相机 SensorTaps 一致；这里给出相机能接受的枚举写法。
IKAP_SENSOR_TAP_VALUES = {
    2: ("TwoTap", "Two Tap", "2Tap", "2"),
    4: ("FourTap", "Four Tap", "4Tap", "4"),
    8: ("EightTap", "Eight Tap", "8Tap", "8"),
    10: ("TenTap", "Ten Tap", "10Tap", "10"),
}
IKAP_SENSOR_TAP_FEATURE_NAMES = ("SensorTaps", "Sensor Taps", "SensorTap", "Sensor Tap")


def ikap_bit_depth(pixel_format):
    text = str(pixel_format).replace(" ", "").replace("-", "").lower()
    for bit_depth in (16, 14, 12, 10, 8):
        if str(bit_depth) in text:
            return bit_depth
    return 8


def ikap_pixel_format_settings(pixel_format):
    bit_depth = ikap_bit_depth(pixel_format)
    return IKAP_PIXEL_FORMATS.get(bit_depth, IKAP_PIXEL_FORMATS[8])


def set_board_int(h_board, name, param, value, required=False):
    res = IKapBoard.IKapSetInfo(h_board, param, int(value))
    if res != IKapBoard.IK_RTN_OK:
        message = "Set IKap board %s=%d failed: %#x" % (name, int(value), res)
        if required:
            raise RuntimeError(message)
        print("Warning:", message)
        return False
    print("IKap board %s=%d" % (name, int(value)))
    return True


def read_board_int(h_board, name, param):
    res, value = IKapBoard.IKapGetInfo(h_board, param)
    if res != IKapBoard.IK_RTN_OK:
        print("Warning: read IKap board %s failed: %#x" % (name, res))
        return False, None
    print("IKap board readback %s=%s" % (name, value))
    return True, value


class IKapLineScanGrabber:
    def __init__(self):
        self.h_board = None
        self.h_camera = ctypes.c_void_p(None)
        self.current_frame_index = ctypes.c_uint(0)
        self.buffer_data = None
        self.frame_ready_count = 0
        self.frame_lost_count = 0
        self.timeout_count = 0
        self.last_frame = None
        self.frame_ready_times = []
        self.width = 0
        self.height = 0
        self.depth = 16
        self.channel_count = 1
        self._callbacks = []
        self.grab_started = False
        # 抓帧缓存：用于"抓 N 张并画图确认"的流程
        self.capture_limit = 0
        self.captured_frames = []
        self.capture_start_time = None
        self.capture_elapsed = None

    def is_board_open(self):
        return self.h_board is not None and self.h_board != -1

    def is_camera_open(self):
        return bool(getattr(self.h_camera, "value", 0))

    def open_first_cameralink(self):
        res, camera_count = IKapC.ItkManGetDeviceCount()
        check_itk(res, "ItkManGetDeviceCount")
        if camera_count == 0:
            raise RuntimeError("No IKap camera found.")

        for index in range(camera_count):
            res, info = IKapC.ItkManGetDeviceInfo(index)
            check_itk(res, "ItkManGetDeviceInfo")
            print(
                "camera[%d]: name=%s, serial=%s, class=%s"
                % (
                    index,
                    decode_bytes(info.FullName),
                    decode_bytes(info.SerialNumber),
                    decode_bytes(info.DeviceClass),
                )
            )
            if info.DeviceClass == b"CameraLink" and info.SerialNumber != b"":
                res, cl_info = IKapC.ItkManGetCLDeviceInfo(index)
                check_itk(res, "ItkManGetCLDeviceInfo")
                res, self.h_camera = IKapC.ItkDevOpen(
                    index,
                    IKapCDef.ITKDEV_VAL_ACCESS_MODE_EXCLUSIVE,
                )
                if res != IKapCDef.ITKSTATUS_OK or self.h_camera.value == 0:
                    low_status = int(res) & 0xFFFF
                    if low_status == getattr(IKapCDef, "ITKSTATUS_SERAIL_PORT_NOT_AVAILABLE", -1):
                        print(
                            "Warning: ItkDevOpen failed because the CameraLink serial port is unavailable "
                            "(%#x). Continuing with board-only mode; camera exposure/trigger features "
                            "will be skipped. Close IKapViewer/other camera software if you need to set "
                            "camera parameters." % res
                        )
                        self.h_camera = ctypes.c_void_p(None)
                    else:
                        check_itk(res, "ItkDevOpen")
                        raise RuntimeError("ItkDevOpen returned null camera handle.")

                self.h_board = IKapBoard.IKapOpen(
                    cl_info.HostInterface,
                    cl_info.BoardIndex,
                )
                if self.h_board is None or self.h_board == -1:
                    if self.is_camera_open():
                        IKapC.ItkDevClose(self.h_camera)
                    self.h_camera = ctypes.c_void_p(None)
                    raise RuntimeError("IKapOpen board failed.")
                print(
                    "Using CameraLink camera index=%d, board_index=%d, camera_control=%s."
                    % (index, cl_info.BoardIndex, self.is_camera_open())
                )
                return
        raise RuntimeError("No CameraLink camera with serial number found.")

    def close(self):
        if self.is_board_open():
            try:
                IKapBoard.IKapClose(self.h_board)
            except Exception:
                pass
            self.h_board = None
        if self.is_camera_open():
            try:
                IKapC.ItkDevClose(self.h_camera)
            except Exception:
                pass
            self.h_camera = ctypes.c_void_p(None)

    def load_configuration(self, vlcf_path):
        if not self.is_board_open():
            raise RuntimeError("IKap board is not open.")
        if not vlcf_path:
            print("No .vlcf file provided; using current/default board configuration.")
            return
        vlcf_path = os.path.abspath(vlcf_path)
        if not os.path.isfile(vlcf_path):
            raise RuntimeError("Configuration file not found: %s" % vlcf_path)
        res = IKapBoard.IKapLoadConfigurationFromFile(self.h_board, vlcf_path.encode("utf-8"))
        check_board(res, "IKapLoadConfigurationFromFile")
        print("Loaded IKap board configuration:", vlcf_path)

    def print_vlcf_settings(self, vlcf_path):
        """打印 .vlcf 的关键设置，并用板卡读回值做一致性校验。"""
        settings = read_vlcf_settings(vlcf_path)
        if not settings:
            print("Warning: no .vlcf settings read from %s" % vlcf_path)
            return settings
        print("IKap .vlcf settings (%s):" % os.path.abspath(vlcf_path))
        for key in VLCF_KEYS_OF_INTEREST:
            if key in settings:
                print("  %-30s = %s" % (key, settings[key]))

        depth = vlcf_int(settings, "PIXEL_DEPTH")
        pfnc = vlcf_int(settings, "PIXEL_FORMAT_PFNC")
        if depth is not None and pfnc is not None:
            pfnc_depth = PFNC_MONO_BIT_DEPTH.get(pfnc)
            if pfnc_depth is not None and pfnc_depth != depth:
                print(
                    "Warning: .vlcf PIXEL_DEPTH=%s but Pixel_Format_PFNC=%s (%d bit)."
                    % (depth, pfnc, pfnc_depth)
                )
        if not self.is_board_open():
            print("IKap board is not open; .vlcf readback skipped.")
            return settings

        print("IKap board readback after loading .vlcf:")
        for name, param in (
            ("IKP_DATA_FORMAT", IKapBoard.IKP_DATA_FORMAT),
            ("IKP_TAP_NUMBER", IKapBoard.IKP_TAP_NUMBER),
            ("IKP_IMAGE_WIDTH", IKapBoard.IKP_IMAGE_WIDTH),
            ("IKP_IMAGE_HEIGHT", IKapBoard.IKP_IMAGE_HEIGHT),
            ("IKP_IMAGE_OFFSET_X", IKapBoard.IKP_IMAGE_OFFSET_X),
            ("IKP_FRAME_SIZE", IKapBoard.IKP_FRAME_SIZE),
            ("IKP_CC1_SOURCE", IKapBoard.IKP_CC1_SOURCE),
            ("IKP_INTEGRATION_TRIGGER_SOURCE", IKapBoard.IKP_INTEGRATION_TRIGGER_SOURCE),
            ("IKP_INTEGRATION_METHOD", IKapBoard.IKP_INTEGRATION_METHOD),
            ("IKP_SHAFT_ENCODER1_CHANNEL", IKapBoard.IKP_SHAFT_ENCODER1_CHANNEL),
            ("IKP_SHAFT_ENCODER1_MULTIPlY_FACTOR", IKapBoard.IKP_SHAFT_ENCODER1_MULTIPlY_FACTOR),
        ):
            read_board_int(self.h_board, name, param)
        return settings

    def check_vlcf_alignment(self, settings, width, height, offset_x, pixel_format, taps=None):
        """告警：命令行传入的几何/位深/tap 与 .vlcf 不一致（板卡仍按 .vlcf 工作）。"""
        for key, label, value in (
            ("IMAGE_WIDTH", "width", width),
            ("IMAGE_HEIGHT", "height", height),
            ("IMAGE_OFFSET_X", "offset_x", offset_x),
            ("TAP_NUMBER", "taps", taps),
        ):
            expected = vlcf_int(settings, key)
            if expected is None or value is None:
                continue
            if int(value) != expected:
                print(
                    "Warning: %s=%s differs from .vlcf %s=%s; the board keeps the .vlcf value, "
                    "so the camera setting may not match." % (label, value, key, expected)
                )
        depth = vlcf_int(settings, "PIXEL_DEPTH")
        if depth is not None and ikap_bit_depth(pixel_format) != depth:
            print(
                "Warning: pixel_format=%s is %d bit but .vlcf PIXEL_DEPTH=%s; "
                "camera bit depth and board bit depth must match."
                % (pixel_format, ikap_bit_depth(pixel_format), depth)
            )

    def configure_board_data_format(self, pixel_format):
        """写入板卡位深 IKP_DATA_FORMAT（仅在需要代码覆盖 .vlcf 时调用）。

        12bit4tap.vlcf 的 PIXEL_DEPTH=12 已经定义了板卡位深，所以 configure_camera()
        不再写它；需要显式覆盖的场景（IKapCamera_ui.py）调用本方法。
        """
        if not self.is_board_open():
            print("IKap board is not open; skipped IKP_DATA_FORMAT.")
            return False
        _camera_pixel_format, board_data_format = ikap_pixel_format_settings(pixel_format)
        return set_board_int(
            self.h_board,
            "IKP_DATA_FORMAT",
            IKapBoard.IKP_DATA_FORMAT,
            board_data_format,
            required=False,
        )

    def set_camera_sensor_taps(self, taps):
        """按 .vlcf 的 TAP_NUMBER 设置相机 SensorTaps，保证与板卡 tap 数一致。"""
        if not self.is_camera_open():
            print("Camera control handle is not open; skipped SensorTaps=%s." % taps)
            return False
        tap_values = IKAP_SENSOR_TAP_VALUES.get(int(taps), (str(int(taps)),))
        for value in tap_values:
            if set_feature_enum_any(
                self.h_camera,
                IKAP_SENSOR_TAP_FEATURE_NAMES,
                value,
                required=False,
            ):
                read_feature_enum(self.h_camera, IKAP_SENSOR_TAP_FEATURE_NAMES)
                return True
        print("Warning: set camera SensorTaps=%d failed." % int(taps))
        return False

    def print_feature_range(self, name):
        """打印整数特征的 min/max/increment，用于判断设置失败是不是步进/范围问题。"""
        if not self.is_camera_open():
            return
        res_min, value_min = IKapC.ItkDevGetInt64Min(self.h_camera, as_bytes(name))
        res_max, value_max = IKapC.ItkDevGetInt64Max(self.h_camera, as_bytes(name))
        res_inc, value_inc = IKapC.ItkDevGetInt64Inc(self.h_camera, as_bytes(name))
        if res_min == IKapCDef.ITKSTATUS_OK and res_max == IKapCDef.ITKSTATUS_OK:
            print(
                "  hint: %s int range min=%d max=%d inc=%s "
                "(合法值需满足 min<=v<=max 且 (v-min)%%inc==0)"
                % (name, value_min, value_max, str(value_inc) if res_inc == IKapCDef.ITKSTATUS_OK else "?")
            )
        else:
            print("  hint: %s is not an integer feature or not readable." % name)

    def print_camera_feature_detail(self, name):
        """打印单个相机特征的可用信息：类型、当前值、访问权限、整数范围、枚举项。"""
        res, info = IKapC.ItkDevGetFeatureInfo(self.h_camera, as_bytes(name))
        feature_type = getattr(info, "Type", None) if res == IKapCDef.ITKSTATUS_OK else None
        res_access, access_mode = IKapC.ItkDevGetAccessMode(self.h_camera, as_bytes(name))
        res_value, value = IKapC.ItkDevToString(self.h_camera, as_bytes(name))
        value_text = (
            decode_bytes(value) if res_value == IKapCDef.ITKSTATUS_OK else ("<unreadable %#x>" % res_value)
        )
        print(
            "  %-22s type=%-8s value=%-16s access=%s"
            % (
                name,
                camera_feature_type_name(feature_type),
                value_text,
                ("%#x" % access_mode) if res_access == IKapCDef.ITKSTATUS_OK else "?",
            )
        )
        res_min, value_min = IKapC.ItkDevGetInt64Min(self.h_camera, as_bytes(name))
        res_max, value_max = IKapC.ItkDevGetInt64Max(self.h_camera, as_bytes(name))
        res_inc, value_inc = IKapC.ItkDevGetInt64Inc(self.h_camera, as_bytes(name))
        if res_min == IKapCDef.ITKSTATUS_OK and res_max == IKapCDef.ITKSTATUS_OK:
            print(
                "        int range: min=%d max=%d inc=%s"
                % (value_min, value_max, str(value_inc) if res_inc == IKapCDef.ITKSTATUS_OK else "?")
            )
        enum_count = getattr(info, "EnumCount", 0) if res == IKapCDef.ITKSTATUS_OK else 0
        if enum_count:
            entries = []
            for index in range(int(enum_count)):
                res_entry, entry = IKapC.ItkDevGetEnumEntryFeatureInfo(
                    self.h_camera, as_bytes(name), index
                )
                if res_entry == IKapCDef.ITKSTATUS_OK:
                    entries.append(decode_bytes(getattr(entry, "ValueStr", b"")))
            print("        enum(%s): %s" % (enum_count, ", ".join(entries)))

    def dump_camera_features(self, pattern=""):
        """列出相机特征（可按 pattern 过滤），并打印关键特征的取值/范围/枚举项。

        用来查清 Width / SensorTaps / Trigger* 等设置失败的原因：是特征名不对，还是取值不对。
        """
        if not self.is_camera_open():
            print("Camera control handle is not open; camera feature dump skipped.")
            return []
        res, feature_count = IKapC.ItkDevGetFeatureCount(self.h_camera)
        check_itk(res, "ItkDevGetFeatureCount")
        names = []
        for index in range(int(feature_count)):
            res_name, raw_name = IKapC.ItkDevGetFeatureName(self.h_camera, index)
            if res_name == IKapCDef.ITKSTATUS_OK:
                names.append(decode_bytes(raw_name))
        needle = str(pattern or "").lower()
        matched = [name for name in names if not needle or needle in name.lower()]
        print(
            "IKap camera features: total=%d, matched=%d%s"
            % (len(names), len(matched), (" (pattern='%s')" % pattern) if needle else "")
        )
        for name in matched:
            print("    " + name)
        detail_names = matched if needle else [name for name in CAMERA_FEATURE_PROBE_NAMES if name in names]
        if detail_names:
            print("IKap camera feature details:")
            for name in detail_names:
                self.print_camera_feature_detail(name)
        return names

    def configure_board_internal_trigger(self, line_rate_hz):
        """用板卡内部行触发（无需外部编码器/ART-DAQ），用于快速确认成像是否正常。

        与 .vlcf 的差别：把 INTEGRATION_TRIGGER_SOURCE 从 ShaftEncoder1(5) 改成 Internal(0)，
        行触发频率写入 IKP_INTEGRATION_TRIGGER_FREQUENCY（即 .vlcf 的 INTERNAL_TRIGGER_FREQUENCY）。
        CC1 仍输出积分信号1给相机（相机需处于 ExternalPulse/外触发模式）。
        """
        if not self.is_board_open():
            raise RuntimeError("IKap board is not open.")
        res = IKapBoard.IKapSetInfo(
            self.h_board,
            IKapBoard.IKP_CC1_SOURCE,
            IKapBoard.IKP_CC_SOURCE_VAL_INTEGRATION_SIGNAL1,
        )
        check_board(res, "IKP_CC1_SOURCE")
        res = IKapBoard.IKapSetInfo(
            self.h_board,
            IKapBoard.IKP_INTEGRATION_TRIGGER_SOURCE,
            IKapBoard.IKP_INTEGRATION_TRIGGER_SOURCE_VAL_INTERNAL,
        )
        check_board(res, "IKP_INTEGRATION_TRIGGER_SOURCE")
        res = IKapBoard.IKapSetInfo(
            self.h_board,
            IKapBoard.IKP_INTEGRATION_METHOD,
            IKapBoard.IKP_INTEGRATION_METHOD_VAL_1,
        )
        check_board(res, "IKP_INTEGRATION_METHOD")
        res = IKapBoard.IKapSetInfo(
            self.h_board,
            IKapBoard.IKP_INTEGRATION_TRIGGER_FREQUENCY,
            int(round(float(line_rate_hz))),
        )
        check_board(res, "IKP_INTEGRATION_TRIGGER_FREQUENCY")
        print(
            "Board internal line trigger configured: %.1f Hz (CC1 = IntegrationSignal1, "
            "no external encoder/ART-DAQ needed)." % float(line_rate_hz)
        )

    def configure_camera_line_trigger_for_external_pulse(self):
        """内部行触发模式下，让相机接受板卡 CC1 输出的行触发脉冲。

        这个相机上 TriggerSelector / TriggerMode / ExternalTriggerSource 等特征名可能并不存在
        （报 ITKSTATUS_INVALID_NAME），这里只设置确实存在的特征，并把失败信息打出来。
        """
        if not self.is_camera_open():
            print("Camera control handle is not open; camera line-trigger settings skipped (board-only mode).")
            return False
        ok = set_feature_enum_any(
            self.h_camera,
            ("SynchronizationMode", "Synchronization Mode"),
            "ExternalPulse",
            required=False,
        )
        if ok:
            read_feature_enum(self.h_camera, ("SynchronizationMode", "Synchronization Mode"))
        for name, value in (
            ("TriggerSelector", "LineStart"),
            ("TriggerMode", "On"),
            ("TriggerSource", "Line1"),
            ("TriggerActivation", "RisingEdge"),
        ):
            set_feature_enum(self.h_camera, name, value, required=False)
        if not ok:
            print(
                "Warning: camera SynchronizationMode could not be set to ExternalPulse; "
                "run with --dump-camera-features to list the real feature names."
            )
        return ok

    def configure_board_image(self, width=None, height=None, offset_x=0):
        if not self.is_board_open():
            raise RuntimeError("IKap board is not open.")
        if width is not None:
            set_board_int(self.h_board, "IKP_IMAGE_WIDTH", IKapBoard.IKP_IMAGE_WIDTH, width, required=False)
        if height is not None:
            set_board_int(self.h_board, "IKP_IMAGE_HEIGHT", IKapBoard.IKP_IMAGE_HEIGHT, height, required=False)
        set_board_int(self.h_board, "IKP_IMAGE_OFFSET_X", IKapBoard.IKP_IMAGE_OFFSET_X, offset_x, required=False)
        self.print_board_readback()

    def print_board_readback(self):
        if not self.is_board_open():
            print("IKap board is not open; board readback skipped.")
            return
        print("IKap board readback:")
        read_board_int(self.h_board, "IKP_IMAGE_WIDTH", IKapBoard.IKP_IMAGE_WIDTH)
        read_board_int(self.h_board, "IKP_IMAGE_HEIGHT", IKapBoard.IKP_IMAGE_HEIGHT)
        read_board_int(self.h_board, "IKP_IMAGE_OFFSET_X", IKapBoard.IKP_IMAGE_OFFSET_X)
        read_board_int(self.h_board, "IKP_FRAME_SIZE", IKapBoard.IKP_FRAME_SIZE)
        read_board_int(self.h_board, "IKP_IMAGE_TYPE", IKapBoard.IKP_IMAGE_TYPE)

    def print_camera_readback(self):
        if not self.is_camera_open():
            print("Camera control handle is not open; camera readback skipped.")
            return
        print("IKap camera readback:")
        read_feature_float(self.h_camera, ("ExposureTime", "ExposureTimeAbs", "ExposureTimeRaw"))
        read_feature_int(self.h_camera, ("Width",))
        read_feature_int(self.h_camera, ("Height",))
        read_feature_int(self.h_camera, ("OffsetX",))
        read_feature_int(self.h_camera, ("OffsetY",))
        read_feature_enum(self.h_camera, ("PixelFormat",))
        read_feature_float(self.h_camera, ("AcquisitionLineRate", "LineRate", "AcquisitionFrameRate"))
        read_feature_float(self.h_camera, ("DigitalGain", "Digital Gain", "Gain"))

    def configure_camera(
        self,
        width,
        height,
        exposure_us,
        line_rate_hz,
        pixel_format,
        offset_x=0,
        sensor_taps=None,
    ):
        """配置相机；板卡侧设置由 .vlcf 决定，这里只写相机侧特征。

        相机侧特征（Width/Height/Offset/PixelFormat/ExposureTime/SensorTaps）
        不在 .vlcf 里，必须在线写入；默认值取自 .vlcf，保证与板卡一致。
        """
        camera_pixel_format, board_data_format = ikap_pixel_format_settings(pixel_format)
        # ---------------------------------------------------------------------------
        # 以下板卡侧硬编码设置与 12bit4tap.vlcf 冲突，已注释掉，改由 .vlcf 决定：
        #   1) 几何尺寸：.vlcf 为 IMAGE_WIDTH=1440 / IMAGE_HEIGHT=400 / IMAGE_OFFSET_X=0，
        #      原来的 configure_board_image() 会把板卡强制成 4096x1，覆盖 .vlcf 的设置，
        #      造成板卡与相机尺寸不一致。
        #   2) 像素位深：.vlcf 为 PIXEL_DEPTH=12（Pixel_Format_PFNC=17825797 = Mono12），
        #      原来的 IKP_DATA_FORMAT 写入会把它覆盖成代码算出来的值（默认 8bit）。
        # 若确实需要用代码覆盖 .vlcf，请取消下面的注释。
        # if self.is_board_open():
        #     self.configure_board_image(width=width, height=height, offset_x=offset_x)
        #     set_board_int(
        #         self.h_board,
        #         "IKP_DATA_FORMAT",
        #         IKapBoard.IKP_DATA_FORMAT,
        #         board_data_format,
        #         required=False,
        #     )
        # ---------------------------------------------------------------------------
        print(
            "configure_camera(): board geometry and pixel depth are not written here; "
            "they come from the .vlcf (or from an explicit configure_board_image/"
            "configure_board_data_format call)."
        )
        if not self.is_camera_open():
            print("Camera control handle is not open; skipping camera feature settings.")
            return
        # 顺序很重要：PixelFormat / SensorTaps 决定 CameraLink 数据宽度，
        # 会改变 Width/Height/Offset 的合法步进(Increment)，所以必须先设位深与 tap 再设几何。
        if pixel_format:
            # 相机 PixelFormat 必须与 .vlcf 的 PIXEL_DEPTH 一致（12bit4tap.vlcf -> Mono12）。
            set_feature_enum(self.h_camera, "PixelFormat", camera_pixel_format, required=False)
            read_feature_enum(self.h_camera, ("PixelFormat",))
        if sensor_taps:
            # 相机 SensorTaps 必须与 .vlcf 的 TAP_NUMBER 一致（12bit4tap.vlcf -> 4 tap）。
            self.set_camera_sensor_taps(sensor_taps)
        set_feature_enum(self.h_camera, "ExposureAuto", "Off", required=False)
        set_feature_float(
            self.h_camera,
            ("ExposureTime", "ExposureTimeAbs", "ExposureTimeRaw"),
            exposure_us,
            required=False,
        )
        read_feature_float(self.h_camera, ("ExposureTime", "ExposureTimeAbs", "ExposureTimeRaw"))
        # 几何：必须与 .vlcf 的 IMAGE_WIDTH/IMAGE_HEIGHT/IMAGE_OFFSET_X 一致，否则板卡组帧会错位。
        if not set_feature_int(self.h_camera, ("Width",), width, required=False):
            self.print_feature_range("Width")
        read_feature_int(self.h_camera, ("Width",))
        if not set_feature_int(self.h_camera, ("Height",), height, required=False):
            self.print_feature_range("Height")
        read_feature_int(self.h_camera, ("Height",))
        # 相机侧 OffsetX 必须与 .vlcf 的 IMAGE_OFFSET_X 相同，否则板卡取到的数据会错位。
        if not set_feature_int(self.h_camera, ("OffsetX",), offset_x, required=False):
            self.print_feature_range("OffsetX")
        set_feature_int(self.h_camera, ("OffsetY",), 0, required=False)
        set_feature_float(
            self.h_camera,
            ("AcquisitionLineRate", "LineRate", "AcquisitionFrameRate"),
            line_rate_hz,
            required=False,
        )
        read_feature_float(
            self.h_camera,
            ("AcquisitionLineRate", "LineRate", "AcquisitionFrameRate"),
        )
        self.print_camera_readback()

    def configure_board_line_trigger(
        self,
        integration_source="ShaftEncoder1",
        general_input_edge="RisingEdge",
        encoder_multiply="1",
    ):
        """板卡侧行触发设置（仅在需要代码覆盖 .vlcf 时调用）。

        12bit4tap.vlcf 已包含 CC1_SOURCE / INTEGRATION_TRIGGER_SOURCE / INTEGRATION_METHOD /
        SHAFT_ENCODER_CHANNEL / SHAFT_ENCODER_MULTIPLY_FACTOR / GENERAL_INPUT1_SAMPLE_MODE，
        因此 configure_line_trigger() 默认不再写这些板卡参数（板卡设置以 .vlcf 为准）。
        本方法保留给需要显式覆盖的场景（IKapCamera_ui.py 仍然调用它）。
        """
        if not self.is_board_open():
            raise RuntimeError("IKap board is not open.")
        integration_value = INTEGRATION_SOURCE_VALUES.get(
            integration_source,
            INTEGRATION_SOURCE_VALUES["ShaftEncoder1"],
        )
        res = IKapBoard.IKapSetInfo(
            self.h_board,
            IKapBoard.IKP_CC1_SOURCE,
            IKapBoard.IKP_CC_SOURCE_VAL_INTEGRATION_SIGNAL1,
        )
        check_board(res, "IKP_CC1_SOURCE")
        res = IKapBoard.IKapSetInfo(
            self.h_board,
            IKapBoard.IKP_INTEGRATION_TRIGGER_SOURCE,
            integration_value,
        )
        check_board(res, "IKP_INTEGRATION_TRIGGER_SOURCE")
        res = IKapBoard.IKapSetInfo(
            self.h_board,
            IKapBoard.IKP_INTEGRATION_METHOD,
            IKapBoard.IKP_INTEGRATION_METHOD_VAL_1,
        )
        check_board(res, "IKP_INTEGRATION_METHOD")

        if integration_value == IKapBoard.IKP_INTEGRATION_TRIGGER_SOURCE_VAL_GENERAL_INPUT1:
            sample_mode = GENERAL_INPUT_SAMPLE_MODE_VALUES.get(
                general_input_edge,
                IKapBoard.IKP_GENERAL_INPUT_SAMPLE_MODE_VAL_RISING_EDGE,
            )
            res = IKapBoard.IKapSetInfo(
                self.h_board,
                IKapBoard.IKP_GENERAL_INPUT1_SAMPLE_MODE,
                sample_mode,
            )
            check_board(res, "IKP_GENERAL_INPUT1_SAMPLE_MODE")
            res = IKapBoard.IKapSetInfo(
                self.h_board,
                IKapBoard.IKP_GENERAL_INPUT1_TRIGGER_MODE,
                IKapBoard.IKP_GENERAL_INPUT_TRIGGER_MODE_VAL_EDGE,
            )
            check_board(res, "IKP_GENERAL_INPUT1_TRIGGER_MODE")

        if integration_value == IKapBoard.IKP_INTEGRATION_TRIGGER_SOURCE_VAL_SHAFT_ENCODER1:
            multiply_value = ENCODER_MULTIPLY_VALUES.get(str(encoder_multiply), ENCODER_MULTIPLY_VALUES["1"])
            res = IKapBoard.IKapSetInfo(
                self.h_board,
                IKapBoard.IKP_SHAFT_ENCODER1_CHANNEL,
                IKapBoard.IKP_SHAFT_ENCODER_CHANNEL_VAL_A,
            )
            check_board(res, "IKP_SHAFT_ENCODER1_CHANNEL")
            res = IKapBoard.IKapSetInfo(
                self.h_board,
                IKapBoard.IKP_SHAFT_ENCODER1_MULTIPlY_FACTOR,
                multiply_value,
            )
            check_board(res, "IKP_SHAFT_ENCODER1_MULTIPlY_FACTOR")
        print(
            "Board line trigger set by code (overrides .vlcf): integration_source=%s, "
            "general_input_edge=%s, encoder_multiply=%s"
            % (integration_source, general_input_edge, encoder_multiply)
        )

    def configure_line_trigger(
        self,
        integration_source="ShaftEncoder1",
        camera_working_mode="Free Run",
        camera_trigger_source="General Input1",
        general_input_edge="RisingEdge",
        encoder_multiply="1",
    ):
        """相机侧行触发设置；板卡侧设置全部交给 .vlcf 文件。

        与 12bit4tap.vlcf 重复/冲突的硬编码板卡写入已注释掉（原实现见
        configure_board_line_trigger()，需要代码覆盖 .vlcf 时改调它）：
          * IKP_CC1_SOURCE                 <-> .vlcf CC1_SOURCE=1
          * IKP_INTEGRATION_TRIGGER_SOURCE <-> .vlcf INTEGRATION_TRIGGER_SOURCE=5
          * IKP_INTEGRATION_METHOD         <-> .vlcf INTEGRATION_METHOD=0
          * IKP_GENERAL_INPUT1_SAMPLE_MODE <-> .vlcf GENERAL_INPUT1_SAMPLE_MODE=0
            （代码用 RisingEdge=2，文件用 ActiveHigh=0，属真冲突）
          * IKP_GENERAL_INPUT1_TRIGGER_MODE <-> .vlcf GENERALINPUT1_TRIGGER_MODE=0
          * IKP_SHAFT_ENCODER1_CHANNEL      <-> .vlcf SHAFT_ENCODER_CHANNEL=0
          * IKP_SHAFT_ENCODER1_MULTIPlY_FACTOR <-> .vlcf SHAFT_ENCODER_MULTIPLY_FACTOR=0
        """
        if not self.is_board_open():
            raise RuntimeError("IKap board is not open.")
        # ---- 板卡侧设置改由 .vlcf 决定，以下硬编码写入已注释掉 ----
        # integration_value = INTEGRATION_SOURCE_VALUES.get(
        #     integration_source,
        #     INTEGRATION_SOURCE_VALUES["ShaftEncoder1"],
        # )
        # res = IKapBoard.IKapSetInfo(self.h_board, IKapBoard.IKP_CC1_SOURCE,
        #                             IKapBoard.IKP_CC_SOURCE_VAL_INTEGRATION_SIGNAL1)
        # check_board(res, "IKP_CC1_SOURCE")
        # res = IKapBoard.IKapSetInfo(self.h_board, IKapBoard.IKP_INTEGRATION_TRIGGER_SOURCE,
        #                             integration_value)
        # check_board(res, "IKP_INTEGRATION_TRIGGER_SOURCE")
        # res = IKapBoard.IKapSetInfo(self.h_board, IKapBoard.IKP_INTEGRATION_METHOD,
        #                             IKapBoard.IKP_INTEGRATION_METHOD_VAL_1)
        # check_board(res, "IKP_INTEGRATION_METHOD")
        # 通用输入1 采样模式/触发模式、编码器通道/倍频分支同样已注释；
        # 需要代码覆盖 .vlcf 时取消下面一行注释（或直接调用 configure_board_line_trigger）。
        # self.configure_board_line_trigger(
        #     integration_source=integration_source,
        #     general_input_edge=general_input_edge,
        #     encoder_multiply=encoder_multiply,
        # )
        print("Board-side line trigger source comes from the .vlcf file (hardcoded writes commented out).")
        if self.is_camera_open():
            set_feature_enum_any(
                self.h_camera,
                ("SynchronizationMode", "Synchronization Mode"),
                "ExternalPulse",
                required=False,
            )
            set_feature_enum_any(
                self.h_camera,
                ("ExternalTriggerSource", "External Trigger Source"),
                "CC1",
                required=False,
            )
            set_feature_enum_any(
                self.h_camera,
                ("WorkingMode", "Working Mode"),
                camera_working_mode,
                required=False,
            )
            if camera_working_mode == "Hardware Trigger":
                set_feature_enum_any(
                    self.h_camera,
                    ("TriggerSource", "Trigger Source"),
                    camera_trigger_source,
                    required=False,
                )
            set_feature_enum(self.h_camera, "TriggerSelector", "FrameStart", required=False)
            set_feature_enum(self.h_camera, "TriggerMode", "Off", required=False)
            set_feature_enum(self.h_camera, "TriggerSelector", "LineStart", required=False)
            set_feature_enum(self.h_camera, "TriggerMode", "On", required=False)
            set_feature_enum(self.h_camera, "TriggerSource", "Line1", required=False)
            set_feature_enum(self.h_camera, "TriggerActivation", "RisingEdge", required=False)
        else:
            print("Camera control handle is not open; only board-side line trigger was configured.")
        print(
            "Line trigger configured: integration_source=%s, working_mode=%s, "
            "camera_trigger_source=%s, general_input_edge=%s, encoder_multiply=%s"
            % (
                integration_source,
                camera_working_mode,
                camera_trigger_source,
                general_input_edge,
                encoder_multiply,
            )
        )

    def start_grab(self, frame_count, timeout_ms=-1):
        if not self.is_board_open():
            raise RuntimeError("IKap board is not open.")
        print("IKap start_grab readback before buffer allocation:")
        self.print_board_readback()
        self.current_frame_index = ctypes.c_uint(0)
        self.frame_ready_count = 0
        self.frame_lost_count = 0
        self.timeout_count = 0
        self.last_frame = None
        self.frame_ready_times = []
        res = IKapBoard.IKapSetInfo(self.h_board, IKapBoard.IKP_FRAME_COUNT, int(frame_count))
        check_board(res, "IKP_FRAME_COUNT")
        # 采集期参数（不属于 .vlcf 的静态配置，必须在 IKapStartGrab 前设定）：
        #   IKP_TIME_OUT = -1 与 .vlcf 的 TIMEOUT=4294967295 等价（无限等待）。
        #   IKP_GRAB_MODE 这里故意用 NON_BLOCK(1)：.vlcf 的 GRAB_MODE=0(BLOCK) 会让
        #   IKapStartGrab 阻塞到所有帧传输完毕，脚本就无法并行输出 ART-DAQ 行触发了。
        #   若确实要遵循 .vlcf 的阻塞模式，把下面那行改成 IKapBoard.IKP_GRAB_BLOCK。
        res = IKapBoard.IKapSetInfo(self.h_board, IKapBoard.IKP_TIME_OUT, int(timeout_ms))
        check_board(res, "IKP_TIME_OUT")
        res = IKapBoard.IKapSetInfo(self.h_board, IKapBoard.IKP_GRAB_MODE, IKapBoard.IKP_GRAB_NON_BLOCK)
        check_board(res, "IKP_GRAB_MODE")
        res = IKapBoard.IKapSetInfo(
            self.h_board,
            IKapBoard.IKP_FRAME_TRANSFER_MODE,
            IKapBoard.IKP_FRAME_TRANSFER_SYNCHRONOUS_NEXT_EMPTY_WITH_PROTECT,
        )
        check_board(res, "IKP_FRAME_TRANSFER_MODE")

        res, frame_size = IKapBoard.IKapGetInfo(self.h_board, IKapBoard.IKP_FRAME_SIZE)
        check_board(res, "IKP_FRAME_SIZE")
        self.buffer_data = ctypes.create_string_buffer(frame_size)

        res, self.width = IKapBoard.IKapGetInfo(self.h_board, IKapBoard.IKP_IMAGE_WIDTH)
        check_board(res, "IKP_IMAGE_WIDTH")
        res, self.height = IKapBoard.IKapGetInfo(self.h_board, IKapBoard.IKP_IMAGE_HEIGHT)
        check_board(res, "IKP_IMAGE_HEIGHT")
        res, data_format = IKapBoard.IKapGetInfo(self.h_board, IKapBoard.IKP_DATA_FORMAT)
        check_board(res, "IKP_DATA_FORMAT")
        res, image_type = IKapBoard.IKapGetInfo(self.h_board, IKapBoard.IKP_IMAGE_TYPE)
        check_board(res, "IKP_IMAGE_TYPE")
        self.depth = 8 if data_format == 8 else 16
        if image_type == IKapBoard.IKP_IMAGE_TYPE_VAL_MONOCHROME:
            self.channel_count = 1
        elif image_type in (IKapBoard.IKP_IMAGE_TYPE_VAL_RGB, IKapBoard.IKP_IMAGE_TYPE_VAL_BGR):
            self.channel_count = 3
        else:
            self.channel_count = 4
        print(
            "IKap image: width=%d, height=%d, depth=%d, channels=%d, frame_size=%d"
            % (self.width, self.height, self.depth, self.channel_count, frame_size)
        )

        self._register_callbacks()
        res = IKapBoard.IKapStartGrab(self.h_board, 0)
        check_board(res, "IKapStartGrab")
        self.grab_started = True
        print("IKap grab started.")

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
        print("IKap grab stopped.")

    def _register_callbacks(self):
        callback_specs = [
            (IKapBoard.IKEvent_GrabStart, self.on_grab_start),
            (IKapBoard.IKEvent_GrabStop, self.on_grab_stop),
            (IKapBoard.IKEvent_FrameReady, self.on_frame_ready),
            (IKapBoard.IKEvent_TimeOut, self.on_timeout),
            (IKapBoard.IKEvent_FrameLost, self.on_frame_lost),
        ]
        self._callbacks = []
        for event_type, func in callback_specs:
            callback = ctypes.CFUNCTYPE(None, ctypes.c_void_p)(func)
            self._callbacks.append(callback)
            IKapBoard.IKapRegisterCallback(
                self.h_board,
                event_type,
                callback,
                ctypes.c_void_p(None),
            )

    def on_grab_start(self, _):
        print("IKap event: grab start")

    def on_grab_stop(self, _):
        print("IKap event: grab stop")

    def on_timeout(self, _):
        self.timeout_count += 1
        print("IKap event: timeout")

    def on_frame_lost(self, _):
        self.frame_lost_count += 1
        print("IKap event: frame lost")

    def on_frame_ready(self, _):
        frame_index = self.current_frame_index.value
        res, status = IKapBoard.IKapGetBufferStatus(self.h_board, self.current_frame_index)
        if res != IKapBoard.IK_RTN_OK:
            print("IKapGetBufferStatus failed: %#x" % res)
            return
        if status.uFull == 1:
            ready_time = time.perf_counter()
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
                        self.last_frame = array[:expected].reshape(self.height, self.width)
                    else:
                        self.last_frame = array[:expected].reshape(
                            self.height,
                            self.width,
                            self.channel_count,
                        )
                else:
                    self.last_frame = array
                if len(self.captured_frames) < self.capture_limit and self.last_frame is not None:
                    if self.capture_start_time is None:
                        self.capture_start_time = ready_time
                    self.captured_frames.append(self.last_frame)
                    if len(self.captured_frames) >= self.capture_limit:
                        self.capture_elapsed = time.perf_counter() - self.capture_start_time
                        print(
                            "IKap capture complete: %d frames in %.3f s (%.1f fps)"
                            % (
                                len(self.captured_frames),
                                self.capture_elapsed,
                                len(self.captured_frames) / max(self.capture_elapsed, 1e-6),
                            )
                        )
            self.frame_ready_count += 1
            self.frame_ready_times.append(ready_time)
            print("IKap event: frame ready #%d buffer=%d" % (self.frame_ready_count, frame_index))
        res, frame_count = IKapBoard.IKapGetInfo(self.h_board, IKapBoard.IKP_FRAME_COUNT)
        if res == IKapBoard.IK_RTN_OK and frame_count > 0:
            self.current_frame_index.value = (frame_index + 1) % int(frame_count)


def parse_args():
    # 默认值全部来自 .vlcf（AKCameraConfig\12bit4tap.vlcf），避免代码里的硬编码与文件冲突。
    vlcf_settings = read_vlcf_settings(DEFAULT_VLCF_PATH)
    default_depth = vlcf_int(vlcf_settings, "PIXEL_DEPTH", 12)
    default_width = vlcf_int(vlcf_settings, "IMAGE_WIDTH", 1440)
    default_height = vlcf_int(vlcf_settings, "IMAGE_HEIGHT", 400)
    default_offset_x = vlcf_int(vlcf_settings, "IMAGE_OFFSET_X", 0)
    default_taps = vlcf_int(vlcf_settings, "TAP_NUMBER", 4)
    default_pixel_format = IKAP_PIXEL_FORMATS.get(default_depth, IKAP_PIXEL_FORMATS[12])[0]

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vlcf",
        default=DEFAULT_VLCF_PATH,
        help="IKap .vlcf board configuration file (default: AKCameraConfig\\12bit4tap.vlcf).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=default_width,
        help="Requested camera width (default from .vlcf IMAGE_WIDTH).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=default_height,
        help="Requested line-scan frame height (default from .vlcf IMAGE_HEIGHT).",
    )
    parser.add_argument(
        "--offset-x",
        type=int,
        default=default_offset_x,
        help="Requested camera OffsetX (default from .vlcf IMAGE_OFFSET_X).",
    )
    parser.add_argument(
        "--taps",
        type=int,
        default=default_taps,
        help="Camera SensorTaps; must match .vlcf TAP_NUMBER.",
    )
    parser.add_argument(
        "--pixel-format",
        default=default_pixel_format,
        help="Requested camera PixelFormat (default from .vlcf PIXEL_DEPTH).",
    )
    parser.add_argument("--exposure-us", type=float, default=10.0, help="Exposure time in microseconds.")
    parser.add_argument(
        "--line-rate",
        type=float,
        default=10000.0,
        help="Internal line rate in Hz (camera side; only meaningful without external line trigger).",
    )
    parser.add_argument("--frames", type=int, default=5, help="IKap frame buffer count.")
    parser.add_argument("--trigger-device", default="Galvo", help="ART-DAQ device name.")
    parser.add_argument("--trigger-line", default="port0/line0", help="ART-DAQ DO line for camera trigger.")
    parser.add_argument(
        "--integration-source",
        default="ShaftEncoder1",
        help="IKap integration trigger source (informational; the board follows .vlcf).",
    )
    parser.add_argument(
        "--encoder-multiply",
        default="1",
        help="ShaftEncoder1 multiply factor 1/2/4/8/16/32 (informational; the board follows .vlcf).",
    )
    parser.add_argument("--duration", type=float, default=1.0, help="Trigger output duration in seconds.")
    parser.add_argument("--save", default="ikap_test_last_frame.npy", help="Output .npy path for last frame.")
    parser.add_argument(
        "--capture",
        type=int,
        default=10,
        help="Number of frames to capture and show in the matplotlib grid (0 = disable).",
    )
    parser.add_argument(
        "--capture-timeout",
        type=float,
        default=8.0,
        help="Seconds to wait for the requested frames before giving up.",
    )
    parser.add_argument(
        "--trigger-mode",
        choices=("internal", "external"),
        default="internal",
        help=(
            "internal = board internal line trigger (no wiring needed, best for image verification); "
            "external = ShaftEncoder1 + ART-DAQ line trigger (original behaviour)."
        ),
    )
    parser.add_argument(
        "--dump-camera-features",
        nargs="?",
        const="",
        default=None,
        metavar="PATTERN",
        help="Print the camera feature list (optionally filtered by PATTERN) to explain failed feature sets.",
    )
    parser.add_argument(
        "--save-dir",
        default="",
        help="Optional directory: save every captured frame as .npy plus a PNG grid image.",
    )
    parser.add_argument(
        "--save-grid-png",
        action="store_true",
        help=(
            "Also write the frame grid as a PNG (requires --save-dir). Off by default because "
            "this machine's matplotlib/Pillow PNG writer can crash the process (0xC06D007E)."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    res = IKapC.ItkManInitialize()
    check_itk(res, "ItkManInitialize")
    grabber = IKapLineScanGrabber()
    try:
        grabber.open_first_cameralink()
        # 1) 先用 .vlcf（默认 AKCameraConfig\12bit4tap.vlcf）配置采集卡，
        #    板卡侧几何/位深/tap/触发源/编码器设置全部以该文件为准。
        grabber.load_configuration(args.vlcf)
        vlcf_settings = grabber.print_vlcf_settings(args.vlcf)
        # 2) 校验命令行传入的几何/位深/tap 是否与 .vlcf 一致（不一致只告警，板卡仍按文件工作）。
        grabber.check_vlcf_alignment(
            vlcf_settings,
            width=args.width,
            height=args.height,
            offset_x=args.offset_x,
            pixel_format=args.pixel_format,
            taps=args.taps,
        )
        # 3) 相机侧特征（不在 .vlcf 中）仍在线写入，数值默认取自 .vlcf 以保证两侧一致。
        grabber.configure_camera(
            width=args.width,
            height=args.height,
            exposure_us=args.exposure_us,
            line_rate_hz=args.line_rate,
            pixel_format=args.pixel_format,
            offset_x=args.offset_x,
            sensor_taps=args.taps,
        )
        if args.dump_camera_features is not None:
            # 打印相机真实特征表：用来解释上面 Width/SensorTaps/Trigger* 之类的设置为何失败。
            grabber.dump_camera_features(args.dump_camera_features)
        # 4) 触发：
        #    internal = 板卡内部行触发（不需要外部接线，最适合先确认成像是否正常）
        #    external = .vlcf 的 ShaftEncoder1 + ART-DAQ 行触发（原来的 OCT 链路）
        if args.trigger_mode == "internal":
            grabber.configure_board_internal_trigger(args.line_rate)
            grabber.configure_camera_line_trigger_for_external_pulse()
        else:
            grabber.configure_line_trigger(
                integration_source=args.integration_source,
                encoder_multiply=args.encoder_multiply,
            )
        capture_count = max(0, int(args.capture))
        grabber.capture_limit = capture_count
        grabber.captured_frames = []
        grabber.capture_start_time = None
        frame_buffers = max(int(args.frames), capture_count, 1)
        grabber.start_grab(frame_buffers)
        if args.trigger_mode == "external":
            time.sleep(0.2)
            output_artdaq_line_triggers(
                device=args.trigger_device,
                line=args.trigger_line,
                trigger_rate_hz=args.line_rate,
                duration_s=args.duration,
            )
        # 等待抓够 capture 张（或超时）
        deadline = time.time() + max(0.1, float(args.capture_timeout))
        while len(grabber.captured_frames) < capture_count and time.time() < deadline:
            time.sleep(0.05)
        time.sleep(0.1)
        try:
            grabber.stop_grab()
        except Exception as error:
            print("Stop grab failed:", error)
        print(
            "Grab summary: frames=%d, captured=%d/%d, lost=%d, timeouts=%d"
            % (
                grabber.frame_ready_count,
                len(grabber.captured_frames),
                capture_count,
                grabber.frame_lost_count,
                grabber.timeout_count,
            )
        )
        frames = list(grabber.captured_frames)
        if frames:
            print("Captured frame statistics:")
            print_frame_stats(frames)
        else:
            print(
                "No frame was captured. Check: .vlcf line-trigger source (ShaftEncoder1 vs internal), "
                "camera external-pulse wiring, and that IKapCViewer/other software is closed."
            )
        if grabber.last_frame is not None and args.save:
            np.save(args.save, grabber.last_frame)
            print("Saved last frame:", os.path.abspath(args.save), grabber.last_frame.shape, grabber.last_frame.dtype)
        grid_path = ""
        if args.save_dir:
            save_frames(frames, args.save_dir)
            if args.save_grid_png:
                grid_path = os.path.join(args.save_dir, "ikap_frames_grid.png")
        if frames:
            show_frames(
                frames,
                title="IKap %s, %d frames (%s)" % (os.path.basename(args.vlcf), len(frames), args.trigger_mode),
                save_path=grid_path or None,
            )
    finally:
        try:
            grabber.stop_grab()
        finally:
            grabber.close()
            IKapC.ItkManTerminate()


if __name__ == "__main__":
    main()
