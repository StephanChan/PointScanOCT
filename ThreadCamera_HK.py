# -*- coding: utf-8 -*-
"""
HiK/MVS camera worker thread with the same queue/memory shape as
ThreadCamera_DH.py.

UI integration notes:
This module reads HiK-specific widgets only. Keep Daheng and HiK camera controls
separate in the UI so changing one camera tab does not silently affect the
other camera backend.
"""

import ctypes
import os
import platform
import sys
import time
import traceback
from ctypes import *

import numpy as np
from PyQt5.QtCore import QThread

from ActionFields import DbackActionField
from CameraUi import (
    downsample_spectral_axis,
    effective_camera_sample_count,
    raw_camera_sample_count,
    spectral_downsample,
    x_pixel_count,
)


global SIM

MVS_DEV_ROOT = r"D:\MVS\MVS\Development"
MVS_RUNTIME_DIR = r"C:\Program Files (x86)\Common Files\MVS\Runtime\Win64_x64"

CONTINUOUS = 0x7FFFFFFF
HK_MEMORY_WRITE_METHOD = "direct"
HK_CONSUMER_WORKERS = 4
HK_DEFAULT_DEVICE_INDEX = 0
HK_DEFAULT_LINE_RATE = 10000
HK_DEFAULT_PIXEL_FORMAT = "Mono12Packed"
HK_DEFAULT_TRIGGER_SELECTOR = "LineStart"
HK_DEFAULT_TRIGGER_SOURCE = "Line1"
HK_DEFAULT_TRIGGER_ACTIVATION = "RisingEdge"
HK_MIN_GRAB_TIMEOUT_MS = 1000
HK_FRAME_TIMEOUT_MARGIN_MS = 1500
HK_PRINT_CAMERA_SETTINGS = False


def _bootstrap_mvs_sdk():
    if platform.system() == "Windows":
        ctypes.windll.kernel32.SetDllDirectoryW(MVS_RUNTIME_DIR)
        os.environ["PATH"] = MVS_RUNTIME_DIR + os.pathsep + os.environ.get("PATH", "")
        import_dir = os.path.join(MVS_DEV_ROOT, "Samples", "Python", "MvImport")
        if import_dir not in sys.path:
            sys.path.append(import_dir)
    else:
        import_dir = os.path.join("..", "..", "MvImport")
        if import_dir not in sys.path:
            sys.path.append(import_dir)


try:
    _bootstrap_mvs_sdk()
    from MvCameraControl_class import *  # noqa: F401,F403,E402
    from PixelType_header import (  # noqa: E402
        PixelType_Gvsp_Mono8,
        PixelType_Gvsp_Mono10,
        PixelType_Gvsp_Mono10_Packed,
        PixelType_Gvsp_Mono12,
        PixelType_Gvsp_Mono12_Packed,
        PixelType_Gvsp_Mono14,
        PixelType_Gvsp_Mono16,
    )

    SIM = False
except Exception as error:
    print(
        "HiK/MVS SDK import failed. Check MVS paths: "
        f"MVS_DEV_ROOT={MVS_DEV_ROOT}, MVS_RUNTIME_DIR={MVS_RUNTIME_DIR}. "
        f"Import error: {error}. Using simulation."
    )
    SIM = True


MONO8_TYPES = {PixelType_Gvsp_Mono8} if not SIM else set()
MONO16_COMPATIBLE_TYPES = (
    {
        PixelType_Gvsp_Mono10,
        PixelType_Gvsp_Mono12,
        PixelType_Gvsp_Mono14,
        PixelType_Gvsp_Mono16,
    }
    if not SIM
    else set()
)
MONO_PACKED_TYPES = (
    {
        PixelType_Gvsp_Mono10_Packed,
        PixelType_Gvsp_Mono12_Packed,
    }
    if not SIM
    else set()
)


def check_ret(ret, message):
    if ret != 0:
        raise RuntimeError("%s ret[0x%x]" % (message, ret))


def warn_ret(ret, message):
    if ret != 0:
        print("warning: %s ret[0x%x]" % (message, ret))
        return False
    return True


def decoding_char(ctypes_char_array):
    byte_str = memoryview(ctypes_char_array).tobytes()
    null_index = byte_str.find(b"\x00")
    if null_index != -1:
        byte_str = byte_str[:null_index]
    for encoding in ("gbk", "utf-8", "latin-1"):
        try:
            return byte_str.decode(encoding)
        except UnicodeDecodeError:
            pass
    return byte_str.decode("latin-1", errors="replace")


def enum_devices():
    device_list = MV_CC_DEVICE_INFO_LIST()
    layer_type = (
        MV_GIGE_DEVICE
        | MV_USB_DEVICE
        | MV_GENTL_CAMERALINK_DEVICE
        | MV_GENTL_CXP_DEVICE
        | MV_GENTL_XOF_DEVICE
    )
    ret = MvCamera.MV_CC_EnumDevices(layer_type, device_list)
    check_ret(ret, "enum devices failed")
    return device_list


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
        if hasattr(widget, "setValue"):
            widget.setValue(value)
            return True
        if hasattr(widget, "setText"):
            widget.setText(str(value))
            return True
    return False


class Camera(QThread):
    def __init__(self):
        super().__init__()
        self.MemoryLoc = 0
        self.exit_message = "HiK camera thread exited."
        self.cam = None
        self.device_info = None
        self.is_open = False
        self.is_grabbing = False
        self.memory_write_method = HK_MEMORY_WRITE_METHOD
        self._mvs_initialized = False
        self.SIM = SIM

    def run(self):
        if not (SIM or self.SIM):
            print("initializing HiK camera...")
            self.initCamera()
            self.GetExposure()
            self.GetGain()
            self.GetTemp()
            self.GetPixelDepth()
        self.QueueOut()

    def QueueOut(self):
        self.item = self.queue.get()
        while self.item.action != "exit":
            try:
                if self.item.action == "ConfigureBoard":
                    self.ConfigureBoard()
                elif self.item.action == "Acquire":
                    if self.cam is not None:
                        self.Stream_on()
                        self.Acquire()
                        self.Stream_off()
                    else:
                        self.simData()
                elif self.item.action == "GetTemp":
                    self.GetTemp()
                else:
                    self.emit_status(f"Unknown HiK camera command: {self.item.action}")
            except Exception as error:
                message = "HiK camera command failed. This action was skipped: " + str(error)
                self.emit_status(message)
                print(message)
                print(traceback.format_exc())
            self.item = self.queue.get()
        self.Close()
        print(self.exit_message)
        self.emit_status(self.exit_message)

    def emit_status(self, message):
        if message is not None:
            self.ui_bridge.status_message.emit(str(message))

    def initCamera(self):
        MvCamera.MV_CC_Initialize()
        self._mvs_initialized = True

        device_list = enum_devices()
        if device_list.nDeviceNum == 0:
            print("No HiK camera found")
            setattr(self.ui, "_hik_camera_found", False)
            self.cam = None
            return
        setattr(self.ui, "_hik_camera_found", True)

        device_index = int(ui_value(self.ui, ["DeviceIndex_HK"], HK_DEFAULT_DEVICE_INDEX))
        if device_index >= device_list.nDeviceNum:
            device_index = HK_DEFAULT_DEVICE_INDEX
        self.device_info = cast(
            device_list.pDeviceInfo[device_index],
            POINTER(MV_CC_DEVICE_INFO),
        ).contents

        self.cam = MvCamera()
        ret = self.cam.MV_CC_CreateHandle(self.device_info)
        check_ret(ret, "create HiK camera handle failed")
        ret = self.cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
        check_ret(ret, "open HiK camera failed")
        self.is_open = True

        if self.device_info.nTLayerType in (MV_GIGE_DEVICE, MV_GENTL_GIGE_DEVICE):
            packet_size = self.cam.MV_CC_GetOptimalPacketSize()
            if int(packet_size) > 0:
                warn_ret(
                    self.cam.MV_CC_SetIntValue("GevSCPSPacketSize", packet_size),
                    "set GevSCPSPacketSize failed",
                )
        warn_ret(self.cam.MV_CC_SetImageNodeNum(128), "set image node number failed")
        self.set_enum("ImageCompressionMode", "Off", strict=False)

    def ConfigureBoard(self):
        self.AlinesPerBline = 1
        self.NSamples_HK = int(raw_camera_sample_count(self.ui))
        self.SpectralDS = spectral_downsample(self.ui)
        self.ProcessedSamples = effective_camera_sample_count(self.ui)
        x_pixels = x_pixel_count(self.ui)
        bline_avg = max(1, int(self.ui.BlineAVG.value()))

        if self.ui.ACQMode.currentText() in ["FiniteBscan"]:
            self.BlinesPerAcq = x_pixels * bline_avg
        elif self.ui.ACQMode.currentText() in [
            "ContinuousBline",
            "ContinuousCscan",
        ]:
            self.BlinesPerAcq = CONTINUOUS
        elif self.ui.ACQMode.currentText() in [
            "FiniteCscan",
        ]:
            self.BlinesPerAcq = self.ui.Ypixels.value() * x_pixels * bline_avg
        else:
            self.BlinesPerAcq = x_pixels * bline_avg

        if self.cam is not None:
            self.SetExposure()
            self.SetGain()
            self.SetPixelDepth()
            self.SetTrigger()
            self.SetLineRate()
            self.SetROI()
            self.GetTemp()
            if HK_PRINT_CAMERA_SETTINGS:
                self.print_configuration_readback()
        self.DbackQueue.put(0)

    def set_enum(self, name, value, strict=True):
        ret = self.cam.MV_CC_SetEnumValueByString(name, str(value))
        if strict:
            check_ret(ret, "set %s to %s failed" % (name, value))
            return True
        return warn_ret(ret, "set %s to %s failed" % (name, value))

    def set_bool(self, name, value, strict=True):
        ret = self.cam.MV_CC_SetBoolValue(name, bool(value))
        if strict:
            check_ret(ret, "set %s failed" % name)
            return True
        return warn_ret(ret, "set %s failed" % name)

    def set_int(self, name, value, strict=True):
        ret = self.cam.MV_CC_SetIntValue(name, int(value))
        if strict:
            check_ret(ret, "set %s failed" % name)
            return True
        return warn_ret(ret, "set %s failed" % name)

    def set_float(self, name, value, strict=True):
        ret = self.cam.MV_CC_SetFloatValue(name, float(value))
        if strict:
            check_ret(ret, "set %s failed" % name)
            return True
        return warn_ret(ret, "set %s failed" % name)

    def get_int_feature(self, name):
        value = MVCC_INTVALUE()
        memset(byref(value), 0, sizeof(value))
        ret = self.cam.MV_CC_GetIntValue(name, value)
        return value if ret == 0 else None

    def get_float_feature(self, name):
        value = MVCC_FLOATVALUE()
        memset(byref(value), 0, sizeof(value))
        ret = self.cam.MV_CC_GetFloatValue(name, value)
        return value if ret == 0 else None

    def get_string_feature(self, name):
        value = MVCC_STRINGVALUE()
        memset(byref(value), 0, sizeof(value))
        ret = self.cam.MV_CC_GetStringValue(name, value)
        return decoding_char(value.chCurValue) if ret == 0 else None

    def get_enum_feature(self, name):
        value = MVCC_ENUMVALUE()
        memset(byref(value), 0, sizeof(value))
        ret = self.cam.MV_CC_GetEnumValue(name, value)
        return value if ret == 0 else None

    def print_int_feature(self, name):
        value = self.get_int_feature(name)
        if value is None:
            print("  %s: unavailable" % name)
            return
        print(
            "  %s: current=%d min=%d max=%d inc=%d"
            % (name, value.nCurValue, value.nMin, value.nMax, value.nInc)
        )

    def print_float_feature(self, name):
        value = self.get_float_feature(name)
        if value is None:
            print("  %s: unavailable" % name)
            return
        print(
            "  %s: current=%.6g min=%.6g max=%.6g"
            % (name, value.fCurValue, value.fMin, value.fMax)
        )

    def print_enum_feature(self, name):
        value = self.get_enum_feature(name)
        if value is None:
            print("  %s: unavailable" % name)
            return
        print("  %s: current_numeric=%d" % (name, value.nCurValue))

    def print_string_feature(self, name):
        value = self.get_string_feature(name)
        if value is None:
            print("  %s: unavailable" % name)
            return
        print("  %s: %s" % (name, value))

    def print_configuration_readback(self):
        print("HiK camera configuration readback:")
        print("  UI/worker request:")
        print("    ACQMode: %s" % self.ui.ACQMode.currentText())
        print("    AlinesPerBline/Height request: %d" % int(self.AlinesPerBline))
        print("    NSamples_HK/Width request: %d" % int(self.NSamples_HK))
        print("    SpectralDS_HK: %d" % int(self.SpectralDS))
        print("    Processed samples: %d" % int(self.ProcessedSamples))
        print("    BlinesPerAcq: %s" % str(self.BlinesPerAcq))
        print("    BlineAVG: %d" % int(self.ui.BlineAVG.value()))
        print("    Ypixels: %d" % int(self.ui.Ypixels.value()))
        print("    PixelFormat_HK: %s" % ui_value(self.ui, ["PixelFormat_HK"], HK_DEFAULT_PIXEL_FORMAT))
        print("    Exposure_HK_ms: %.6g" % float(ui_value(self.ui, ["Exposure_HK"], 1.0)))
        print("    DGain_HK: %.6g" % float(ui_value(self.ui, ["DGain_HK"], 0.0)))
        print("    LineRate_HK: %d" % int(ui_value(self.ui, ["LineRate_HK"], HK_DEFAULT_LINE_RATE)))
        print("    TriggerSelector_HK: %s" % ui_value(self.ui, ["TriggerSelector_HK"], HK_DEFAULT_TRIGGER_SELECTOR))
        print("    TriggerON_HK: %s" % ui_value(self.ui, ["TriggerON_HK"], "Off"))
        print("    TriggerSource_HK: %s" % ui_value(self.ui, ["TriggerSource_HK"], HK_DEFAULT_TRIGGER_SOURCE))
        print("    TriggerActivation_HK: %s" % ui_value(self.ui, ["TriggerActivation_HK"], HK_DEFAULT_TRIGGER_ACTIVATION))
        print("    OffsetX_HK: %s" % str(ui_value(self.ui, ["offsetX_HK"], 0)))
        print("    OffsetY_HK: %s" % str(ui_value(self.ui, ["offsetY_HK"], 0)))
        try:
            print("    Memory[0].shape: %s dtype=%s" % (str(self.Memory[0].shape), self.Memory[0].dtype))
        except Exception as error:
            print("    Memory[0].shape: unavailable (%s)" % error)

        print("  Camera identity:")
        for name in ("DeviceModelName", "DeviceSerialNumber", "DeviceUserID"):
            self.print_string_feature(name)

        print("  Camera ROI/payload:")
        for name in ("Width", "Height", "OffsetX", "OffsetY", "PayloadSize"):
            self.print_int_feature(name)

        print("  Camera trigger/format:")
        for name in (
            "PixelFormat",
            "TriggerSelector",
            "TriggerMode",
            "TriggerSource",
            "TriggerActivation",
            "ImageCompressionMode",
        ):
            self.print_enum_feature(name)

        print("  Camera timing/exposure/gain:")
        for name in ("AcquisitionLineRate", "ResultingLineRate"):
            self.print_int_feature(name)
        for name in ("ExposureTime", "Gain", "DeviceTemperature"):
            self.print_float_feature(name)

    def SetROI(self):
        # HiK line-scan camera axes:
        # camera Width = spectral samples, camera Height = one A-line.
        self.set_int("OffsetX", ui_value(self.ui, ["offsetW_HK", "offsetX_HK"], 0), strict=False)
        self.set_int("OffsetY", ui_value(self.ui, ["offsetY_HK"], 0), strict=False)
        self.set_int("Width", self.NSamples_HK, strict=True)
        self.set_int("Height", self.AlinesPerBline, strict=True)

    def SetTrigger(self):
        trigger_selector = ui_value(
            self.ui, ["TriggerSelector_HK"], HK_DEFAULT_TRIGGER_SELECTOR
        )
        trigger_mode = ui_value(self.ui, ["TriggerON_HK"], "Off")
        trigger_source = ui_value(
            self.ui, ["TriggerSource_HK"], HK_DEFAULT_TRIGGER_SOURCE
        )
        trigger_activation = ui_value(
            self.ui,
            ["TriggerActivation_HK"],
            HK_DEFAULT_TRIGGER_ACTIVATION,
        )

        self.set_enum("TriggerSelector", trigger_selector, strict=False)
        if trigger_mode == "Off":
            ret = self.cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
            check_ret(ret, "set TriggerMode Off failed")
            return

        self.set_enum("TriggerSource", trigger_source, strict=True)
        self.set_enum("TriggerActivation", trigger_activation, strict=False)
        ret = self.cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_ON)
        check_ret(ret, "set TriggerMode On failed")

    def SetLineRate(self):
        line_rate = int(ui_value(self.ui, ["LineRate_HK"], HK_DEFAULT_LINE_RATE))
        self.set_int("AcquisitionLineRate", line_rate, strict=True)
        self.set_bool("AcquisitionLineRateEnable", True, strict=False)
        resulting = self.get_int_feature("ResultingLineRate")
        if resulting is not None:
            set_ui_value(self.ui, ["LineRate_display_HK"], int(resulting.nCurValue))
            if HK_PRINT_CAMERA_SETTINGS:
                print("HiK ResultingLineRate: %d" % int(resulting.nCurValue))

    def SetPixelDepth(self):
        pixel_format = ui_value(
            self.ui, ["PixelFormat_HK"], HK_DEFAULT_PIXEL_FORMAT
        )
        self.set_enum("PixelFormat", pixel_format, strict=True)
        set_ui_value(self.ui, ["PixelFormat_display_HK"], pixel_format)

    def GetPixelDepth(self):
        pixel_format = ui_value(
            self.ui, ["PixelFormat_HK"], HK_DEFAULT_PIXEL_FORMAT
        )
        set_ui_value(self.ui, ["PixelFormat_display_HK"], pixel_format)

    def SetExposure(self):
        exposure_ms = float(ui_value(self.ui, ["Exposure_HK"], 1.0))
        self.set_enum("ExposureAuto", "Off", strict=False)
        self.set_float("ExposureTime", exposure_ms * 1000.0, strict=True)
        self.GetExposure()

    def GetExposure(self):
        if self.cam is None:
            return
        value = self.get_float_feature("ExposureTime")
        if value is not None:
            set_ui_value(self.ui, ["Exposure_display_HK"], value.fCurValue / 1000.0)

    def SetGain(self):
        gain = float(ui_value(self.ui, ["DGain_HK"], 0.0))
        self.set_enum("GainAuto", "Off", strict=False)
        self.set_float("Gain", gain, strict=True)
        self.GetGain()

    def GetGain(self):
        if self.cam is None:
            return
        value = self.get_float_feature("Gain")
        if value is not None:
            set_ui_value(self.ui, ["DGain_display_HK"], value.fCurValue)

    def GetTemp(self):
        if self.cam is None:
            return
        value = self.get_float_feature("DeviceTemperature")
        if value is not None:
            set_ui_value(self.ui, ["Temporature_HK"], value.fCurValue)

    def estimate_frame_timeout_ms(self):
        height = self.get_int_feature("Height")
        line_rate = self.get_int_feature("AcquisitionLineRate")
        if height is None or line_rate is None or line_rate.nCurValue <= 0:
            return HK_MIN_GRAB_TIMEOUT_MS
        frame_ms = int(height.nCurValue / float(line_rate.nCurValue) * 1000.0)
        return max(HK_MIN_GRAB_TIMEOUT_MS, frame_ms + HK_FRAME_TIMEOUT_MARGIN_MS)

    def Stream_on(self):
        if self.cam is not None and not self.is_grabbing:
            ret = self.cam.MV_CC_StartGrabbing()
            check_ret(ret, "start HiK grabbing failed")
            self.is_grabbing = True

    def Stream_off(self):
        if self.cam is not None and self.is_grabbing:
            ret = self.cam.MV_CC_StopGrabbing()
            warn_ret(ret, "stop HiK grabbing failed")
            self.is_grabbing = False

    def frame_to_numpy(self, frame):
        info = frame.stFrameInfo
        pixel_type = info.enPixelType
        width = int(info.nWidth)
        height = int(info.nHeight)
        frame_len = int(info.nFrameLen)

        if pixel_type in MONO8_TYPES:
            data = np.ctypeslib.as_array(
                cast(frame.pBufAddr, POINTER(c_ubyte)),
                shape=(frame_len,),
            )
            return data[: width * height].reshape(height, width).copy()

        if pixel_type in MONO16_COMPATIBLE_TYPES and frame_len >= width * height * 2:
            data = np.ctypeslib.as_array(
                cast(frame.pBufAddr, POINTER(c_ushort)),
                shape=(width * height,),
            )
            return data.reshape(height, width).copy()

        if pixel_type in MONO_PACKED_TYPES:
            return self.convert_frame_to_mono16(frame)

        raise RuntimeError("unsupported HiK pixel type: %s" % pixel_type)

    def convert_frame_to_mono16(self, frame):
        info = frame.stFrameInfo
        width = int(info.nWidth)
        height = int(info.nHeight)
        dst_len = width * height * 2
        dst_buf = (c_ubyte * dst_len)()

        cvt = MV_CC_PIXEL_CONVERT_PARAM_EX()
        memset(byref(cvt), 0, sizeof(cvt))
        cvt.nWidth = width
        cvt.nHeight = height
        cvt.enSrcPixelType = info.enPixelType
        cvt.pSrcData = frame.pBufAddr
        cvt.nSrcDataLen = info.nFrameLen
        cvt.enDstPixelType = PixelType_Gvsp_Mono16
        cvt.pDstBuffer = dst_buf
        cvt.nDstBufferSize = dst_len

        ret = self.cam.MV_CC_ConvertPixelTypeEx(cvt)
        check_ret(ret, "convert HiK frame to Mono16 failed")
        data = np.ctypeslib.as_array(cast(cvt.pDstBuffer, POINTER(c_ushort)), shape=(width * height,))
        data = data.reshape(height, width).copy()
        if info.enPixelType == PixelType_Gvsp_Mono12_Packed:
            data >>= 4
        elif info.enPixelType == PixelType_Gvsp_Mono10_Packed:
            data >>= 6
        return data

    def Acquire(self):
        NBlines = self.Memory[0].shape[0]
        BlinesCount = 0
        timeout_ms = self.estimate_frame_timeout_ms()
        current_slot = self.MemoryFreeQueue.get() if hasattr(self, "MemoryFreeQueue") else self.MemoryLoc
        package_start_time = time.perf_counter()
        self.DbackQueue.put(0)

        while BlinesCount < self.BlinesPerAcq and self.ui.RunButton.isChecked():
            frame = MV_FRAME_OUT()
            memset(byref(frame), 0, sizeof(frame))
            ret = self.cam.MV_CC_GetImageBuffer(frame, int(timeout_ms))
            if ret != 0 or frame.pBufAddr is None:
                print("HiK camera timeout ret[0x%x] timeout[%d ms]" % (ret, timeout_ms))
                bline = np.zeros(
                    [1, self.NSamples_HK],
                    dtype=self.Memory[current_slot].dtype,
                )
            else:
                try:
                    bline = self.frame_to_numpy(frame)
                finally:
                    warn_ret(self.cam.MV_CC_FreeImageBuffer(frame), "free HiK image buffer failed")
            bline = downsample_spectral_axis(bline, self.SpectralDS, axis=1)

            self.write_bline_to_memory(bline, current_slot, BlinesCount % NBlines)
            BlinesCount += 1
            # print(BlinesCount)

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

    def write_bline_to_memory(self, bline, memory_slot, frame_index):
        dest = self.Memory[memory_slot][frame_index]
        bline = np.asarray(bline)
        if bline.ndim == 1:
            bline = bline.reshape(1, -1)
        if self.memory_write_method == "direct":
            if bline.shape == dest.shape:
                np.copyto(dest, bline)
            elif bline.T.shape == dest.shape:
                np.copyto(dest, bline.T)
            else:
                raise ValueError(
                    "HiK frame shape %s does not match destination %s"
                    % (str(bline.shape), str(dest.shape))
                )
        elif self.memory_write_method == "auto":
            if bline.shape == dest.shape:
                np.copyto(dest, bline)
            elif bline.T.shape == dest.shape:
                np.copyto(dest, bline.T)
            else:
                raise ValueError(
                    "HiK frame shape %s does not match destination %s"
                    % (str(bline.shape), str(dest.shape))
                )
        elif self.memory_write_method == "copyto_transpose":
            np.copyto(dest, bline.T)
        elif self.memory_write_method == "swapaxes_copyto":
            np.copyto(dest, np.swapaxes(bline, 0, 1))
        elif self.memory_write_method == "assign_transpose":
            dest[...] = bline.T
        else:
            raise ValueError(f"Unknown HiK memory write method: {self.memory_write_method}")

    def Close(self):
        if self.cam is not None:
            self.Stream_off()
            if self.is_open:
                warn_ret(self.cam.MV_CC_CloseDevice(), "close HiK camera failed")
                self.is_open = False
            warn_ret(self.cam.MV_CC_DestroyHandle(), "destroy HiK camera handle failed")
            self.cam = None
        if self._mvs_initialized:
            MvCamera.MV_CC_Finalize()
            self._mvs_initialized = False

    def simData(self):
        NBlines = self.Memory[0].shape[0]
        BlinesCount = 0
        current_slot = self.MemoryFreeQueue.get() if hasattr(self, "MemoryFreeQueue") else self.MemoryLoc
        package_start_time = time.perf_counter()
        sim_start_time = time.perf_counter()
        sim_line_rate = max(1.0, float(ui_value(self.ui, ["LineRate_HK"], HK_DEFAULT_LINE_RATE)))
        self.DbackQueue.put(0)
        while BlinesCount < self.BlinesPerAcq and self.ui.RunButton.isChecked():
            pixel_format = ui_value(self.ui, ["PixelFormat_HK"], HK_DEFAULT_PIXEL_FORMAT)
            if pixel_format == "Mono8":
                bline = np.zeros([1, self.ProcessedSamples], dtype=np.uint8)
            else:
                bline = np.zeros([1, self.ProcessedSamples], dtype=np.uint16)
            self.Memory[current_slot][BlinesCount % NBlines] = bline
            BlinesCount += 1
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
