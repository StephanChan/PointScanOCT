# -*- coding: utf-8 -*-
"""PyQt UI for I-TEK IKap line-scan camera acquisition and ART-DAQ triggers."""

import os
import sys
import time

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from IKapCamera_test import (
    IKapBoard,
    IKapC,
    IKapCDef,
    IKapLineScanGrabber,
    check_itk,
    describe_itk_status,
    output_artdaq_line_triggers_count,
    read_board_int,
    read_feature_enum,
    read_feature_float,
    read_feature_int,
    set_board_int,
    set_feature_enum,
    set_feature_enum_any,
    set_feature_float,
    set_feature_int,
)


AK_CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "AKCameraConfig")
AK_TAP_VLCF_FILES = {
    ("8-bit", 8): "8bit8tap.vlcf",
    ("8-bit", 10): "8bit10tap.vlcf",
    ("10-bit", 8): "10bit8tap.vlcf",
}
CAMERA_SENSOR_TAP_VALUES = {
    2: ("TwoTap", "Two Tap", "2Tap", "2"),
    4: ("FourTap", "Four Tap", "4Tap", "4"),
    8: ("EightTap", "Eight Tap", "8Tap", "8"),
    10: ("TenTap", "Ten Tap", "10Tap", "10"),
}


def matching_ccf_path(vlcf_path):
    """与 .vlcf 同目录同名的相机配置文件 .ccf（不存在时返回空字符串）。

    板卡侧参数由 .vlcf 决定，相机侧参数（触发模式 SynchronizationMode 等）由 .ccf 决定，
    因此加载 .vlcf 时应当同时加载同名 .ccf，保持两侧一致。
    """
    text = str(vlcf_path or "").strip()
    if not text:
        return ""
    path = os.path.splitext(os.path.abspath(text))[0] + ".ccf"
    return path if os.path.isfile(path) else ""


def frame_to_qimage(frame):
    if frame is None:
        return None
    data = np.asarray(frame)
    if data.ndim == 3:
        data = data[:, :, 0]
    if data.ndim != 2:
        return None
    data = np.asarray(data)
    if data.dtype != np.uint8:
        max_value = float(np.max(data)) if data.size else 0.0
        min_value = float(np.min(data)) if data.size else 0.0
        if max_value <= min_value:
            scaled = np.zeros(data.shape, dtype=np.uint8)
        else:
            scaled = ((data.astype(np.float32) - min_value) * (255.0 / (max_value - min_value))).astype(np.uint8)
    else:
        scaled = data
    scaled = np.ascontiguousarray(scaled)
    height, width = scaled.shape
    return QtGui.QImage(scaled.data, width, height, width, QtGui.QImage.Format_Grayscale8).copy()


def frame_for_display(frame):
    data = np.asarray(frame)
    if data.ndim == 3:
        data = data[:, :, 0]
    if data.ndim != 2:
        return data
    if data.shape[0] < 64:
        repeat_count = int(np.ceil(128 / max(1, data.shape[0])))
        data = np.repeat(data, repeat_count, axis=0)[:128, :]
    return data


class TriggerWorker(QtCore.QThread):
    status = QtCore.pyqtSignal(str)
    failed = QtCore.pyqtSignal(str)
    finished_ok = QtCore.pyqtSignal()
    output_started = QtCore.pyqtSignal(float)

    def __init__(self, device, line, rate_hz, trigger_count):
        super().__init__()
        self.device = device
        self.line = line
        self.rate_hz = rate_hz
        self.trigger_count = trigger_count

    def run(self):
        try:
            self.status.emit(
                "ART-DAQ output started: %s/%s, %.3f Hz, %d triggers"
                % (self.device, self.line, self.rate_hz, self.trigger_count)
            )
            output_artdaq_line_triggers_count(
                device=self.device,
                line=self.line,
                trigger_rate_hz=self.rate_hz,
                trigger_count=self.trigger_count,
                start_callback=self.output_started.emit,
            )
            self.status.emit("ART-DAQ output completed.")
            self.finished_ok.emit()
        except Exception as error:
            self.failed.emit(str(error))


class IKapCameraWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("IKap 线阵相机采集控制")
        self.resize(1180, 760)
        self.grabber = None
        self.ikap_initialized = False
        self.trigger_worker = None
        self.last_saved_dir = os.getcwd()
        self.last_displayed_count = -1
        self.image_timing_start_s = None
        self.trigger_timing_start_s = None
        self.trigger_timing_base_frame = 0
        self.trigger_timing_last_logged_frame = 0
        self.trigger_timing_rate_hz = 0.0
        self.trigger_timing_height = 1
        self.trigger_timing_count = 0
        self._build_ui()
        self._connect_signals()
        self.display_timer = QtCore.QTimer(self)
        self.display_timer.timeout.connect(self.refresh_display)
        self.display_timer.start(100)

    def _build_ui(self):
        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)
        root = QtWidgets.QHBoxLayout(central)

        controls = QtWidgets.QWidget()
        controls.setMaximumWidth(390)
        controls_layout = QtWidgets.QVBoxLayout(controls)
        root.addWidget(controls)

        display_panel = QtWidgets.QWidget()
        display_layout = QtWidgets.QVBoxLayout(display_panel)
        root.addWidget(display_panel, 1)

        device_box = QtWidgets.QGroupBox("设备")
        device_form = QtWidgets.QFormLayout(device_box)
        self.vlcf_edit = QtWidgets.QLineEdit()
        self.vlcf_button = QtWidgets.QPushButton("浏览")
        vlcf_row = QtWidgets.QHBoxLayout()
        vlcf_row.addWidget(self.vlcf_edit, 1)
        vlcf_row.addWidget(self.vlcf_button)
        device_form.addRow("配置文件 VLCF", vlcf_row)
        self.open_button = QtWidgets.QPushButton("打开相机")
        self.close_button = QtWidgets.QPushButton("关闭")
        open_row = QtWidgets.QHBoxLayout()
        open_row.addWidget(self.open_button)
        open_row.addWidget(self.close_button)
        device_form.addRow(open_row)
        controls_layout.addWidget(device_box)

        camera_box = QtWidgets.QGroupBox("相机参数")
        camera_form = QtWidgets.QFormLayout(camera_box)
        self.width_spin = QtWidgets.QSpinBox()
        self.width_spin.setRange(1, 20000)
        self.width_spin.setValue(4096)
        self.height_spin = QtWidgets.QSpinBox()
        self.height_spin.setRange(1, 20000)
        self.height_spin.setValue(1)
        self.exposure_spin = QtWidgets.QDoubleSpinBox()
        self.exposure_spin.setRange(0.1, 1000000.0)
        self.exposure_spin.setDecimals(3)
        self.exposure_spin.setValue(10.0)
        self.line_rate_spin = QtWidgets.QDoubleSpinBox()
        self.line_rate_spin.setRange(0.1, 200000.0)
        self.line_rate_spin.setDecimals(3)
        self.line_rate_spin.setValue(10000.0)
        self.pixel_format_combo = QtWidgets.QComboBox()
        self.pixel_format_combo.addItems(["8-bit", "10-bit", "12-bit", "14-bit", "16-bit"])
        self.camera_tap_combo = QtWidgets.QComboBox()
        self.camera_tap_combo.addItems(["2", "4", "8", "10"])
        self.camera_tap_combo.setCurrentText("8")
        self.board_tap_combo = QtWidgets.QComboBox()
        self.board_tap_combo.addItems(["1", "2", "3", "4", "6", "8", "10"])
        self.board_tap_combo.setCurrentText("8")
        self.auto_vlcf_checkbox = QtWidgets.QCheckBox("Auto load tap VLCF")
        self.auto_vlcf_checkbox.setChecked(True)
        self.pga_gain_combo = QtWidgets.QComboBox()
        self.pga_gain_combo.addItems(["x1", "x2", "x5", "x10"])
        self.digital_gain_spin = QtWidgets.QDoubleSpinBox()
        self.digital_gain_spin.setRange(0.01, 8.0)
        self.digital_gain_spin.setDecimals(2)
        self.digital_gain_spin.setSingleStep(0.01)
        self.digital_gain_spin.setValue(1.0)
        self.digital_offset_spin = QtWidgets.QSpinBox()
        self.digital_offset_spin.setRange(-4095, 4095)
        self.digital_offset_spin.setValue(0)
        self.apply_camera_button = QtWidgets.QPushButton("应用相机参数")
        self.readback_button = QtWidgets.QPushButton("读取实际参数")
        camera_form.addRow("宽度", self.width_spin)
        camera_form.addRow("高度", self.height_spin)
        camera_form.addRow("曝光 us", self.exposure_spin)
        camera_form.addRow("行频 Hz", self.line_rate_spin)
        camera_form.addRow("像素格式", self.pixel_format_combo)
        camera_form.addRow("PGA 增益", self.pga_gain_combo)
        camera_form.addRow("数字增益", self.digital_gain_spin)
        camera_form.addRow("数字偏置", self.digital_offset_spin)
        camera_form.addRow("SensorTaps", self.camera_tap_combo)
        camera_form.addRow("Tap number", self.board_tap_combo)
        camera_form.addRow(self.auto_vlcf_checkbox)
        camera_form.addRow(self.apply_camera_button)
        camera_form.addRow(self.readback_button)
        controls_layout.addWidget(camera_box)

        trigger_box = QtWidgets.QGroupBox("触发 / 采集")
        trigger_form = QtWidgets.QFormLayout(trigger_box)
        self.acq_mode_combo = QtWidgets.QComboBox()
        self.acq_mode_combo.addItem("行触发：ShaftEncoder1 + CC1", "LineTrigger_ARTDAQ")
        self.acq_mode_combo.addItem("自由运行：InternalFreeRun", "FreeRun")
        self.acq_mode_combo.addItem("通用输入1硬触发", "GeneralInput1_Hardware")
        self.acq_mode_combo.addItem("通用输入1门控 + 行触发", "GeneralInput1_ShaftEncoder")
        self.acq_mode_combo.addItem("软件触发", "Software")
        self.working_mode_combo = QtWidgets.QComboBox()
        self.working_mode_combo.addItems(["Free Run", "Hardware Trigger"])
        self.integration_source_combo = QtWidgets.QComboBox()
        self.integration_source_combo.addItems(["ShaftEncoder1", "GeneralInput1", "GeneralInput2", "Internal", "Software"])
        self.encoder_multiply_combo = QtWidgets.QComboBox()
        self.encoder_multiply_combo.addItems(["1", "2", "4", "8", "16", "32"])
        self.trigger_selector_combo = QtWidgets.QComboBox()
        self.trigger_selector_combo.addItems(["LineStart", "FrameStart"])
        self.trigger_source_combo = QtWidgets.QComboBox()
        self.trigger_source_combo.addItems(["General Input1", "General Input2", "Line1", "Line2", "Software"])
        self.trigger_activation_combo = QtWidgets.QComboBox()
        self.trigger_activation_combo.addItems(["RisingEdge", "FallingEdge", "AnyEdge"])
        self.artdaq_device_edit = QtWidgets.QLineEdit("Galvo")
        self.artdaq_line_edit = QtWidgets.QLineEdit("port0/line0")
        self.trigger_rate_spin = QtWidgets.QDoubleSpinBox()
        self.trigger_rate_spin.setRange(0.1, 200000.0)
        self.trigger_rate_spin.setDecimals(3)
        self.trigger_rate_spin.setValue(10000.0)
        self.trigger_count_spin = QtWidgets.QSpinBox()
        self.trigger_count_spin.setRange(1, 100000000)
        self.trigger_count_spin.setValue(1000)
        self.frame_count_spin = QtWidgets.QSpinBox()
        self.frame_count_spin.setRange(1, 100000)
        self.frame_count_spin.setValue(64)
        self.timeout_spin = QtWidgets.QSpinBox()
        self.timeout_spin.setRange(-1, 300000)
        self.timeout_spin.setValue(-1)
        self.configure_trigger_button = QtWidgets.QPushButton("配置触发")
        self.start_button = QtWidgets.QPushButton("开始采集")
        self.stop_button = QtWidgets.QPushButton("停止采集")
        self.artdaq_once_button = QtWidgets.QPushButton("输出阿尔泰触发")
        trigger_form.addRow("采集模式", self.acq_mode_combo)
        trigger_form.addRow("相机工作模式", self.working_mode_combo)
        trigger_form.addRow("积分触发源", self.integration_source_combo)
        trigger_form.addRow("编码器倍频", self.encoder_multiply_combo)
        trigger_form.addRow("触发选择", self.trigger_selector_combo)
        trigger_form.addRow("触发源", self.trigger_source_combo)
        trigger_form.addRow("触发边沿", self.trigger_activation_combo)
        trigger_form.addRow("阿尔泰设备", self.artdaq_device_edit)
        trigger_form.addRow("阿尔泰 DO 线", self.artdaq_line_edit)
        trigger_form.addRow("阿尔泰触发频率 Hz", self.trigger_rate_spin)
        trigger_form.addRow("阿尔泰触发个数", self.trigger_count_spin)
        trigger_form.addRow("帧缓冲数", self.frame_count_spin)
        trigger_form.addRow("采集超时 ms", self.timeout_spin)
        trigger_form.addRow(self.configure_trigger_button)
        trigger_form.addRow(self.start_button)
        trigger_form.addRow(self.artdaq_once_button)
        trigger_form.addRow(self.stop_button)
        controls_layout.addWidget(trigger_box)

        save_box = QtWidgets.QGroupBox("保存")
        save_layout = QtWidgets.QVBoxLayout(save_box)
        self.save_npy_button = QtWidgets.QPushButton("保存当前帧 .npy")
        self.save_png_button = QtWidgets.QPushButton("保存显示图 .png")
        save_layout.addWidget(self.save_npy_button)
        save_layout.addWidget(self.save_png_button)
        controls_layout.addWidget(save_box)
        controls_layout.addStretch(1)

        self.image_label = QtWidgets.QLabel()
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.image_label.setMinimumSize(700, 420)
        self.image_label.setStyleSheet("background:#111; color:#ddd;")
        self.image_label.setText("暂无图像")
        display_layout.addWidget(self.image_label, 1)

        stats_layout = QtWidgets.QHBoxLayout()
        self.frame_count_label = QtWidgets.QLabel("帧数: 0")
        self.shape_label = QtWidgets.QLabel("尺寸: -")
        self.value_label = QtWidgets.QLabel("最小/最大: -")
        stats_layout.addWidget(self.frame_count_label)
        stats_layout.addWidget(self.shape_label)
        stats_layout.addWidget(self.value_label)
        stats_layout.addStretch(1)
        display_layout.addLayout(stats_layout)

        self.log_edit = QtWidgets.QPlainTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setMaximumHeight(190)
        display_layout.addWidget(self.log_edit)
        self.statusBar().showMessage("就绪")

    def _connect_signals(self):
        self.vlcf_button.clicked.connect(self.choose_vlcf)
        self.open_button.clicked.connect(self.open_camera)
        self.close_button.clicked.connect(self.close_camera)
        self.apply_camera_button.clicked.connect(self.apply_camera_settings)
        self.readback_button.clicked.connect(self.readback_camera_settings)
        self.configure_trigger_button.clicked.connect(self.configure_trigger)
        self.start_button.clicked.connect(self.start_grab)
        self.stop_button.clicked.connect(self.stop_grab)
        self.artdaq_once_button.clicked.connect(self.output_artdaq_trigger)
        self.camera_tap_combo.currentTextChanged.connect(self.sync_board_tap_to_camera_tap)
        self.save_npy_button.clicked.connect(self.save_npy)
        self.save_png_button.clicked.connect(self.save_png)

    def log(self, message):
        stamp = time.strftime("%H:%M:%S")
        text = "[%s] %s" % (stamp, message)
        self.log_edit.appendPlainText(text)
        self.statusBar().showMessage(message)
        print(text)

    def choose_vlcf(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "选择 IKap VLCF 配置文件",
            self.last_saved_dir,
            "IKap config (*.vlcf);;All files (*)",
        )
        if path:
            self.vlcf_edit.setText(path)
            self.last_saved_dir = os.path.dirname(path)
            ccf = matching_ccf_path(path)
            self.log(
                "将同时使用同名相机配置 .ccf: %s"
                % (os.path.basename(ccf) if ccf else "（未找到）")
            )

    def ensure_ikap_initialized(self):
        if self.ikap_initialized:
            return
        res = IKapC.ItkManInitialize()
        check_itk(res, "ItkManInitialize")
        self.ikap_initialized = True
        self.log("IKap 已初始化。")

    def require_grabber(self):
        if self.grabber is None or not self.grabber.is_board_open():
            raise RuntimeError("Camera/board is not open.")
        return self.grabber

    def sync_board_tap_to_camera_tap(self, text):
        if self.board_tap_combo.findText(str(text)) >= 0:
            self.board_tap_combo.setCurrentText(str(text))

    def selected_camera_taps(self):
        return int(self.camera_tap_combo.currentText())

    def selected_board_taps(self):
        return int(self.board_tap_combo.currentText())

    def matching_tap_vlcf_path(self):
        file_name = AK_TAP_VLCF_FILES.get(
            (self.pixel_format_combo.currentText(), self.selected_camera_taps())
        )
        if not file_name:
            return ""
        path = os.path.join(AK_CONFIG_DIR, file_name)
        return path if os.path.isfile(path) else ""

    def load_matching_tap_vlcf(self, grabber):
        if not self.auto_vlcf_checkbox.isChecked():
            return False
        path = self.matching_tap_vlcf_path()
        if not path:
            self.log(
                "No matching tap VLCF preset for %s %d-tap; using current board arrangement."
                % (self.pixel_format_combo.currentText(), self.selected_camera_taps())
            )
            return False
        grabber.load_configuration(path)
        self.vlcf_edit.setText(path)
        self.log("Loaded tap VLCF preset: %s" % os.path.basename(path))
        self.load_matching_ccf(grabber, path)
        return True

    def load_matching_ccf(self, grabber, vlcf_path=None):
        """加载 .vlcf 之后同时加载同目录同名的 .ccf（相机侧参数以它为准）。

        与主程序 ThreadCamera_AK.LoadCameraConfigurationFile() 同一语义：
        .ccf 里与 .vlcf 冲突的 ROI/位深/tap 等随后仍由板卡/相机设置按 .vlcf 覆盖。
        """
        if grabber is None or not grabber.is_camera_open():
            self.log("相机控制句柄未打开，跳过同名 .ccf（仅板卡模式）。")
            return False
        source = (vlcf_path or self.vlcf_edit.text().strip())
        path = matching_ccf_path(source)
        if not path:
            self.log(
                "未找到与 %s 同名的 .ccf，相机触发保持当前设置。"
                % (os.path.basename(source) if source else "（未选 .vlcf）")
            )
            return False
        res = IKapC.ItkDevLoadConfigurationFromFile(grabber.h_camera, path.encode("utf-8"))
        if res != IKapCDef.ITKSTATUS_OK:
            self.log(
                "ItkDevLoadConfigurationFromFile 失败: %#x (%s)"
                % (res, describe_itk_status(res))
            )
            return False
        self.loaded_ccf_path = path
        self.log("已加载同名 .ccf: %s" % os.path.basename(path))
        return True

    def set_camera_sensor_taps(self, grabber):
        if not grabber.is_camera_open():
            self.log("Camera control handle is not open; skipped SensorTaps.")
            return False
        taps = self.selected_camera_taps()
        for value in CAMERA_SENSOR_TAP_VALUES.get(taps, (str(taps),)):
            if set_feature_enum_any(
                grabber.h_camera,
                ("SensorTaps", "Sensor Taps", "SensorTap", "Sensor Tap"),
                value,
                required=False,
            ):
                return True
        self.log("Set SensorTaps=%d failed." % taps)
        return False

    def set_board_tap_number(self, grabber):
        taps = self.selected_board_taps()
        return set_board_int(
            grabber.h_board,
            "IKP_TAP_NUMBER",
            IKapBoard.IKP_TAP_NUMBER,
            taps,
            required=False,
        )

    def apply_tap_settings(self, grabber):
        self.load_matching_tap_vlcf(grabber)
        self.set_board_tap_number(grabber)
        self.set_camera_sensor_taps(grabber)
        if self.pixel_format_combo.currentText() in ("10-bit", "12-bit") and self.selected_camera_taps() == 10:
            self.log("Warning: 10/12-bit modes usually cannot use 10 taps; use 8 taps for 200 kHz tests.")

    def open_camera(self):
        try:
            self.ensure_ikap_initialized()
            if self.grabber is not None:
                self.close_camera()
                self.ensure_ikap_initialized()
            self.grabber = IKapLineScanGrabber()
            self.grabber.open_first_cameralink()
            self.grabber.load_configuration(self.vlcf_edit.text().strip())
            self.load_matching_ccf(self.grabber, self.vlcf_edit.text().strip())
            self.log("已打开 IKap 相机/采集卡。相机参数控制=%s" % self.grabber.is_camera_open())
            self.apply_camera_settings()
            self.configure_trigger()
        except Exception as error:
            self.log("打开失败: %s" % error)

    def close_camera(self):
        try:
            if self.grabber is not None:
                self.grabber.stop_grab()
                self.grabber.close()
                self.grabber = None
            if self.ikap_initialized:
                IKapC.ItkManTerminate()
                self.ikap_initialized = False
            self.log("已关闭 IKap 相机/采集卡。")
        except Exception as error:
            self.log("关闭失败: %s" % error)

    def apply_camera_settings(self):
        try:
            grabber = self.require_grabber()
            self.apply_tap_settings(grabber)
            # IKapCamera_test.configure_camera() 现在不再写板卡几何（改由 .vlcf 决定），
            # 这里按 UI 原有行为显式设置板卡几何尺寸。
            grabber.configure_board_image(
                width=self.width_spin.value(),
                height=self.height_spin.value(),
                offset_x=0,
            )
            grabber.configure_board_data_format(self.pixel_format_combo.currentText())
            grabber.configure_camera(
                width=self.width_spin.value(),
                height=self.height_spin.value(),
                exposure_us=self.exposure_spin.value(),
                line_rate_hz=self.line_rate_spin.value(),
                pixel_format=self.pixel_format_combo.currentText(),
            )
            if grabber.is_camera_open():
                set_feature_enum_any(
                    grabber.h_camera,
                    ("PGAGain", "PGA Gain"),
                    self.pga_gain_combo.currentText(),
                    required=False,
                )
                set_feature_float(
                    grabber.h_camera,
                    ("DigitalGain", "Digital Gain"),
                    self.digital_gain_spin.value(),
                    required=False,
                )
                set_feature_int(
                    grabber.h_camera,
                    ("DigitalOffset", "Digital Offset"),
                    self.digital_offset_spin.value(),
                    required=False,
                )
            self.log("相机参数已应用。")
        except Exception as error:
            self.log("应用相机参数失败: %s" % error)

    def readback_camera_settings(self):
        try:
            grabber = self.require_grabber()
            grabber.print_board_readback()
            if not grabber.is_camera_open():
                self.log("相机控制句柄未打开，无法读取相机参数。")
                return
            grabber.print_camera_readback()
            read_board_int(grabber.h_board, "IKP_TAP_NUMBER", IKapBoard.IKP_TAP_NUMBER)
            read_feature_enum(grabber.h_camera, ("SensorTaps", "Sensor Taps", "SensorTap", "Sensor Tap"))
            read_feature_enum(grabber.h_camera, ("PGAGain", "PGA Gain"))
            read_feature_float(grabber.h_camera, ("DigitalGain", "Digital Gain"))
            read_feature_int(grabber.h_camera, ("DigitalOffset", "Digital Offset"))
            self.log("相机实际参数已打印到控制台。")
        except Exception as error:
            self.log("读取实际参数失败: %s" % error)

    def acquisition_mode(self):
        return self.acq_mode_combo.currentData() or self.acq_mode_combo.currentText()

    def configure_trigger(self):
        try:
            grabber = self.require_grabber()
            mode = self.acquisition_mode()
            if mode == "FreeRun":
                self.configure_freerun(grabber)
            elif mode == "Software":
                self.configure_software_trigger(grabber)
            elif mode == "GeneralInput1_Hardware":
                self.configure_general_input_hardware_trigger(grabber)
            elif mode == "GeneralInput1_ShaftEncoder":
                self.configure_external_trigger(grabber, camera_working_mode="Hardware Trigger")
            else:
                self.configure_external_trigger(grabber, camera_working_mode="Free Run")
            self.log("触发已配置: %s" % self.acq_mode_combo.currentText())
        except Exception as error:
            self.log("配置触发失败: %s" % error)

    def configure_freerun(self, grabber):
        if grabber.is_camera_open():
            set_feature_enum_any(
                grabber.h_camera,
                ("SynchronizationMode", "Synchronization Mode"),
                "InternalFreeRun",
                required=False,
            )
            set_feature_enum_any(
                grabber.h_camera,
                ("WorkingMode", "Working Mode"),
                "Free Run",
                required=False,
            )
            set_feature_enum(grabber.h_camera, "TriggerSelector", "FrameStart", required=False)
            set_feature_enum(grabber.h_camera, "TriggerMode", "Off", required=False)
            set_feature_enum(grabber.h_camera, "TriggerSelector", "LineStart", required=False)
            set_feature_enum(grabber.h_camera, "TriggerMode", "Off", required=False)
        else:
            self.log("相机控制不可用，已跳过 FreeRun 相机参数设置。")

    def configure_software_trigger(self, grabber):
        if not grabber.is_camera_open():
            self.log("相机控制不可用，已跳过软件触发参数设置。")
            return
        set_feature_enum(grabber.h_camera, "TriggerSelector", self.trigger_selector_combo.currentText(), required=False)
        set_feature_enum(grabber.h_camera, "TriggerMode", "On", required=False)
        set_feature_enum(grabber.h_camera, "TriggerSource", "Software", required=False)
        set_feature_enum(grabber.h_camera, "TriggerActivation", self.trigger_activation_combo.currentText(), required=False)

    def configure_general_input_hardware_trigger(self, grabber):
        if grabber.is_camera_open():
            set_feature_enum_any(
                grabber.h_camera,
                ("SynchronizationMode", "Synchronization Mode"),
                "InternalFreeRun",
                required=False,
            )
            set_feature_enum_any(
                grabber.h_camera,
                ("WorkingMode", "Working Mode"),
                "Hardware Trigger",
                required=False,
            )
            set_feature_enum_any(
                grabber.h_camera,
                ("TriggerSource", "Trigger Source"),
                "General Input1",
                required=False,
            )
            set_feature_enum(grabber.h_camera, "TriggerSelector", "FrameStart", required=False)
            set_feature_enum(grabber.h_camera, "TriggerMode", "On", required=False)
            set_feature_enum(grabber.h_camera, "TriggerActivation", self.trigger_activation_combo.currentText(), required=False)
        else:
            self.log("相机控制不可用，已跳过通用输入1硬触发相机参数设置。")

    def configure_external_trigger(self, grabber, camera_working_mode=None):
        working_mode = camera_working_mode or self.working_mode_combo.currentText()
        integration_source = self.integration_source_combo.currentText()
        if self.acquisition_mode() in ("LineTrigger_ARTDAQ", "GeneralInput1_ShaftEncoder"):
            integration_source = "ShaftEncoder1"
        # IKapCamera_test.configure_line_trigger() 现在只写相机侧（板卡侧交给 .vlcf），
        # 这里按 UI 原有行为显式写入板卡侧触发设置。
        grabber.configure_board_line_trigger(
            integration_source=integration_source,
            general_input_edge=self.trigger_activation_combo.currentText(),
            encoder_multiply=self.encoder_multiply_combo.currentText(),
        )
        grabber.configure_line_trigger(
            integration_source=integration_source,
            camera_working_mode=working_mode,
            camera_trigger_source=self.trigger_source_combo.currentText(),
            general_input_edge=self.trigger_activation_combo.currentText(),
            encoder_multiply=self.encoder_multiply_combo.currentText(),
        )

    def start_grab(self):
        try:
            grabber = self.require_grabber()
            if grabber.grab_started:
                self.log("采集已经在运行。")
                return
            grabber.start_grab(self.frame_count_spin.value(), timeout_ms=self.timeout_spin.value())
            self.last_displayed_count = -1
            self.log(
                "采集已开始。实际图像尺寸: width=%d, height=%d, depth=%d"
                % (grabber.width, grabber.height, grabber.depth)
            )
            if self.acquisition_mode() in ("LineTrigger_ARTDAQ", "GeneralInput1_ShaftEncoder"):
                self.output_artdaq_trigger()
        except Exception as error:
            self.log("开始采集失败: %s" % error)

    def stop_grab(self):
        try:
            if self.grabber is not None:
                self.grabber.stop_grab()
            self.log("采集已停止。")
        except Exception as error:
            self.log("停止采集失败: %s" % error)

    def output_artdaq_trigger(self):
        if self.trigger_worker is not None and self.trigger_worker.isRunning():
            self.log("阿尔泰触发正在输出。")
            return
        if self.grabber is None or not self.grabber.grab_started:
            self.log("请先开始采集，再输出阿尔泰触发。")
            return
        self.image_timing_start_s = None
        self.trigger_timing_start_s = None
        self.trigger_timing_base_frame = int(self.grabber.frame_ready_count)
        self.trigger_timing_last_logged_frame = int(self.grabber.frame_ready_count)
        self.trigger_timing_rate_hz = float(self.trigger_rate_spin.value())
        actual_height = int(getattr(self.grabber, "height", 0) or 0)
        self.trigger_timing_height = max(1, actual_height or int(self.height_spin.value()))
        self.trigger_timing_count = int(self.trigger_count_spin.value())
        self.last_displayed_count = self.trigger_timing_base_frame
        self.trigger_worker = TriggerWorker(
            device=self.artdaq_device_edit.text().strip(),
            line=self.artdaq_line_edit.text().strip(),
            rate_hz=self.trigger_rate_spin.value(),
            trigger_count=self.trigger_count_spin.value(),
        )
        self.trigger_worker.status.connect(self.log)
        self.trigger_worker.output_started.connect(self.on_trigger_output_started)
        self.trigger_worker.failed.connect(lambda msg: self.log("阿尔泰触发失败: %s" % msg))
        self.trigger_worker.finished_ok.connect(lambda: self.log("阿尔泰触发输出完成。"))
        self.trigger_worker.start()

    def on_trigger_output_started(self, start_s):
        self.image_timing_start_s = float(start_s)
        self.trigger_timing_start_s = float(start_s)
        expected_frames = self.trigger_timing_count // max(1, self.trigger_timing_height)
        partial_lines = self.trigger_timing_count % max(1, self.trigger_timing_height)
        per_frame_time = max(0, self.trigger_timing_height - 1) / max(1e-12, self.trigger_timing_rate_hz)
        self.log(
            "第一行触发开始。Height=%d, TriggerCount=%d, TriggerRate=%.3f Hz, "
            "预计完整帧数=%d, 不足一帧剩余行数=%d, 每帧第一行到最后一行理论时间=%.6f s"
            % (
                self.trigger_timing_height,
                self.trigger_timing_count,
                self.trigger_timing_rate_hz,
                expected_frames,
                partial_lines,
                per_frame_time,
            )
        )

    def refresh_display(self):
        grabber = self.grabber
        if grabber is None:
            return
        self.frame_count_label.setText(
            "帧数: %d, 丢帧: %d, 超时: %d"
            % (grabber.frame_ready_count, grabber.frame_lost_count, grabber.timeout_count)
        )
        frame = grabber.last_frame
        if frame is None:
            if grabber.grab_started:
                self.image_label.setText("等待触发 / 等待图像")
            return
        if grabber.frame_ready_count == self.last_displayed_count:
            return
        self.last_displayed_count = grabber.frame_ready_count
        self.log_frame_timing(int(grabber.frame_ready_count))
        self.shape_label.setText("尺寸: %s" % (frame.shape,))
        self.log(
            "实际收到帧 #%d: frame.shape=%s, board width=%d, board height=%d"
            % (
                int(grabber.frame_ready_count),
                str(frame.shape),
                int(getattr(grabber, "width", 0) or 0),
                int(getattr(grabber, "height", 0) or 0),
            )
        )
        self.value_label.setText("最小/最大: %s/%s" % (np.min(frame), np.max(frame)))
        image = frame_to_qimage(frame_for_display(frame))
        if image is None:
            return
        pixmap = QtGui.QPixmap.fromImage(image)
        self.image_label.setPixmap(
            pixmap.scaled(
                self.image_label.size(),
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.FastTransformation,
            )
        )

    def log_frame_timing(self, frame_ready_count):
        if self.trigger_timing_start_s is None:
            return
        rate_hz = max(1e-12, float(self.trigger_timing_rate_hz))
        height = max(1, int(self.trigger_timing_height))
        theoretical_line_span = max(0, height - 1) / rate_hz
        expected_frames = self.trigger_timing_count // height
        start_abs = max(self.trigger_timing_last_logged_frame + 1, self.trigger_timing_base_frame + 1)
        ready_times = list(getattr(self.grabber, "frame_ready_times", []) or [])
        for abs_frame in range(start_abs, frame_ready_count + 1):
            frame_index = abs_frame - self.trigger_timing_base_frame
            if expected_frames and frame_index > expected_frames:
                continue
            frame_first_line_s = self.trigger_timing_start_s + (frame_index - 1) * height / rate_hz
            if 0 < abs_frame <= len(ready_times):
                ready_elapsed = ready_times[abs_frame - 1] - frame_first_line_s
            else:
                ready_elapsed = time.perf_counter() - frame_first_line_s
            self.log(
                "第 %d 帧: 第一行到最后一行理论采集时间=%.6f s, "
                "第一行触发到FrameReady/显示=%.6f s"
                % (frame_index, theoretical_line_span, ready_elapsed)
            )
        self.trigger_timing_last_logged_frame = max(self.trigger_timing_last_logged_frame, frame_ready_count)
        if expected_frames and self.trigger_timing_last_logged_frame >= self.trigger_timing_base_frame + expected_frames:
            self.trigger_timing_start_s = None
            self.image_timing_start_s = None

    def save_npy(self):
        try:
            frame = None if self.grabber is None else self.grabber.last_frame
            if frame is None:
                self.log("没有可保存的图像。")
                return
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self,
                "保存 NumPy 原始帧",
                os.path.join(self.last_saved_dir, "ikap_frame.npy"),
                "NumPy (*.npy);;All files (*)",
            )
            if not path:
                return
            np.save(path, frame)
            self.last_saved_dir = os.path.dirname(path)
            self.log("已保存原始帧: %s" % path)
        except Exception as error:
            self.log("保存 .npy 失败: %s" % error)

    def save_png(self):
        try:
            frame = None if self.grabber is None else self.grabber.last_frame
            if frame is None:
                self.log("没有可保存的图像。")
                return
            image = frame_to_qimage(frame_for_display(frame))
            if image is None:
                self.log("无法将当前帧转换为显示图。")
                return
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self,
                "保存显示图",
                os.path.join(self.last_saved_dir, "ikap_frame.png"),
                "PNG (*.png);;All files (*)",
            )
            if not path:
                return
            image.save(path)
            self.last_saved_dir = os.path.dirname(path)
            self.log("已保存显示图: %s" % path)
        except Exception as error:
            self.log("保存 .png 失败: %s" % error)

    def closeEvent(self, event):
        self.close_camera()
        super().closeEvent(event)


def main():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    window = IKapCameraWindow()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
