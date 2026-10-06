# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 19:41:46 2023

@author: admin
"""
# DO configure: port0 line 0 for X stage, port0 line 1 for Y stage, port 0 line 2 for Z stage, port 0 line 3 for Digitizer enable

# Generating Galvo X direction waveforms based on step size, Xsteps, Aline averages and objective
# StepSize in unit of um
# bias in unit of mm

import numpy as np
import os
import sys
import threading
from HardwareSpecs import get_objective_spec
from PyQt5.QtGui import QPixmap, QImage
from matplotlib import pyplot as plt
from PyQt5.QtGui import QPixmap, QPainter, QPen, QColor
from PyQt5.QtCore import Qt, QPointF, QRectF

# 线图最多绘制的点数：整卷 C-scan 波形可达几十万点，直接画会卡住界面，超出即抽稀。
LINE_PLOT_MAX_POINTS = 20000


class _LogTeeStream:
    def __init__(self, logger, original_stream):
        self.logger = logger
        self.original_stream = original_stream

    def write(self, text):
        if self.original_stream is not None:
            self.original_stream.write(text)
        self.logger.write_stream(text)
        return len(text)

    def flush(self):
        if self.original_stream is not None:
            self.original_stream.flush()


class LOG():
    def __init__(self, ui):
        super().__init__()
        import datetime
        current_time = datetime.datetime.now()
        self.ui = ui
        suffix = (
            str(current_time.year)+"-"+
            str(current_time.month)+"-"+
            str(current_time.day)+"-"+
            str(current_time.hour)+"-"+
            str(current_time.minute)+"-"+
            str(current_time.second)
        )
        self._suffix = suffix
        self._lock = threading.Lock()
        self._stdout_installed = False
        self._stdout = None
        self._stderr = None
        self._suppress_next_newline = False
        self._suppressed_stream_fragments = (
            "waiting for camera data...",
            "time to fetch data:",
        )

    def _current_log_dir(self):
        base_dir = os.getcwd()
        try:
            candidate_dir = self.ui.DIR.toPlainText().strip()
            if candidate_dir:
                base_dir = candidate_dir
        except Exception:
            pass
        log_dir = os.path.join(base_dir, "log_files")
        return log_dir

    def _current_file_path(self):
        return os.path.join(self._current_log_dir(), "log_" + self._suffix + ".txt")

    def _append_text(self, path, text):
        if text is None or text == "":
            return
        with self._lock:
            try:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "a", encoding="utf-8") as fp:
                    fp.write(text)
            except OSError:
                fallback_dir = os.path.join(os.getcwd(), "log_files")
                os.makedirs(fallback_dir, exist_ok=True)
                fallback_path = os.path.join(fallback_dir, os.path.basename(path))
                with open(fallback_path, "a", encoding="utf-8") as fp:
                    fp.write(text)

    def write(self, message):
        self._append_text(self._current_file_path(), str(message) + "\n")

    def write_stream(self, text):
        if text is None:
            return
        text_str = str(text)
        if self._suppress_next_newline and text_str in ("\n", "\r\n", "\r"):
            self._suppress_next_newline = False
            return
        self._suppress_next_newline = False
        for fragment in self._suppressed_stream_fragments:
            if fragment in text_str:
                self._suppress_next_newline = True
                return
        self._append_text(self._current_file_path(), text_str)

    def install_stream_redirects(self):
        if self._stdout_installed:
            return
        self._stdout = sys.stdout
        self._stderr = sys.stderr
        sys.stdout = _LogTeeStream(self, self._stdout)
        sys.stderr = _LogTeeStream(self, self._stderr)
        self._stdout_installed = True


def GenGalvoWave(StepSize=1, Steps=1000, AVG=1, obj="5X", postclocks=50, Galvo_bias=0):
    objective = get_objective_spec(obj)
    if objective is None:
        status = "objective not calibrated, abort generating Galvo waveform"
        return None, status
    angle2mmratio = objective.angle_to_mm_ratio
    Xrange = StepSize * Steps / 1000
    Vmax = (Xrange / 2) / angle2mmratio / 2 + Galvo_bias
    Vmin = (-Xrange / 2) / angle2mmratio / 2 + Galvo_bias
    steps2 = postclocks
    waveform = np.linspace(Vmax, Vmin, Steps)
    waveform = np.tile(waveform, (AVG, 1)).transpose().flatten()
    Postwave = (Vmin - Vmax) / 2 * np.cos(np.arange(0, np.pi, np.pi / steps2)) + (Vmax + Vmin) / 2
    prewave = np.ones(50) * waveform[0]
    waveform = np.append(np.append(prewave, waveform), Postwave)
    status = "waveform updated"
    return waveform, status


def _scan_voltage_limits(length_mm, obj="5X", bias=0.0):
    objective = get_objective_spec(obj)
    if objective is None:
        raise ValueError("objective not calibrated, abort generating Galvo waveform")
    angle2mmratio = objective.angle_to_mm_ratio
    vmax = (length_mm / 2.0) / angle2mmratio / 2.0 + bias
    vmin = (-length_mm / 2.0) / angle2mmratio / 2.0 + bias
    return vmin, vmax


def _camera_trigger_train(trigger_count):
    trigger_count = max(1, int(trigger_count))
    waveform = np.zeros([trigger_count, 2], dtype=np.uint32)
    waveform[:, 1] = 1
    return waveform.flatten()


def _cos_flyback(v_from, v_to, clocks):
    clocks = max(1, int(clocks))
    theta = np.linspace(0.0, np.pi, clocks, endpoint=False)
    return (v_from - v_to) / 2.0 * np.cos(theta) + (v_from + v_to) / 2.0


def GenAODO(
    mode="ContinuousBline",
    obj="5X",
    postclocks=50,
    YStepSize=1,
    YSteps=200,
    BVG=1,
    BlineRepeat=1,
    Galvo_bias=0,
    XStepSize=1,
    XSteps=200,
    GalvoY_bias=None,
):
    """Generate two-axis spot-scan galvo AO plus camera DO trigger.

    AOwaveform is returned as [2, samples]: channel 0 is X fast axis and
    channel 1 is Y slow axis. The DO waveform triggers the line-scan camera
    only during valid X forward scans.
    """
    preclocks = 0
    postclocks = max(1, int(postclocks))
    XSteps = max(1, int(XSteps))
    YSteps = max(1, int(YSteps))
    BVG = max(1, int(BVG))
    BlineRepeat = max(1, int(BlineRepeat))
    y_bias = Galvo_bias if GalvoY_bias is None else GalvoY_bias

    x_length_mm = float(XStepSize) * XSteps / 1000.0
    y_length_mm = float(YStepSize) * YSteps / 1000.0
    x_min, x_max = _scan_voltage_limits(x_length_mm, obj=obj, bias=Galvo_bias)
    y_min, y_max = _scan_voltage_limits(y_length_mm, obj=obj, bias=y_bias)

    x_forward = np.repeat(np.linspace(x_min, x_max, XSteps), 2 * BVG)
    do_forward = _camera_trigger_train(XSteps * BVG)
    x_pre = np.ones(preclocks) * x_min
    x_back = _cos_flyback(x_max, x_min, postclocks)

    if mode in ["ContinuousBline", "ContinueBline", "FiniteBscan", "Xfrequency"]:
        x_segments = []
        y_segments = []
        do_segments = []
        for _repeat in range(BlineRepeat):
            x_segments.extend([x_pre, x_forward, x_back])
            y_segments.extend([
                np.ones(preclocks) * y_bias,
                np.ones_like(x_forward) * y_bias,
                np.ones(postclocks) * y_bias,
            ])
            do_segments.extend([
                np.zeros(preclocks, dtype=np.uint32),
                do_forward,
                np.zeros(postclocks, dtype=np.uint32),
            ])
        x_wave = np.concatenate(x_segments)
        y_wave = np.concatenate(y_segments)
        do_wave = np.concatenate(do_segments)
        return np.uint32(do_wave), np.vstack([x_wave, y_wave]), "waveform updated"

    elif mode in ["FiniteCscan", "ContinuousCscan"]:
        y_positions = np.linspace(y_min, y_max, YSteps)
        x_segments = []
        y_segments = []
        do_segments = []
        for row, y_pos in enumerate(y_positions):
            next_y = y_positions[row + 1] if row + 1 < YSteps else y_positions[0]
            for repeat in range(BlineRepeat):
                is_last_repeat = repeat == BlineRepeat - 1
                flyback_y_to = next_y if is_last_repeat else y_pos
                x_segments.append(x_pre)
                y_segments.append(np.ones(preclocks) * y_pos)
                do_segments.append(np.zeros(preclocks, dtype=np.uint32))

                x_segments.append(x_forward)
                y_segments.append(np.ones_like(x_forward) * y_pos)
                do_segments.append(do_forward)

                x_segments.append(x_back)
                y_segments.append(_cos_flyback(y_pos, flyback_y_to, postclocks))
                do_segments.append(np.zeros(postclocks, dtype=np.uint32))

        x_wave = np.concatenate(x_segments)
        y_wave = np.concatenate(y_segments)
        do_wave = np.concatenate(do_segments)
        return np.uint32(do_wave), np.vstack([x_wave, y_wave]), "waveform updated"

    else:
        status = "invalid task type! Abort action"
        return None, None, status


def LinePlot(AOwaveform, DOwaveform=None, m=2, M=4):
    plt.cla()
    AOwaveform = np.asarray(AOwaveform)
    if AOwaveform.ndim == 2:
        AOwaveform = AOwaveform[0]
    if np.any(DOwaveform):
        plt.plot(range(len(DOwaveform)), DOwaveform, linewidth=2)
    plt.plot(range(len(AOwaveform)), AOwaveform, linewidth=2)
    plt.ylim([m, M])
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.rcParams["savefig.dpi"] = 150
    plt.savefig("lineplot.jpg")
    pixmap = QPixmap("lineplot.jpg")
    return pixmap


def draw_voltage_axis(painter, left, right, top, bottom, m, M, y_scale, unit="V"):
    """波形纵坐标：轴线 + 5 个电压刻度 + 水平网格线（供 fastLinePlot 的 y_axis=True 使用）。"""
    painter.setPen(QPen(QColor(120, 120, 120), 1))
    painter.drawLine(left, top, left, bottom)
    painter.drawLine(left, bottom, right, bottom)
    grid_pen = QPen(QColor(228, 228, 228), 1)
    tick_pen = QPen(QColor(70, 70, 70), 1)
    value_format = "%.2f" if abs(float(M) - float(m)) < 20.0 else "%.1f"
    for index in range(5):
        value = float(m) + (float(M) - float(m)) * index / 4.0
        y = bottom - (value - float(m)) * y_scale
        painter.setPen(grid_pen)
        painter.drawLine(left + 1, int(round(y)), right, int(round(y)))
        painter.setPen(tick_pen)
        painter.drawLine(left, int(round(y)), left + 4, int(round(y)))
        painter.drawText(
            QRectF(0.0, y - 7.0, float(left) - 6.0, 14.0),
            int(Qt.AlignRight | Qt.AlignVCenter),
            value_format % value,
        )
    if unit:
        painter.setPen(QPen(QColor(40, 40, 40), 1))
        painter.drawText(
            QRectF(0.0, float(top), float(left) - 6.0, 12.0),
            int(Qt.AlignRight | Qt.AlignTop),
            str(unit),
        )


def fastLinePlot(AOwaveform, DOwaveform=None, width=800, height=300, m=2, M=4,
                 y_axis=False, y_unit="V"):
    AOwaveform = np.asarray(AOwaveform)
    plot_series = []
    if AOwaveform.ndim == 1:
        plot_series.append(("X", AOwaveform, QColor(0, 0, 200)))
    elif AOwaveform.ndim == 2:
        labels = ("X", "Y")
        colors = (QColor(0, 0, 200), QColor(210, 80, 0))
        for index in range(AOwaveform.shape[0]):
            label = labels[index] if index < len(labels) else f"AO{index}"
            color = colors[index] if index < len(colors) else QColor(80, 80, 80)
            plot_series.append((label, AOwaveform[index], color))

    if DOwaveform is not None and np.size(DOwaveform) > 0:
        plot_series.append(("DO", np.asarray(DOwaveform), QColor(0, 150, 70)))

    if not plot_series:
        plot_series.append(("X", np.asarray([0.0]), QColor(0, 0, 200)))

    # 长波形（例如整卷 C-scan 的几十万点）按步长抽稀，避免画图卡住界面。
    limited_series = []
    for label, series, color in plot_series:
        series = np.asarray(series).ravel()
        if series.size > LINE_PLOT_MAX_POINTS:
            step = int(np.ceil(series.size / float(LINE_PLOT_MAX_POINTS)))
            series = series[::step]
        limited_series.append((label, series, color))
    plot_series = limited_series

    finite_values = [
        np.asarray(series, dtype=np.float32).ravel()
        for _label, series, _color in plot_series
        if np.size(series) > 0
    ]
    finite_values = [series[np.isfinite(series)] for series in finite_values]
    finite_values = [series for series in finite_values if series.size > 0]
    if m is None:
        m = min(float(np.min(series)) for series in finite_values) if finite_values else 0.0
    if M is None:
        M = max(float(np.max(series)) for series in finite_values) if finite_values else 1.0
    if M <= m:
        M = m + 1.0

    pixmap = QPixmap(int(width), int(height))
    pixmap.fill(Qt.white)

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    # y_axis=True 时左侧留出电压刻度文字的空间
    left = 52 if y_axis else 8
    right = max(left + 1, int(width) - 8)
    top = 8
    bottom = max(top + 1, int(height) - 18)
    plot_width = max(1, right - left)
    plot_height = max(1, bottom - top)
    y_scale = plot_height / float(M - m)

    if y_axis:
        draw_voltage_axis(painter, left, right, top, bottom, m, M, y_scale, y_unit)

    for label, series, color in plot_series:
        series = np.asarray(series, dtype=np.float32).ravel()
        if series.size <= 0:
            continue
        if series.size == 1:
            points = [QPointF(left, bottom - ((float(series[0]) - m) * y_scale))]
        else:
            x_scale = plot_width / float(series.size - 1)
            points = [
                QPointF(left + i * x_scale, bottom - ((float(series[i]) - m) * y_scale))
                for i in range(series.size)
            ]
        painter.setPen(QPen(color, 2))
        painter.drawPolyline(*points)

    legend_x = left + 4
    legend_y = top + 12
    for label, _series, color in plot_series[:4]:
        painter.setPen(QPen(color, 2))
        painter.drawText(legend_x, legend_y, label)
        legend_x += 32
    painter.end()

    return pixmap


def RGBImagePlot(matrix1=None, matrix2=None, m=0, M=1):
    if matrix1 is None:
        matrix1 = []
    if matrix2 is None:
        matrix2 = []
    if len(matrix2) > 0:
        scale = 1
        matrix2 = np.float32(np.array(matrix2))
        matrix2[matrix2 < m] = m
        matrix2[matrix2 > M] = M
        matrix2 = np.uint8((matrix2 - m + 0.01) / np.abs(M - m + 0.1) * 127)
    else:
        scale = 2
        matrix2 = np.zeros(np.array(matrix1).shape)

    matrix1 = np.float32(np.array(matrix1))
    matrix1[matrix1 < m] = m
    matrix1[matrix1 > M] = M
    matrix1 = np.uint8((matrix1 - m + 0.01) / np.abs(M - m + 0.1) * 127 * scale)

    height, width = matrix1.shape
    rgb_array = np.zeros((height, width, 3), dtype=np.uint8)
    rgb_array[..., 0] = matrix1 + matrix2
    rgb_array[..., 1] = matrix1
    rgb_array[..., 2] = matrix1

    bytes_per_line = 3 * width
    qimage = QImage(rgb_array.data, width, height, bytes_per_line, QImage.Format_RGB888)
    pixmap = QPixmap.fromImage(qimage)
    return pixmap


def _normalize_to_uint8(matrix, m=None, M=None):
    matrix = np.float32(np.array(matrix))
    if m is None:
        m = np.nanpercentile(matrix, 1)
    if M is None:
        M = np.nanpercentile(matrix, 99)
    if M <= m:
        M = m + 1e-5
    matrix = np.clip(matrix, m, M)
    return np.uint8((matrix - m) / (M - m + 1e-5) * 255)


def findchangept(signal, step):
    L = len(signal)
    z = np.argmax(signal)
    last = np.min([z + 30, L - 2])
    signal = signal[1:last]
    L = len(signal)
    residual_error = np.ones(L) * 9999999
    for ii in range(2, L - 2, step):
        residual_error[ii] = (ii - 1) * np.var(signal[0:ii]) + (L - ii + 1) * np.var(signal[ii + 1:L])
    pts = np.argmin(residual_error)
    return pts

