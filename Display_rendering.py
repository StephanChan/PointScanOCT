# -*- coding: utf-8 -*-
"""Display and overlay rendering helpers for the OCT control UI."""

import numpy as np
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt

from Generaic_functions import RGBImagePlot, fastLinePlot
from UsbCamera import stage_to_usb_image


def display_array(array):
    if isinstance(array, np.ndarray) and array.dtype.kind == "c":
        return np.abs(array)
    return array


def rgb_pixmap(rgb):
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"RGB image must have shape (height, width, 3), got {rgb.shape}")
    rgb = np.ascontiguousarray(np.clip(rgb, 0, 255).astype(np.uint8, copy=False))
    from PyQt5.QtGui import QImage
    height, width, _ = rgb.shape
    qimage = QImage(rgb.data, width, height, 3 * width, QImage.Format_RGB888).copy()
    return QPixmap.fromImage(qimage)


def z_depth_index(ui, z_pixels):
    if z_pixels <= 0:
        return 0
    if not hasattr(ui, "ZDepthBar"):
        return 0
    return max(0, min(int(ui.ZDepthBar.value()), int(z_pixels) - 1))


def bline_index(ui, y_pixels):
    if y_pixels <= 0:
        return 0
    if not hasattr(ui, "BlineBar"):
        return y_pixels // 2
    return max(0, min(int(ui.BlineBar.value()), int(y_pixels) - 1))


def update_bline_bar_limits(ui, y_pixels):
    if y_pixels <= 0 or not hasattr(ui, "BlineBar"):
        return
    max_index = int(y_pixels) - 1
    if hasattr(ui.BlineBar, "setMinimum"):
        ui.BlineBar.setMinimum(0)
    if hasattr(ui.BlineBar, "setMaximum"):
        ui.BlineBar.setMaximum(max_index)
    if ui.BlineBar.value() > max_index:
        ui.BlineBar.setValue(max_index)


def z_plane_from_volume(ui, volume):
    if volume is None or np.size(volume) == 0:
        return None
    volume = np.asarray(volume)
    if volume.ndim == 3:
        return volume[:, :, z_depth_index(ui, volume.shape[2])]
    raise ValueError(f"XY volume must have shape (Y, X, Z), got {volume.shape}")


def render_xz_pixmap(ui, intensity):
    intensity = display_array(intensity)
    ym = ui.XZmin.value()
    yM = ui.XZmax.value()
    return RGBImagePlot(matrix1=intensity, m=ym, M=yM)


def _ui_number(ui, name, default=1.0):
    """读 UI 上的数值：既支持控件（有 value()），也支持直接挂在 ui 上的数字。"""
    value = getattr(ui, name, None)
    if value is None:
        return default
    if hasattr(value, "value"):
        try:
            return float(value.value())
        except Exception:
            return default
    try:
        return float(value)
    except Exception:
        return default


def xz_pixel_sizes_um(ui):
    """XZ 显示两轴的物理尺度 (µm/px)。

    横向 = XStepSize：B-scan 在显示前已经把 AlineAVG 个 A-line 折叠成一列
    （``ThreadDnS.Process_bline``），所以每列正好对应一个振镜位置，步距就是 UI 的
    XStepSize，不能再乘 AlineAVG；轴向 = ``ui.axial_pixel_size_um``（水中 4.96 /
    空气 6.6，由标尺 checkbox 决定）。
    """
    x_um = _ui_number(ui, "XStepSize", 1.0)
    z_um = _ui_number(ui, "axial_pixel_size_um", 4.0)
    if x_um <= 0:
        x_um = 1.0
    if z_um <= 0:
        z_um = 1.0
    return x_um, z_um


def xy_pixel_sizes_um(ui):
    """en-face (Y, X) 显示两轴的物理尺度 (µm/px)。

    横 = XStepSize、竖 = YStepSize：``Process_Cscan`` 已把 AlineAVG（列）与
    BlineAVG（行内重复）折叠掉，所以每列/每行对应一个振镜位置，直接用 UI 步距。
    """
    x_um = _ui_number(ui, "XStepSize", 1.0)
    y_um = _ui_number(ui, "YStepSize", 1.0)
    if x_um <= 0:
        x_um = 1.0
    if y_um <= 0:
        y_um = 1.0
    return x_um, y_um


def _geometry_for(ui, pixmap, um_per_col, um_per_row, bottom_caption, left_caption):
    width = float(pixmap.width())
    height = float(pixmap.height())
    return {
        "um_per_col": float(um_per_col),
        "um_per_row": float(um_per_row),
        "bottom_axis": {"caption": bottom_caption, "origin": "left"},
        "left_axis": {"caption": left_caption, "origin": "top"},
        "bottom_total_um": width * float(um_per_col),
        "left_total_um": height * float(um_per_row),
    }


def set_xzplane_with_aspect(ui, pixmap, label=None, x_pixel_um=None, z_pixel_um=None):
    """把 (Z, X) 的 B-scan 图交给 XZplane 显示，并带上物理标尺几何。

    数组行 = 深度 Z（µm = z_um，原点在顶部），列 = 横向 X（µm = x_um）。
    控件若是 PanZoomImageView（有 set_data）则按物理纵横比显示并画尺；
    否则退回旧的 ``setPixmap``（兼容未启用标尺的情形）。
    """
    if label is None:
        label = getattr(ui, "XZplane", None)
    if label is None or pixmap is None or pixmap.isNull():
        return
    try:
        default_x_um, default_z_um = xz_pixel_sizes_um(ui)
        x_um = default_x_um if x_pixel_um is None else float(x_pixel_um)
        z_um = default_z_um if z_pixel_um is None else float(z_pixel_um)
        if x_um <= 0 or z_um <= 0:
            label.setPixmap(pixmap)
            return
        geometry = _geometry_for(ui, pixmap, x_um, z_um, "lateral X", "depth")
        if hasattr(label, "set_data"):
            label.set_data(pixmap, geometry)
        else:
            label.setPixmap(pixmap)
    except Exception as error:
        print("XZplane aspect display failed: %s" % error)
        try:
            label.setPixmap(pixmap)
        except Exception:
            pass


def set_xyplane_with_aspect(ui, pixmap, x_pixel_um=None, y_pixel_um=None):
    """把 (Y, X) 的 en-face 图交给 XYplane 显示（横=X、竖=Y，保持本项目朝向）。"""
    label = getattr(ui, "XYplane", None)
    if label is None or pixmap is None or pixmap.isNull():
        return
    try:
        default_x_um, default_y_um = xy_pixel_sizes_um(ui)
        x_um = default_x_um if x_pixel_um is None else float(x_pixel_um)
        y_um = default_y_um if y_pixel_um is None else float(y_pixel_um)
        if x_um <= 0 or y_um <= 0:
            label.setPixmap(pixmap)
            return
        geometry = _geometry_for(ui, pixmap, x_um, y_um, "lateral X", "scan Y")
        if hasattr(label, "set_data"):
            label.set_data(pixmap, geometry)
        else:
            label.setPixmap(pixmap.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
    except Exception as error:
        print("XYplane aspect display failed: %s" % error)
        try:
            label.setPixmap(pixmap)
        except Exception:
            pass


def set_xy_projection(ui, intensity=None, volume=None):
    if not hasattr(ui, "XYplane"):
        return
    if intensity is None and volume is None:
        return
    volume_plane = z_plane_from_volume(ui, volume)
    if volume_plane is not None:
        intensity = volume_plane
    pixmap = RGBImagePlot(matrix1=display_array(intensity), m=ui.XZmin.value(), M=ui.XZmax.value())
    set_xyplane_with_aspect(ui, pixmap)


def waveform_plot_size(label, default=(320, 220)):
    """绘图尺寸：优先用控件当前尺寸，控件还没布局时回退到默认值。"""
    width = int(getattr(label, "width", lambda: 0)() or 0)
    height = int(getattr(label, "height", lambda: 0)() or 0)
    if width < 60 or height < 60:
        return default
    return width, height


def describe_waveform(ao_waveform, do_waveform=None):
    """波形摘要（放到 tooltip）：X/Y 电压范围、DO 触发数、总采样数。"""
    ao = np.asarray(ao_waveform)
    lines = []
    if ao.ndim == 2 and ao.shape[0] >= 1:
        x_wave = np.asarray(ao[0], dtype=np.float64)
        lines.append("X: %.4f .. %.4f V" % (float(x_wave.min()), float(x_wave.max())))
        if ao.shape[0] >= 2:
            y_wave = np.asarray(ao[1], dtype=np.float64)
            lines.append("Y: %.4f .. %.4f V" % (float(y_wave.min()), float(y_wave.max())))
    elif ao.ndim == 1 and ao.size:
        lines.append("AO: %.4f .. %.4f V" % (float(ao.min()), float(ao.max())))
    if do_waveform is not None and np.size(do_waveform) > 0:
        lines.append("DO triggers: %d" % int(np.count_nonzero(np.asarray(do_waveform))))
    lines.append("samples: %d" % int(ao.shape[-1]) if ao.size else "samples: 0")
    return "\n".join(lines)



def render_aodo_waveform_ready(ui, payload):
    if hasattr(ui, "XwaveformLabel"):
        # 绘制 AODO 线程 ConfigTask 回传的真实波形（X/Y 扫描 + DO 相机触发）
        label = ui.XwaveformLabel
        payload = payload or {}
        ao_waveform = payload.get("ao_waveform", None)
        do_waveform = payload.get("do_waveform", None)
        if ao_waveform is None or np.size(ao_waveform) <= 0:
            label.clear()
            label.setText("no waveform")
            return
        ao_waveform = np.asarray(ao_waveform)
        # 只画一条 B-line（几千点）：整卷 C-scan 波形有几十万点，画全会卡住界面。
        full_samples = int(ao_waveform.shape[-1])
        line_samples = int(payload.get("line_samples", 0) or 0)
        shown_samples = full_samples
        if 0 < line_samples < full_samples:
            ao_waveform = ao_waveform[:, :line_samples]
            shown_samples = int(ao_waveform.shape[-1])
            if do_waveform is not None and np.size(do_waveform) > line_samples:
                do_waveform = np.asarray(do_waveform)[:line_samples]
        width, height = waveform_plot_size(label)
        label.setPixmap(
            fastLinePlot(ao_waveform, do_waveform, width=width, height=height, m=None, M=None,
                         y_axis=True, y_unit="V")
        )
        label.setToolTip(
            "%s\n%s\nB-line window: %d of %d samples"
            % (payload.get("mode", ""), describe_waveform(ao_waveform, do_waveform),
               shown_samples, full_samples)
        )


def render_aline_ready(ui, payload):
    aline = payload.get("aline", None)
    if aline is None:
        return
    aline = display_array(aline)
    ym = ui.XZmin.value()
    yM = ui.XZmax.value()
    pixmap = fastLinePlot(aline, width=ui.XZplane.width(), height=ui.XZplane.height(), m=ym, M=yM)
    ui.XZplane.setPixmap(pixmap)


def render_bline_ready(ui, payload):
    bline = payload.get("bline", None)
    if bline is None:
        return
    set_xzplane_with_aspect(ui, render_xz_pixmap(ui, bline))


def render_cscan_ready(ui, payload, follow_latest_bline=True):
    bline = payload.get("bline", None)
    aip = payload.get("aip", None)
    volume = payload.get("volume", None)

    if volume is not None and np.size(volume) > 0:
        volume = display_array(np.asarray(volume))
        if volume.ndim == 3:
            update_bline_bar_limits(ui, volume.shape[0])
            latest_bline_index = payload.get("latest_bline_index", None)
            if follow_latest_bline and latest_bline_index is not None and hasattr(ui, "BlineBar"):
                ui.BlineBar.setValue(max(0, min(int(latest_bline_index), int(volume.shape[0]) - 1)))
            bline = np.transpose(volume[bline_index(ui, volume.shape[0]), :, :]).copy()

    if bline is not None:
        set_xzplane_with_aspect(ui, render_xz_pixmap(ui, bline))

    set_xy_projection(ui, aip, volume)



