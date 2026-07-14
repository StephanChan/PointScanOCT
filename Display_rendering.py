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


def set_xy_projection(ui, intensity=None, volume=None):
    if not hasattr(ui, "XYplane"):
        return
    if intensity is None and volume is None:
        return
    volume_plane = z_plane_from_volume(ui, volume)
    if volume_plane is not None:
        intensity = volume_plane
    pixmap = RGBImagePlot(matrix1=display_array(intensity), m=ui.XZmin.value(), M=ui.XZmax.value())
    ui.XYplane.setPixmap(pixmap.scaled(ui.XYplane.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))


def render_aodo_waveform_ready(ui, payload):
    ao_waveform = payload.get("ao_waveform", None)
    do_waveform = payload.get("do_waveform", None)
    if ao_waveform is None or do_waveform is None:
        return
    ao_waveform = np.asarray(ao_waveform, dtype=np.float32)
    do_waveform = np.asarray(do_waveform, dtype=np.float32)
    wave_min = float(min(np.min(ao_waveform), np.min(do_waveform)))
    wave_max = float(max(np.max(ao_waveform), np.max(do_waveform)))
    if wave_max <= wave_min:
        margin = 1.0
    else:
        margin = 0.05 * (wave_max - wave_min)
    pixmap = fastLinePlot(
        ao_waveform,
        do_waveform,
        m=wave_min - margin,
        M=wave_max + margin,
    )
    ui.XwaveformLabel.setPixmap(pixmap)


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
    ui.XZplane.setPixmap(render_xz_pixmap(ui, bline))


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
        ui.XZplane.setPixmap(render_xz_pixmap(ui, bline))

    set_xy_projection(ui, aip, volume)



