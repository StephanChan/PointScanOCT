# -*- coding: utf-8 -*-
"""Display and overlay rendering helpers for the OCT control UI."""

import numpy as np
import cv2
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor
from PyQt5.QtCore import Qt, QRectF

from Generaic_functions import RGBImagePlot, fastLinePlot, LinePlot
from UsbCamera import USB_PIXEL_SIZE_MM, stage_to_usb_image


RGB_DYNAMIC_HUE_HZ_PER_CONTRAST_UNIT = 15.0 / 1000.0
RGB_DYNAMIC_SATURATION_BANDWIDTH_RANGE_HZ = (0.0, 8.0)
RGB_DYNAMIC_VALUE_DYNAMIC_RANGE = (0.0, 500.0)
RGB_DYNAMIC_VALUE_GAMMA = 1.0


def display_array(array):
    if isinstance(array, np.ndarray) and array.dtype.kind == 'c':
        return np.abs(array)
    return array


def rgb_pixmap(rgb):
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"RGB image must have shape (height, width, 3), got {rgb.shape}")
    rgb = np.ascontiguousarray(np.clip(rgb, 0, 255).astype(np.uint8, copy=False))
    height, width, _ = rgb.shape
    qimage = QImage(rgb.data, width, height, 3 * width, QImage.Format_RGB888).copy()
    return QPixmap.fromImage(qimage)


def rgb_display_limits(min_widget, max_widget):
    control_max = max(
        1,
        int(min_widget.maximum()) if hasattr(min_widget, "maximum") else 255,
        int(max_widget.maximum()) if hasattr(max_widget, "maximum") else 255,
    )
    scale = 255.0 / float(control_max)
    return float(min_widget.value()) * scale, float(max_widget.value()) * scale


def dynamic_rgb_display_array(ui, rgb, min_widget, max_widget):
    rgb = np.asarray(rgb, dtype=np.float32)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"RGB image must have shape (height, width, 3), got {rgb.shape}")
    m, M = rgb_display_limits(min_widget, max_widget)
    adjusted = (rgb - m) / (M - m + 1e-5) * 255.0
    if hasattr(ui, "DynContrast"):
        adjusted *= float(ui.DynContrast.value()) / 50.0
    return np.ascontiguousarray(np.clip(adjusted, 0, 255).astype(np.uint8))


def hue_frequency_range_from_controls(min_widget, max_widget):
    return (
        float(min_widget.value()) * RGB_DYNAMIC_HUE_HZ_PER_CONTRAST_UNIT,
        float(max_widget.value()) * RGB_DYNAMIC_HUE_HZ_PER_CONTRAST_UNIT,
    )


def normalize_to_unit_interval(image, value_range, gamma=1.0):
    low_value, high_value = float(value_range[0]), float(value_range[1])
    if high_value <= low_value:
        raise ValueError(f"Invalid display normalization range: {value_range}")
    normalized = (np.asarray(image, dtype=np.float32) - low_value) / (high_value - low_value)
    normalized = np.clip(normalized, 0.0, 1.0)
    gamma = float(gamma)
    if np.isfinite(gamma) and gamma > 0.0 and abs(gamma - 1.0) > 1e-6:
        normalized = normalized ** (1.0 / gamma)
    return normalized


def hsv_to_rgb_array(hue, saturation, value):
    hue = np.mod(np.asarray(hue, dtype=np.float32), 1.0)
    saturation = np.clip(np.asarray(saturation, dtype=np.float32), 0.0, 1.0)
    value = np.clip(np.asarray(value, dtype=np.float32), 0.0, 1.0)

    h6 = hue * 6.0
    i = np.floor(h6).astype(np.int32)
    f = h6 - i.astype(np.float32)
    p = value * (1.0 - saturation)
    q = value * (1.0 - saturation * f)
    t = value * (1.0 - saturation * (1.0 - f))
    i_mod = np.mod(i, 6)

    rgb = np.empty(hue.shape + (3,), dtype=np.float32)
    masks = [
        (i_mod == 0, value, t, p),
        (i_mod == 1, q, value, p),
        (i_mod == 2, p, value, t),
        (i_mod == 3, p, q, value),
        (i_mod == 4, t, p, value),
        (i_mod == 5, value, p, q),
    ]
    for mask, red, green, blue in masks:
        rgb[..., 0][mask] = red[mask]
        rgb[..., 1][mask] = green[mask]
        rgb[..., 2][mask] = blue[mask]
    return np.ascontiguousarray(np.clip(np.rint(rgb * 255.0), 0, 255).astype(np.uint8))


def dynamic_metric_rgb_display_array(ui, frequency_hz, bandwidth_hz, value, min_widget, max_widget):
    hue_range = hue_frequency_range_from_controls(min_widget, max_widget)
    hue = normalize_to_unit_interval(frequency_hz, hue_range)
    saturation = normalize_to_unit_interval(
        bandwidth_hz,
        RGB_DYNAMIC_SATURATION_BANDWIDTH_RANGE_HZ,
    )
    value = normalize_to_unit_interval(
        value,
        RGB_DYNAMIC_VALUE_DYNAMIC_RANGE,
        gamma=RGB_DYNAMIC_VALUE_GAMMA,
    )
    if hasattr(ui, "DynContrast"):
        value = np.clip(value * (float(ui.DynContrast.value()) / 50.0), 0.0, 1.0)
    return hsv_to_rgb_array(hue, saturation, value)


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
    if volume.ndim == 4 and volume.shape[-1] == 3:
        return volume[:, :, z_depth_index(ui, volume.shape[2]), :]
    raise ValueError(f"XY volume must have shape (Y, X, Z) or (Y, X, Z, 3), got {volume.shape}")


def render_xz_pixmap(ui, intensity, rgb=None, hsv=None, frequency_hz=None, bandwidth_hz=None, value=None):
    if hsv is not None and np.size(hsv) > 0:
        hsv = np.asarray(hsv, dtype=np.float32)
        return rgb_pixmap(dynamic_metric_rgb_display_array(ui, hsv[..., 0], hsv[..., 1], hsv[..., 2], ui.XZmin, ui.XZmax))
    if rgb is not None and np.size(rgb) > 0:
        if frequency_hz is not None and bandwidth_hz is not None and value is not None:
            return rgb_pixmap(dynamic_metric_rgb_display_array(ui, frequency_hz, bandwidth_hz, value, ui.XZmin, ui.XZmax))
        return rgb_pixmap(dynamic_rgb_display_array(ui, rgb, ui.XZmin, ui.XZmax))
    intensity = display_array(intensity)
    ym = ui.XZmin.value()
    yM = ui.XZmax.value()
    return RGBImagePlot(matrix1=intensity, m=ym, M=yM)


def set_xy_projection(
    ui,
    intensity,
    rgb=None,
    hsv=None,
    frequency_hz=None,
    bandwidth_hz=None,
    value=None,
    volume=None,
    hsv_volume=None,
):
    if not hasattr(ui, "XYplane"):
        return
    if intensity is None and volume is None:
        return
    volume_plane = z_plane_from_volume(ui, volume)
    if volume_plane is not None:
        intensity = volume_plane
    hsv_volume_plane = z_plane_from_volume(ui, hsv_volume)
    if hsv_volume_plane is not None:
        hsv = hsv_volume_plane
    if hsv is not None and np.size(hsv) > 0:
        hsv = np.asarray(hsv, dtype=np.float32)
        pixmap = rgb_pixmap(dynamic_metric_rgb_display_array(ui, hsv[..., 0], hsv[..., 1], hsv[..., 2], ui.XZmin, ui.XZmax))
    elif rgb is not None and np.size(rgb) > 0:
        if frequency_hz is not None and bandwidth_hz is not None and value is not None:
            pixmap = rgb_pixmap(dynamic_metric_rgb_display_array(ui, frequency_hz, bandwidth_hz, value, ui.XZmin, ui.XZmax))
        else:
            pixmap = rgb_pixmap(dynamic_rgb_display_array(ui, rgb, ui.XZmin, ui.XZmax))
    else:
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
    pixmap = LinePlot(
        ao_waveform,
        do_waveform,
        wave_min - margin,
        wave_max + margin,
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
    rgb = payload.get("rgb", None)
    ui.XZplane.setPixmap(
        render_xz_pixmap(
            ui,
            bline,
            rgb,
            payload.get("hsv", None),
            payload.get("freq", None),
            payload.get("bandwidth", None),
            payload.get("value", None),
        )
    )


def render_cscan_ready(ui, payload, follow_latest_bline=True):
    bline = payload.get("bline", None)
    rgbb = payload.get("rgbb", None)
    aip = payload.get("aip", None)
    rgb = payload.get("rgb", None)
    volume = payload.get("volume", None)

    if volume is not None and np.size(volume) > 0:
        volume = display_array(np.asarray(volume))
        if volume.ndim == 3:
            update_bline_bar_limits(ui, volume.shape[0])
            latest_bline_index = payload.get("latest_bline_index", None)
            if follow_latest_bline and latest_bline_index is not None and hasattr(ui, "BlineBar"):
                ui.BlineBar.setValue(max(0, min(int(latest_bline_index), int(volume.shape[0]) - 1)))
            bline = np.transpose(volume[bline_index(ui, volume.shape[0]), :, :]).copy()
            rgbb = None

    if bline is not None:
        ui.XZplane.setPixmap(
            render_xz_pixmap(
                ui,
                bline,
                rgbb,
                payload.get("hsvb", None),
                payload.get("freqb", None),
                payload.get("bandwidthb", None),
                payload.get("valueb", None),
            )
        )

    set_xy_projection(
        ui,
        aip,
        rgb,
        payload.get("hsv", None),
        payload.get("freq", None),
        payload.get("bandwidth", None),
        payload.get("value", None),
        volume,
        payload.get("hsv_volume", None),
    )


