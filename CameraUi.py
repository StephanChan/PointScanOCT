"""Camera UI helpers shared by acquisition, processing, and display code."""


CAMERA_DAHENG = "Daheng"
CAMERA_PHOTONFOCUS = "PhotonFocus"
CAMERA_HIK = "HiK"
CAMERA_HIK_ALIASES = {CAMERA_HIK, "XingTu", "HIK", "HK"}
CAMERA_AK = "AiKe"
CAMERA_AK_ALIASES = {CAMERA_AK, "AK", "Aike", "IKap", "I-TEK", "PL4KCL", "PL4KCL-200KT"}


def widget_value(ui, name, default=None):
    widget = getattr(ui, name, None)
    if widget is None:
        return default
    if hasattr(widget, "value"):
        return widget.value()
    if hasattr(widget, "currentText"):
        return widget.currentText()
    if hasattr(widget, "text"):
        return widget.text()
    return default


def x_pixel_count(ui):
    x_length = widget_value(ui, "XLength", None)
    x_step = widget_value(ui, "XStepSize", None)
    if x_length is not None and x_step is not None and float(x_step) > 0:
        positions = max(1, int(round(float(x_length) * 1000.0 / float(x_step))))
        aline_avg = max(1, int(widget_value(ui, "AlineAVG", 1)))
        return max(1, int(positions) * aline_avg)
    for name in ("XPixels", "Xpixels", "AlinesPerBline"):
        value = widget_value(ui, name, None)
        if value is not None:
            return max(1, int(value))
    return 1


def x_position_count(ui):
    x_length = widget_value(ui, "XLength", None)
    x_step = widget_value(ui, "XStepSize", None)
    if x_length is not None and x_step is not None and float(x_step) > 0:
        return max(1, int(round(float(x_length) * 1000.0 / float(x_step))))
    for name in ("XPixels", "Xpixels", "AlinesPerBline"):
        value = widget_value(ui, name, None)
        if value is not None:
            return max(1, int(value))
    return 1


def set_widget_value(ui, name_options, value):
    for name in name_options:
        widget = getattr(ui, name, None)
        if widget is not None and hasattr(widget, "setValue"):
            widget.setValue(value)
            return True
    return False


def current_camera_name(ui):
    return str(widget_value(ui, "Camera", ""))


def spectral_downsample(ui):
    camera = current_camera_name(ui)
    if camera == CAMERA_PHOTONFOCUS:
        return max(1, int(widget_value(ui, "SpectralDS_PF", 1)))
    if camera in CAMERA_HIK_ALIASES and hasattr(ui, "SpectralDS_HK"):
        return max(1, int(widget_value(ui, "SpectralDS_HK", 1)))
    if camera in CAMERA_AK_ALIASES and hasattr(ui, "SpectralDS_AK"):
        return max(1, int(widget_value(ui, "SpectralDS_AK", 1)))
    return max(1, int(widget_value(ui, "SpectralDS_DH", 1)))


def raw_camera_sample_count(ui):
    camera = current_camera_name(ui)
    if camera == CAMERA_PHOTONFOCUS:
        return int(widget_value(ui, "NSamples_PF", 1024))
    if camera in CAMERA_HIK_ALIASES and hasattr(ui, "NSamples_HK"):
        return int(widget_value(ui, "NSamples_HK", 1024))
    if camera in CAMERA_AK_ALIASES and hasattr(ui, "NSamples_AK"):
        return int(widget_value(ui, "NSamples_AK", 4096))
    return int(widget_value(ui, "NSamples_DH", 1024))


def effective_camera_sample_count(ui):
    raw_samples = raw_camera_sample_count(ui)
    ds = spectral_downsample(ui)
    if raw_samples % ds != 0:
        raise ValueError(
            "SpectralDS must divide the selected camera sample count: "
            f"raw_samples={raw_samples}, SpectralDS={ds}"
        )
    return raw_samples // ds


def camera_pixel_format(ui):
    camera = current_camera_name(ui)
    if camera == CAMERA_PHOTONFOCUS:
        return str(widget_value(ui, "PixelFormat_display_PF", "Mono12"))
    if camera in CAMERA_HIK_ALIASES and hasattr(ui, "PixelFormat_display_HK"):
        return str(widget_value(ui, "PixelFormat_display_HK", "Mono12"))
    if camera in CAMERA_AK_ALIASES and hasattr(ui, "PixelFormatDisplay_AK"):
        return str(widget_value(ui, "PixelFormatDisplay_AK", "Mono8"))
    return str(widget_value(ui, "PixelFormat_display_DH", "Mono12"))


def downsample_spectral_axis(data, ratio, axis):
    ratio = max(1, int(ratio))
    if ratio == 1:
        return data
    samples = int(data.shape[axis])
    if samples % ratio != 0:
        raise ValueError(
            f"SpectralDS={ratio} must divide spectral samples={samples}"
        )
    out_samples = samples // ratio
    moved = data.swapaxes(axis, -1)
    out_shape = moved.shape[:-1] + (out_samples, ratio)
    downsampled = moved.reshape(out_shape).mean(axis=-1)
    downsampled = downsampled.swapaxes(axis, -1)
    return downsampled.astype(data.dtype, copy=False)
