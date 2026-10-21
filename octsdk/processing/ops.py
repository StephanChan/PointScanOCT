"""与后端无关的数组算子。

每个函数第一个参数是 ``xp``——``numpy`` 或 ``cupy``，两条路径共用同一套公式，
因此 CPU 与 GPU 的结果可以逐元素对比（见 ``tests/test_processor_parity.py``）。

处理链顺序（与主程序 ``ThreadGPU`` 的 GPU 路径一致）：

    光谱域 A-line/B-line 平均 → 背景减 → 插值(色散重采样) → X 陷波
    → 乘色散相位并 FFT → 深度裁剪 → 幅度(/保留复数) → 增益 → X 归一化
"""

import numpy as np

from ..errors import ProcessingError


def apply_spectral_average(xp, spectra, x_pixels, aline_avg=1, bline_avg=1):
    """在光谱域把 ``[Y][Bline重复][X位置][Aline重复]`` 求平均。

    采集顺序由 GenAODO 决定：每个 Y 行内先重复 BlineAVG 条 B-line，每条 B-line 内
    每个振镜位置重复 AlineAVG 个 A-line，所以行序是 ``[Y][重复][X][Aline]``。
    FFT 是线性运算，因此这里平均等价于对复数 FFT 结果平均，但数据量更小、噪声底更低。
    """
    aline_avg = max(1, int(aline_avg))
    bline_avg = max(1, int(bline_avg))
    x_pixels = max(1, int(x_pixels))
    if x_pixels % aline_avg:
        raise ProcessingError(
            "AlineAVG must divide the A-lines per B-line: "
            f"x_pixels={x_pixels}, aline_avg={aline_avg}"
        )
    if aline_avg == 1 and bline_avg == 1:
        return spectra
    x_positions = x_pixels // aline_avg
    group_rows = x_pixels * bline_avg
    rows = int(spectra.shape[0])
    complete = (rows // group_rows) * group_rows
    if complete <= 0:
        return spectra[:0]
    samples = int(spectra.shape[-1])
    block = spectra[:complete].reshape(
        complete // group_rows, bline_avg, x_positions, aline_avg, samples
    )
    averaged = block.mean(axis=3, dtype=xp.float32).mean(axis=1, dtype=xp.float32)
    return averaged.reshape(complete // (aline_avg * bline_avg), samples)


def subtract_background(xp, spectra, background):
    """原地减背景；``background`` 形状 ``(1, samples)``，直接广播。"""
    spectra -= background
    return spectra


def interpolate_spectra(xp, spectra, arrays):
    """线性插值到线性 k 轴（与主程序 ``interp1d`` CUDA 内核同公式）。"""
    indice1 = arrays["indice1"]
    indice2 = arrays["indice2"]
    x0 = arrays["intp_x"][indice1]
    x1 = arrays["intp_x"][indice2]
    xt = arrays["intp_xp"]
    y0 = spectra[:, indice1]
    y1 = spectra[:, indice2]
    denominator = x1 - x0
    denominator = xp.where(xp.abs(denominator) < 1e-5, xp.float32(1e-5), denominator)
    return y0 + (xt - x0) * (y1 - y0) / denominator


def apply_dispersion_and_fft(xp, spectra, arrays, samples):
    """乘色散相位后沿光谱轴做 FFT，并除以 FFT 长度（与主程序一致）。"""
    samples = int(samples)
    if arrays is None:
        data = xp.fft.fft(spectra, axis=1) / samples
    else:
        data = xp.fft.fft(spectra * arrays["dispersion"], axis=1) / samples
    if data.dtype != xp.complex64:
        data = data.astype(xp.complex64)
    return data


def crop_depth(xp, data, start, pixels):
    """按深度窗裁剪（等价于主程序 ``DepthStart / DepthRange``）。"""
    start = int(start)
    pixels = int(pixels)
    samples = int(data.shape[-1])
    if start < 0 or pixels < 1:
        raise ProcessingError(f"invalid depth window: start={start}, pixels={pixels}")
    if start + pixels > samples:
        raise ProcessingError(
            f"depth window exceeds FFT length: start={start}, pixels={pixels}, samples={samples}"
        )
    return data[:, start:start + pixels]


def apply_x_normalization(xp, data, normalization, x_pixels):
    """按每个 Y 行内的 X 位置做归一化（``normalization`` 长度 = 有效 X 位置数）。"""
    rows = int(data.shape[0])
    x_pixels = max(1, int(x_pixels))
    n_y = rows // x_pixels
    if n_y < 1:
        return data
    view = data[: n_y * x_pixels].reshape(n_y, x_pixels, int(data.shape[1]))
    view /= normalization[xp.newaxis, :, xp.newaxis]
    return data


# -- 陷波滤波 --------------------------------------------------------------- #
def notch_weights(xp, length, half_width_fraction):
    """DC 居中的带阻权重：``|f| <= fraction × Nyquist`` 的频点被置零。

    与主程序同一约定：native FFT 顺序（0 频在两端），Nyquist = 0.5 cycles/sample。
    """
    length = max(1, int(length))
    fraction = float(np.clip(float(half_width_fraction), 0.0, 1.0))
    return (xp.abs(xp.fft.fftfreq(length)) > xp.float32(fraction * 0.5)).astype(xp.float32)


def apply_x_notch(xp, spectra, x_pixels, half_width_fraction):
    """沿 X（振镜快轴）在光谱域带阻：``[B-line组][X][光谱]`` 变换 → 掩膜 → 变回。"""
    rows = int(spectra.shape[0])
    x_pixels = max(1, int(x_pixels))
    if rows % x_pixels:
        raise ProcessingError(f"X notch needs whole B-lines: rows={rows}, x_pixels={x_pixels}")
    samples = int(spectra.shape[1])
    field = spectra.reshape(rows // x_pixels, x_pixels, samples)
    spectrum = xp.fft.fft(field, axis=1)
    spectrum *= notch_weights(xp, x_pixels, half_width_fraction)[xp.newaxis, :, xp.newaxis]
    filtered = xp.fft.ifft(spectrum, axis=1)
    if filtered.dtype != xp.complex64:
        filtered = filtered.astype(xp.complex64)
    return filtered.reshape(rows, samples)


def apply_y_notch(xp, volume, half_width_fraction, z_chunk=32):
    """沿 Y（C-scan 慢扫轴 = ``CSCAN_PACKAGE_Y_LINES`` 方向）对整卷带阻。

    ``volume`` 形状 ``(Y, X, Z)``。按 Z 分块只为限制显存/内存峰值，Y 方向始终是完整
    长度，所以结果与"整卷一次做"完全等价；复数输入保持复数，实数输入取幅度。
    """
    if getattr(volume, "ndim", 0) != 3:
        raise ProcessingError(
            f"Y notch needs a (Y, X, Z) volume, got {getattr(volume, 'shape', None)}"
        )
    y_pixels = int(volume.shape[0])
    z_pixels = int(volume.shape[2])
    weights = notch_weights(xp, y_pixels, half_width_fraction)
    complex_input = volume.dtype.kind == "c"
    out = xp.empty(volume.shape, dtype=xp.complex64 if complex_input else xp.float32)
    chunk = max(1, int(z_chunk))
    for start in range(0, z_pixels, chunk):
        stop = min(start + chunk, z_pixels)
        spectrum = xp.fft.fft(volume[:, :, start:stop], axis=0)
        spectrum *= weights[:, xp.newaxis, xp.newaxis]
        filtered = xp.fft.ifft(spectrum, axis=0)
        if complex_input:
            out[:, :, start:stop] = filtered.astype(xp.complex64, copy=False)
        else:
            out[:, :, start:stop] = xp.abs(filtered)
    return out


# -- 对外 dtype 编码 --------------------------------------------------------- #
FP16_MAX = 65504.0


def encode_output(data, mode="float32", amplitude_scale=None):
    """把内部 float32/complex64 结果编码成对外 dtype，返回 ``(array, metadata)``。

    ``mode``：
        * ``float32``            —— 幅度（复数取模），推荐默认
        * ``complex64``          —— 保留幅度+相位，需要调用方自己取模
        * ``float16``            —— 幅度半精度；超过 65504 会被裁剪并置 metadata.clipped
        * ``amp_phase_fp16``     —— 末轴 ``[幅度, 相位(rad)]`` 两个 float16
        * ``amp_phase_int16``    —— 末轴 ``[幅度/scale, 相位]`` 两个 int16，scale 见 metadata
    """
    array = np.asarray(data)
    metadata = {"source_dtype": str(array.dtype), "encoding": mode}
    if mode in ("float32", "float16"):
        amplitude = np.abs(array) if array.dtype.kind == "c" else array
        amplitude = np.ascontiguousarray(amplitude, dtype=np.float32)
        if mode == "float32":
            return amplitude, metadata
        peak = float(np.max(amplitude)) if amplitude.size else 0.0
        metadata["peak"] = peak
        if peak > FP16_MAX:
            metadata["clipped"] = True
            np.clip(amplitude, 0.0, FP16_MAX, out=amplitude)
        return amplitude.astype(np.float16), metadata
    if mode == "complex64":
        if array.dtype.kind != "c":
            raise ProcessingError(
                f"complex64 encoding needs complex data, got {array.dtype}"
            )
        return np.ascontiguousarray(array, dtype=np.complex64), metadata
    if mode not in ("amp_phase_fp16", "amp_phase_int16"):
        raise ProcessingError(f"unknown output encoding: {mode!r}")
    if array.dtype.kind != "c":
        raise ProcessingError(f"{mode} encoding needs complex data, got {array.dtype}")

    amplitude = np.abs(array).astype(np.float64)
    phase = np.angle(array).astype(np.float64)
    metadata["phase_unit"] = "rad"
    if mode == "amp_phase_fp16":
        peak = float(amplitude.max()) if amplitude.size else 0.0
        metadata["peak"] = peak
        if peak > FP16_MAX:
            metadata["clipped"] = True
            np.clip(amplitude, 0.0, FP16_MAX, out=amplitude)
        return np.stack([amplitude, phase], axis=-1).astype(np.float16), metadata

    scale = float(amplitude_scale) if amplitude_scale else float(amplitude.max() or 1.0)
    metadata["amplitude_scale"] = scale
    amplitude_i = np.clip(amplitude / scale, 0.0, 1.0) * 32767.0
    phase_i = (phase / (2.0 * np.pi) + 0.5) * 65535.0 - 32768.0
    return (
        np.stack([amplitude_i, phase_i], axis=-1).astype(np.int16),
        metadata,
    )

