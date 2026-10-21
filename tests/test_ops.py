"""数组算子：光谱域平均、陷波、输出编码。"""

import numpy as np
import pytest

from octsdk.processing import ops


def test_spectral_average_matches_manual_mean():
    x_pixels, aline_avg, bline_avg, y_lines, samples = 8, 2, 3, 2, 5
    rows = y_lines * bline_avg * x_pixels
    truth = np.arange(y_lines * (x_pixels // aline_avg), dtype=np.float32).reshape(
        y_lines, x_pixels // aline_avg, 1
    )
    spectra = np.zeros((rows, samples), dtype=np.float32)
    for y in range(y_lines):
        for repeat in range(bline_avg):
            for position in range(x_pixels // aline_avg):
                for aline in range(aline_avg):
                    row = ((y * bline_avg + repeat) * x_pixels) + position * aline_avg + aline
                    spectra[row, :] = truth[y, position, 0] + aline * 100.0 + repeat * 1000.0

    averaged = ops.apply_spectral_average(np, spectra, x_pixels, aline_avg, bline_avg)

    assert averaged.shape == (rows // (aline_avg * bline_avg), samples)
    expected = (truth + 0.5 * 100.0 + 1.0 * 1000.0).reshape(-1)
    assert np.allclose(averaged[:, 0], expected)


def test_spectral_average_drops_incomplete_group():
    # group_rows = x_pixels * bline_avg = 4 * 2 = 8，10 行只剩 8 行 → 2 行输出
    averaged = ops.apply_spectral_average(
        np, np.zeros((10, 4), dtype=np.float32), 4, aline_avg=2, bline_avg=2
    )
    assert averaged.shape == (2, 4)


def test_spectral_average_rejects_unaligned_aline_avg():
    with pytest.raises(Exception):
        ops.apply_spectral_average(np, np.zeros((8, 4), np.float32), 5, aline_avg=2)


def test_y_notch_removes_dc_component():
    volume = np.full((8, 4, 3), 5.0, dtype=np.float32)  # 沿 Y 恒定 = 纯 DC
    filtered = ops.apply_y_notch(np, volume, 0.25, z_chunk=2)
    assert filtered.shape == volume.shape
    assert float(np.max(np.abs(filtered))) < 1e-5


def test_y_notch_keeps_nyquist_component():
    alternating = ((-1.0) ** np.arange(8)).astype(np.float32)
    volume = np.repeat(alternating[:, None, None], 4, axis=1).repeat(3, axis=2)
    filtered = ops.apply_y_notch(np, volume, 0.25, z_chunk=1)
    assert float(np.abs(filtered).min()) > 0.9


def test_x_notch_rejects_partial_bline():
    with pytest.raises(Exception):
        ops.apply_x_notch(np, np.zeros((10, 4), np.float32), x_pixels=4, half_width_fraction=0.1)


def test_notch_weights_shape_and_dc_block():
    weights = ops.notch_weights(np, 16, 0.0)
    assert weights.shape == (16,)
    # fraction=0 只挡 DC 一个频点（native 顺序下在索引 0）
    assert weights[0] == 0.0
    assert weights[1] == 1.0


def test_encode_output_amp_phase_int16_metadata():
    data = np.array([[3 + 4j, 0.0 + 0j]], dtype=np.complex64)
    encoded, metadata = ops.encode_output(data, "amp_phase_int16")
    assert encoded.dtype == np.int16
    assert encoded.shape == (1, 2, 2)
    assert metadata["amplitude_scale"] == pytest.approx(5.0)
    assert metadata["phase_unit"] == "rad"
    assert encoded[0, 0, 0] == 32767  # 峰值幅度归一化到满量程


def test_encode_output_float32_takes_amplitude():
    data = np.array([[3 + 4j]], dtype=np.complex64)
    encoded, metadata = ops.encode_output(data, "float32")
    assert encoded.dtype == np.float32
    assert encoded[0, 0] == pytest.approx(5.0)
    assert metadata["encoding"] == "float32"


def test_encode_output_rejects_real_input_for_complex_modes():
    with pytest.raises(Exception):
        ops.encode_output(np.zeros((2, 2), dtype=np.float32), "complex64")
