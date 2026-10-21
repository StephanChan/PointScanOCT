"""``OctProcessor`` 端到端：形状、FFT 参考值、平均、复数、标定加载、CPU/GPU 一致性。"""

import numpy as np
import pytest

from octsdk import OctConfig, OctProcessor, ProcessingConfig, ScanConfig
from octsdk.errors import ProcessingError


def make_config(samples=64, **overrides):
    processing = {"raw_spectral_samples": samples, "x_pixels": 4}
    processing.update(overrides)
    config = OctConfig(
        scan=ScanConfig(x_step_um=10.0, x_length_um=40.0),
        processing=ProcessingConfig(**processing),
    )
    return config.resolved_processing()


def make_processor(samples=64, backend="cpu", **overrides):
    processor = OctProcessor(make_config(samples, **overrides), backend=backend)
    processor.prepare()
    return processor


def test_process_reports_shape_and_metadata():
    processor = make_processor()
    raw = np.random.rand(8, 64).astype(np.float32)
    package = processor.process(raw)
    assert package.data.shape == (8, 64)
    assert package.data.dtype == np.float32
    assert (package.rows, package.x_pixels, package.depth_pixels) == (8, 4, 64)
    assert package.backend == "cpu"
    assert package.metadata["calibration_used"] is False
    assert package.metadata["background_subtracted"] is False


def test_process_matches_reference_fft():
    processor = make_processor(amplification=2.0)
    raw = np.random.rand(8, 64).astype(np.float32)
    package = processor.process(raw)
    reference = np.abs(np.fft.fft(raw, axis=1)) / 64.0 * 2.0
    assert np.allclose(package.data, reference.astype(np.float32), rtol=1e-5, atol=1e-6)


def test_process_applies_spectral_average_in_spectral_domain():
    # 1 条 Y 行：BlineAVG=2 × 4 个位置 × AlineAVG=2 = 16 行
    processor = make_processor(aline_avg=2, bline_avg=2, amplification=1.0)
    raw = np.random.rand(16, 64).astype(np.float32)
    package = processor.process(raw)
    assert package.rows == 4
    assert package.metadata["spectral_aline_avg"] == 2
    assert package.metadata["spectral_bline_avg"] == 2
    # 16 行 = 2 条 Y 行 × BlineAVG 2 × 2 个位置 × AlineAVG 2（x_pixels=4 含 AlineAVG）
    averaged = raw.reshape(2, 2, 2, 2, 64).mean(axis=(1, 3)).reshape(4, 64)
    reference = np.abs(np.fft.fft(averaged, axis=1)) / 64.0
    assert np.allclose(package.data, reference.astype(np.float32), rtol=1e-5, atol=1e-6)


def test_complex_output_keeps_phase():
    processor = make_processor(keep_complex=True)
    raw = np.random.rand(8, 64).astype(np.float32)
    package = processor.process(raw)
    assert package.data.dtype == np.complex64
    assert np.allclose(np.abs(package.data), package.amplitude(), rtol=0, atol=0)


def test_y_notch_on_complex_volume_keeps_complex():
    processor = make_processor(keep_complex=True)
    volume = np.full((4, 2, 2), 3.0 + 1.0j, dtype=np.complex64)
    filtered = processor.apply_y_notch(volume)
    assert filtered.dtype == np.complex64
    assert filtered.shape == volume.shape
    assert float(np.max(np.abs(filtered))) < 1e-5


def test_wrong_spectral_length_raises():
    processor = make_processor()
    with pytest.raises(ProcessingError):
        processor.process(np.zeros((4, 32), dtype=np.float32))


def test_calibration_and_background_round_trip(tmp_path):
    samples = 16
    axis = np.arange(samples, dtype=np.float32)
    (tmp_path / "intpX.bin").write_bytes(axis.tobytes())
    (tmp_path / "intpXp.bin").write_bytes(axis.tobytes())
    indice = np.vstack([np.arange(samples), np.clip(np.arange(samples) + 1, 0, samples - 1)])
    (tmp_path / "intpIndice.bin").write_bytes(indice.astype(np.uint16).tobytes())
    (tmp_path / "dspPhase.bin").write_bytes(np.zeros(samples, dtype=np.float32).tobytes())
    background = tmp_path / "background.bin"
    background.write_bytes(np.zeros(samples, dtype=np.float32).tobytes())

    processor = make_processor(
        samples=samples,
        calibration_dir=str(tmp_path),
        background_path=str(background),
        amplification=1.0,
    )
    report = processor.prepare()
    assert report["calibration"]["usable"] is True
    assert report["background"] is True

    raw = np.random.rand(8, samples).astype(np.float32)
    package = processor.process(raw)
    reference = np.abs(np.fft.fft(raw, axis=1)) / samples
    assert package.metadata["calibration_used"] is True
    assert np.allclose(package.data, reference.astype(np.float32), rtol=1e-4, atol=1e-5)


def test_gpu_and_cpu_backends_agree():
    pytest.importorskip("cupy")
    overrides = {
        "aline_avg": 2,
        "x_notch_enabled": True,
        "x_notch_half_width_fraction": 0.1,
        "amplification": 3.0,
    }
    raw = np.random.rand(8, 64).astype(np.float32)
    cpu_package = make_processor(backend="cpu", **overrides).process(raw)
    gpu_package = make_processor(backend="gpu", **overrides).process(raw)
    assert gpu_package.backend == "gpu"
    assert gpu_package.data.shape == cpu_package.data.shape
    assert np.allclose(gpu_package.data, cpu_package.data, rtol=1e-4, atol=1e-4)


def test_gpu_and_cpu_agree_on_y_notch():
    pytest.importorskip("cupy")
    volume = np.random.rand(16, 3, 5).astype(np.float32)
    cpu_filtered = make_processor(backend="cpu").apply_y_notch(volume)
    gpu_filtered = make_processor(backend="gpu").apply_y_notch(volume)
    assert np.allclose(gpu_filtered, cpu_filtered, rtol=1e-4, atol=1e-4)
