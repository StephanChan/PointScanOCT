"""配置对象的校验与序列化。"""

import pytest

from octsdk import OctConfig, ProcessingConfig, ScanConfig
from octsdk.errors import ConfigurationError


def test_scan_geometry_derives_x_pixels():
    config = OctConfig(
        scan=ScanConfig(x_step_um=10.0, x_length_um=1000.0),
        processing=ProcessingConfig(raw_spectral_samples=1440, aline_avg=4),
    )
    assert config.scan.x_positions == 100
    assert config.resolved_processing().x_pixels == 400


def test_explicit_x_pixels_wins():
    config = OctConfig(
        scan=ScanConfig(x_step_um=10.0, x_length_um=1000.0),
        processing=ProcessingConfig(raw_spectral_samples=1440, x_pixels=32),
    )
    assert config.resolved_processing().x_pixels == 32


def test_downsample_must_divide_samples():
    with pytest.raises(ConfigurationError):
        ProcessingConfig(raw_spectral_samples=1440, spectral_downsample=7)


def test_aline_avg_must_divide_x_pixels():
    config = OctConfig(processing=ProcessingConfig(raw_spectral_samples=64, x_pixels=5, aline_avg=2))
    with pytest.raises(ConfigurationError):
        config.resolved_processing()


def test_depth_window_must_fit_inside_spectrum():
    config = OctConfig(
        processing=ProcessingConfig(raw_spectral_samples=1024, depth_start=1000, depth_pixels=100)
    )
    with pytest.raises(ConfigurationError):
        config.resolved_processing()


def test_unknown_output_dtype_rejected():
    with pytest.raises(ConfigurationError):
        ProcessingConfig(output_dtype="float12")


def test_lossy_encoder_with_keep_complex_rejected():
    with pytest.raises(ConfigurationError):
        ProcessingConfig(keep_complex=True, output_dtype="float16")


def test_config_round_trip():
    config = OctConfig()
    restored = OctConfig.from_dict(config.to_dict())
    assert restored.to_dict() == config.to_dict()


def test_unknown_keys_are_ignored_on_load():
    payload = OctConfig().to_dict()
    payload["processing"]["legacy_field_from_old_version"] = 42
    restored = OctConfig.from_dict(payload)
    assert isinstance(restored, OctConfig)
    assert restored.processing.raw_spectral_samples == 1440


def test_aodo_sample_rate_follows_line_rate():
    config = OctConfig()
    assert config.aodo.resolved_sample_rate_hz(config.camera) == pytest.approx(
        2.0 * config.camera.line_rate_hz
    )
