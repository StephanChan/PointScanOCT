"""Pipeline 累积 + OctEngine 编排（SIM 相机 / SIM AODO / CPU 处理，无需硬件）。"""

import numpy as np
import pytest

from octsdk import (
    AcqModes,
    AodoConfig,
    CameraConfig,
    EngineConfig,
    OctConfig,
    OctEngine,
    Pipeline,
    ProcessingConfig,
    ScanConfig,
)
from octsdk.errors import ProcessingError


class _FakeProcessed:
    def __init__(self, data):
        self.data = data


def make_processing(x_pixels=4, aline_avg=1, bline_avg=1, samples=8, **overrides):
    params = {
        "raw_spectral_samples": samples,
        "x_pixels": x_pixels,
        "aline_avg": aline_avg,
        "bline_avg": bline_avg,
    }
    params.update(overrides)
    return ProcessingConfig(**params)


def make_engine_config(y_lines=4, y_notch_enabled=False, samples=256, package_y_lines=2):
    return OctConfig(
        scan=ScanConfig(
            x_step_um=10.0, x_length_um=40.0, y_step_um=10.0, y_lines=y_lines,
            flyback_samples=8,
        ),
        processing=ProcessingConfig(
            raw_spectral_samples=samples,
            aline_avg=1,
            bline_avg=1,
            depth_start=8,
            amplification=1.0,
            y_notch_enabled=y_notch_enabled,
        ),
        camera=CameraConfig(simulate=True),
        aodo=AodoConfig(simulate=True, x_mm_per_volt=2.3776, y_mm_per_volt=2.3776),
        engine=EngineConfig(package_y_lines=package_y_lines),
    )


def make_engine(config):
    return OctEngine(
        config,
        backend="cpu",
        camera_options={"pace": False, "target_depth_fraction": 0.25, "seed": 5},
    )


# -- Pipeline --------------------------------------------------------------- #
def test_pipeline_accumulates_by_package_start():
    processing = make_processing()
    pipeline = Pipeline(processing, y_lines=2)
    row = np.arange(4 * 8, dtype=np.float32).reshape(4, 8)
    assert pipeline.add_package(_FakeProcessed(row), package_start=0) == 1
    assert pipeline.add_package(_FakeProcessed(row + 100.0), package_start=4) == 1
    assert pipeline.filled_y == 2
    assert pipeline.is_complete is True
    assert np.allclose(pipeline.volume[0], row)
    assert np.allclose(pipeline.volume[1], row + 100.0)
    assert pipeline.describe()["packages"] == 2


def test_pipeline_rejects_bad_shape_and_oversized_volume():
    pipeline = Pipeline(make_processing(), y_lines=1)
    with pytest.raises(ProcessingError):
        pipeline.add_package(_FakeProcessed(np.zeros((5, 8), dtype=np.float32)), 0)
    with pytest.raises(ProcessingError):
        Pipeline(
            make_processing(samples=1024, x_pixels=100, depth_pixels=1024),
            y_lines=100000,
            max_volume_gb=0.001,
        )


def test_pipeline_single_row_keeps_latest_bline():
    pipeline = Pipeline(make_processing(), y_lines=1)
    pipeline.add_package(_FakeProcessed(np.ones((4, 8), dtype=np.float32)), 0)
    pipeline.add_package(_FakeProcessed(np.full((4, 8), 7.0, dtype=np.float32)), 4)
    assert pipeline.filled_y == 1
    assert np.allclose(pipeline.bline(), 7.0)
    assert pipeline.aip().shape == (1, 4)


# -- Engine ----------------------------------------------------------------- #
def test_engine_single_bscan_reaches_target_depth():
    config = make_engine_config(y_lines=1)
    with make_engine(config) as engine:
        result = engine.single_scan(AcqModes.FINITE_BSCAN)
    assert result.completed is True
    assert result.packages >= 1
    assert result.bline is not None
    depth_start = int(config.resolved_processing().depth_start)
    peak = int(np.argmax(result.bline)) + depth_start
    assert abs(peak - 256 // 4) <= 1


def test_engine_single_cscan_notch_and_save(tmp_path):
    pytest.importorskip("tifffile")
    import tifffile

    config = make_engine_config(y_lines=4, y_notch_enabled=True)
    path = tmp_path / "cscan.tif"
    with make_engine(config) as engine:
        result = engine.single_scan(AcqModes.FINITE_CSCAN, save_path=path)

    processing = config.resolved_processing()
    assert result.completed is True
    assert result.filled_y == 4
    assert result.volume.shape == (4, processing.x_pixels, processing.resolved_depth_pixels)
    assert result.aip.shape == (4, processing.x_pixels)
    assert result.y_notch_applied is True
    assert result.saved_path == str(path)
    with tifffile.TiffFile(path) as tiff:
        assert len(tiff.pages) == 4


def test_engine_repeat_scan_stops_by_package_count():
    config = make_engine_config(y_lines=1)
    with make_engine(config) as engine:
        result = engine.repeat_scan(AcqModes.CONTINUOUS_BLINE, max_packages=3)
    assert result.mode == AcqModes.CONTINUOUS_BLINE
    assert result.packages >= 3
    assert result.bline is not None
    assert result.completed is True


def test_engine_rejects_wrong_scan_api():
    config = make_engine_config(y_lines=2)
    with make_engine(config) as engine:
        with pytest.raises(Exception):
            engine.single_scan(AcqModes.CONTINUOUS_BLINE)
        with pytest.raises(Exception):
            engine.repeat_scan(AcqModes.FINITE_BSCAN)
