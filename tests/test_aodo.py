"""AODO：波形纯函数 + SIM 设备（不需要 DAQ）。"""

import types

import numpy as np
import pytest

from octsdk import AodoConfig, ProcessingConfig, ScanConfig
from octsdk.aodo import build_waveform, open_aodo, scan_voltage_limits
from octsdk.config import AcqModes
from octsdk.errors import ConfigurationError


def make_scan(x_step_um=10.0, positions=4, y_lines=2, flyback=4):
    return ScanConfig(
        x_step_um=x_step_um,
        x_length_um=x_step_um * positions,
        y_step_um=10.0,
        y_lines=y_lines,
        flyback_samples=flyback,
    )


def make_processing(aline_avg=1, bline_avg=1):
    return ProcessingConfig(
        raw_spectral_samples=64,
        x_pixels=4 * aline_avg,
        aline_avg=aline_avg,
        bline_avg=bline_avg,
    )


def make_aodo(**overrides):
    params = {"x_mm_per_volt": 2.0, "y_mm_per_volt": 2.0, "sample_rate_hz": 20000.0}
    params.update(overrides)
    return AodoConfig(**params)


def test_bline_waveform_layout():
    spec = build_waveform(make_scan(), make_processing(), make_aodo(), AcqModes.FINITE_BSCAN)
    assert spec.ao.shape == (2, spec.samples)
    assert spec.line_samples == 2 * 4 + 4
    assert spec.samples == spec.line_samples  # BlineAVG = 1
    assert int(np.count_nonzero(spec.do)) == 4  # 4 个振镜位置，各 1 个脉冲
    assert np.allclose(spec.ao[1], 0.0)  # Y 保持偏置
    assert spec.ao[0].max() > spec.ao[0].min()


def test_cscan_waveform_layout():
    scan = make_scan(y_lines=3)
    spec = build_waveform(
        scan, make_processing(bline_avg=2), make_aodo(), AcqModes.FINITE_CSCAN
    )
    line_samples = 2 * 4 + 4
    assert spec.line_samples == line_samples
    assert spec.samples == 3 * 2 * line_samples
    assert int(np.count_nonzero(spec.do)) == 3 * 2 * 4
    assert spec.ao[1].min() < 0.0 < spec.ao[1].max()


def test_voltage_limits_follow_app_convention():
    vmin, vmax = scan_voltage_limits(1.0, 2.0, 0.0)
    assert vmax == pytest.approx(1.0 / 2.0 / 2.0 / 2.0)
    assert vmin == pytest.approx(-1.0 / 2.0 / 2.0 / 2.0)
    biased = scan_voltage_limits(1.0, 2.0, 0.5)
    assert biased[0] == pytest.approx(0.5 - 0.125)
    assert biased[1] == pytest.approx(0.5 + 0.125)


def test_missing_scale_calibration_raises():
    with pytest.raises(ConfigurationError):
        build_waveform(make_scan(), make_processing(), AodoConfig(), AcqModes.FINITE_BSCAN)


def test_voltage_limit_is_enforced():
    scan = make_scan(x_step_um=10000.0, positions=100)
    with pytest.raises(ConfigurationError):
        build_waveform(
            scan, make_processing(), make_aodo(ao_voltage_limit_v=1.0), AcqModes.FINITE_CSCAN
        )


def test_unsupported_mode_raises():
    with pytest.raises(ConfigurationError):
        build_waveform(make_scan(), make_processing(), make_aodo(), "NotAMode")


def test_simulated_aodo_configures_and_runs():
    camera = types.SimpleNamespace(line_rate_hz=10000.0)
    device = open_aodo(make_aodo(sample_rate_hz=None, simulate=True), camera_config=camera)
    device.open()
    report = device.configure(make_scan(), make_processing(), AcqModes.FINITE_BSCAN)
    assert report["sample_rate_hz"] == pytest.approx(20000.0)
    assert report["do_pulses"] == 4
    assert report["ao_channel_spec"] == "Galvo/ao0:1"
    assert report["channel_order"] == ["Y", "X"]
    assert device.is_running is False
    device.start()
    assert device.is_running is True
    device.stop()
    assert device.is_running is False
    device.close()
    assert device.is_open is False


# -- ART-DAQ 映射：假 SDK 验证通道/采样率/启动顺序（不需要 DAQ） ------------- #
_START_ORDER = []


class _FakeChannelCollection:
    def __init__(self, sink):
        self._sink = sink

    def add_ao_voltage_chan(self, **kwargs):
        self._sink.append(("ao_chan", kwargs))

    def add_do_chan(self, **kwargs):
        self._sink.append(("do_chan", kwargs))


class _FakeTiming:
    def __init__(self, sink):
        self._sink = sink

    def cfg_samp_clk_timing(self, **kwargs):
        self._sink.append(("timing", kwargs))


class _FakeStartTrigger:
    def __init__(self, sink):
        self._sink = sink

    def cfg_dig_edge_start_trig(self, terminal):
        self._sink.append(("start_trigger", terminal))


class _FakeTriggers:
    def __init__(self, sink):
        self.start_trigger = _FakeStartTrigger(sink)


class _FakeExportSignals:
    def __init__(self, sink):
        self._sink = sink

    def export_signal(self, signal_id, output_terminal):
        self._sink.append(("export", (signal_id, output_terminal)))


class _FakeOutStream:
    def __init__(self):
        self.regen_mode = None


class _FakeTask:
    def __init__(self, name):
        self.name = name
        self.calls = []
        self.ao_channels = _FakeChannelCollection(self.calls)
        self.do_channels = _FakeChannelCollection(self.calls)
        self.timing = _FakeTiming(self.calls)
        self.triggers = _FakeTriggers(self.calls)
        self.export_signals = _FakeExportSignals(self.calls)
        self.out_stream = _FakeOutStream()
        self.started = False

    def start(self):
        self.started = True
        _START_ORDER.append(self.name)

    def stop(self):
        self.started = False

    def close(self):
        pass


class _FakeWriter:
    instances = []

    def __init__(self, stream, auto_start=False):
        self.stream = stream
        self.verify_array_shape = True
        self.written = None
        _FakeWriter.instances.append(self)

    def write_many_sample(self, data):
        self.written = np.asarray(data)

    def write_many_sample_port_u_32(self, data):
        self.written = np.asarray(data)


def _install_fake_sdk(monkeypatch):
    from octsdk.aodo import artdaq as artdaq_module

    fake_sdk = types.SimpleNamespace(
        ni=types.SimpleNamespace(
            Task=_FakeTask,
            constants=types.SimpleNamespace(VoltageUnits=types.SimpleNamespace(VOLTS="V")),
        ),
        AcquisitionType=types.SimpleNamespace(CONTINUOUS="continuous", FINITE="finite"),
        Edge=types.SimpleNamespace(RISING="rising"),
        Signal=types.SimpleNamespace(START_TRIGGER="start_trigger"),
        RegenerationMode=types.SimpleNamespace(ALLOW_REGENERATION="allow"),
        LineGrouping=types.SimpleNamespace(CHAN_PER_LINE="per_line"),
        AnalogMultiChannelWriter=_FakeWriter,
        DigitalSingleChannelWriter=_FakeWriter,
    )
    monkeypatch.setattr(artdaq_module, "bootstrap_artdaq", lambda root=None: fake_sdk)
    _START_ORDER.clear()
    _FakeWriter.instances.clear()
    return artdaq_module


def test_artdaq_maps_channels_and_starts_do_first(monkeypatch):
    artdaq_module = _install_fake_sdk(monkeypatch)
    camera = types.SimpleNamespace(line_rate_hz=10000.0)
    device = artdaq_module.ArtDaqAodoDevice(make_aodo(), camera_config=camera)
    device.open()
    report = device.configure(make_scan(), make_processing(), AcqModes.FINITE_BSCAN)

    assert report["ao_channel_spec"] == "Galvo/ao0:1"
    assert report["channel_order"] == ["Y", "X"]  # X=ao1、Y=ao0，相邻通道按编号排序
    assert report["trigger_out"] == "/Galvo/PFI3"
    assert report["sync_do_line"] == "Galvo/port0/line0"

    ao_task, do_task = device._ao_task, device._do_task
    ao_chan = [payload for kind, payload in ao_task.calls if kind == "ao_chan"][0]
    assert ao_chan["physical_channel"] == "Galvo/ao0:1"
    assert ao_chan["max_val"] == pytest.approx(10.0)
    ao_timing = [payload for kind, payload in ao_task.calls if kind == "timing"][0]
    assert ao_timing["rate"] == pytest.approx(20000.0)
    assert ao_timing["samps_per_chan"] == device.waveform.samples
    assert ao_timing["sample_mode"] == "finite"
    assert ("export", ("start_trigger", "/Galvo/PFI3")) in ao_task.calls

    do_chan = [payload for kind, payload in do_task.calls if kind == "do_chan"][0]
    assert do_chan["lines"] == "Galvo/port0/line0"
    do_timing = [payload for kind, payload in do_task.calls if kind == "timing"][0]
    assert do_timing["samps_per_chan"] == device.waveform.do.size
    assert ("start_trigger", "/Galvo/PFI7") in do_task.calls

    device.start()
    assert _START_ORDER == ["octsdk-do", "octsdk-ao"]
    ao_written = _FakeWriter.instances[0].written
    do_written = _FakeWriter.instances[1].written
    assert ao_written.shape == (2, device.waveform.samples)
    assert np.allclose(ao_written[0], device.waveform.ao[1])  # Y 在前（ao0）
    assert np.allclose(ao_written[1], device.waveform.ao[0])  # X 在后（ao1）
    assert do_written.size == device.waveform.do.size
    device.stop()
    device.close()


def test_artdaq_continuous_mode_uses_continuous_timing(monkeypatch):
    artdaq_module = _install_fake_sdk(monkeypatch)
    camera = types.SimpleNamespace(line_rate_hz=10000.0)
    device = artdaq_module.ArtDaqAodoDevice(make_aodo(), camera_config=camera)
    device.open()
    report = device.configure(make_scan(), make_processing(), AcqModes.CONTINUOUS_BLINE)
    assert report["continuous"] is True
    ao_timing = [payload for kind, payload in device._ao_task.calls if kind == "timing"][0]
    assert ao_timing["sample_mode"] == "continuous"
    device.close()
