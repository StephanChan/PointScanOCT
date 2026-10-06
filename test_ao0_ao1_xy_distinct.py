# -*- coding: utf-8 -*-
"""Standalone ART-DAQ distinct X/Y waveform test on ao0:1."""

import argparse
import os
import sys
import time

import numpy as np


ARTDAQ_PYTHON_LIB_DIR = (
    r"C:\Program Files (x86)\ART Technology\ART-DAQ\Samples\Python\LIB"
)
if os.path.isdir(ARTDAQ_PYTHON_LIB_DIR):
    sys.path.append(ARTDAQ_PYTHON_LIB_DIR)

import artdaq as ni
from artdaq.constants import AcquisitionType, Edge
from artdaq.stream_writers import AnalogMultiChannelWriter


def build_xy_waveform(sample_rate_hz, duration_s, x_amp_v, y_amp_v, x_freq_hz, y_freq_hz):
    sample_count = int(round(float(sample_rate_hz) * float(duration_s)))
    t = np.arange(sample_count, dtype=np.float64) / float(sample_rate_hz)
    x = x_amp_v * np.sin(2.0 * np.pi * x_freq_hz * t)
    y = y_amp_v * np.sin(2.0 * np.pi * y_freq_hz * t + np.pi / 2.0)
    return np.ascontiguousarray(np.vstack([x, y]), dtype=np.float64)


def run_test(device, sample_rate_hz, duration_s, x_amp_v, y_amp_v, x_freq_hz, y_freq_hz):
    ao_channels = f"{device}/ao0:1"
    waveform = build_xy_waveform(
        sample_rate_hz,
        duration_s,
        x_amp_v,
        y_amp_v,
        x_freq_hz,
        y_freq_hz,
    )
    print(
        "Starting distinct two-axis AO test: "
        f"channels={ao_channels}, shape={waveform.shape}, "
        f"X={x_amp_v:g} V @ {x_freq_hz:g} Hz, "
        f"Y={y_amp_v:g} V @ {y_freq_hz:g} Hz, duration={duration_s:g} s"
    )

    with ni.Task("ao0_ao1_xy_distinct") as task:
        task.ao_channels.add_ao_voltage_chan(
            physical_channel=ao_channels,
            min_val=-10.0,
            max_val=10.0,
            units=ni.constants.VoltageUnits.VOLTS,
        )
        task.timing.cfg_samp_clk_timing(
            rate=float(sample_rate_hz),
            active_edge=Edge.RISING,
            sample_mode=AcquisitionType.FINITE,
            samps_per_chan=waveform.shape[1],
        )
        writer = AnalogMultiChannelWriter(task.out_stream, auto_start=False)
        writer.verify_array_shape = False
        written = writer.write_many_sample(waveform)
        print(f"Samples written per channel: {written}")
        task.start()
        task.wait_until_done(timeout=float(duration_s) + 5.0)
        task.stop()

    with ni.Task("ao0_ao1_zero") as task:
        task.ao_channels.add_ao_voltage_chan(
            physical_channel=ao_channels,
            min_val=-10.0,
            max_val=10.0,
            units=ni.constants.VoltageUnits.VOLTS,
        )
        writer = AnalogMultiChannelWriter(task.out_stream, auto_start=True)
        writer.verify_array_shape = False
        writer.write_one_sample(np.array([0.0, 0.0], dtype=np.float64))
        time.sleep(0.05)
        task.stop()

    print("AO test complete. ao0 and ao1 returned to 0 V.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="Galvo", help="ART-DAQ device name.")
    parser.add_argument("--rate", type=float, default=1000.0, help="Sample rate in Hz.")
    parser.add_argument("--duration", type=float, default=10.0, help="Output duration in seconds.")
    parser.add_argument("--xamp", type=float, default=1.0, help="X/ao0 sine amplitude in volts.")
    parser.add_argument("--yamp", type=float, default=1.0, help="Y/ao1 sine amplitude in volts.")
    parser.add_argument("--xfreq", type=float, default=7.0, help="X/ao0 sine frequency in Hz.")
    parser.add_argument("--yfreq", type=float, default=13.0, help="Y/ao1 sine frequency in Hz.")
    args = parser.parse_args()
    run_test(args.device, args.rate, args.duration, args.xamp, args.yamp, args.xfreq, args.yfreq)


if __name__ == "__main__":
    main()
