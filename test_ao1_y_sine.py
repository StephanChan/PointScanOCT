 
q# -*- coding: utf-8 -*-
"""Standalone ART-DAQ AO1/Y-axis sine-wave test."""

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
from artdaq.stream_writers import AnalogMultiChannelWriter, AnalogSingleChannelWriter


def build_sine(sample_rate_hz, frequency_hz, duration_s, amplitude_v, offset_v):
    sample_count = int(round(float(sample_rate_hz) * float(duration_s)))
    t = np.arange(sample_count, dtype=np.float64) / float(sample_rate_hz)
    return offset_v + amplitude_v * np.sin(2.0 * np.pi * frequency_hz * t)


def run_test(device, channel, sample_rate_hz, frequency_hz, duration_s, amplitude_v, offset_v):
    ao_channel = f"{device}/{channel}"
    waveform = np.ascontiguousarray(
        build_sine(sample_rate_hz, frequency_hz, duration_s, amplitude_v, offset_v),
        dtype=np.float64,
    )
    print(
        "Starting single-axis AO test: "
        f"channel={ao_channel}, samples={waveform.size}, "
        f"sample_rate={sample_rate_hz:g} Hz, frequency={frequency_hz:g} Hz, "
        f"amplitude={amplitude_v:g} V, offset={offset_v:g} V, duration={duration_s:g} s"
    )

    with ni.Task("ao1_y_sine_test") as task:
        task.ao_channels.add_ao_voltage_chan(
            physical_channel=ao_channel,
            min_val=-10.0,
            max_val=10.0,
            units=ni.constants.VoltageUnits.VOLTS,
        )
        task.timing.cfg_samp_clk_timing(
            rate=float(sample_rate_hz),
            active_edge=Edge.RISING,
            sample_mode=AcquisitionType.FINITE,
            samps_per_chan=waveform.size,
        )
        writer = AnalogSingleChannelWriter(task.out_stream, auto_start=False)
        writer.verify_array_shape = False
        written = writer.write_many_sample(waveform)
        print(f"Samples written: {written}")
        task.start()
        task.wait_until_done(timeout=float(duration_s) + 5.0)
        task.stop()

    with ni.Task("ao1_y_zero") as task:
        task.ao_channels.add_ao_voltage_chan(
            physical_channel=ao_channel,
            min_val=-10.0,
            max_val=10.0,
            units=ni.constants.VoltageUnits.VOLTS,
        )
        writer = AnalogSingleChannelWriter(task.out_stream, auto_start=True)
        writer.verify_array_shape = False
        writer.write_one_sample(float(offset_v))
        time.sleep(0.05)
        task.stop()

    print(f"AO test complete. {ao_channel} returned to {offset_v:g} V.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="Galvo", help="ART-DAQ device name.")
    parser.add_argument("--channel", default="ao1", help="AO channel, usually ao1 for Y.")
    parser.add_argument("--rate", type=float, default=1000.0, help="Sample rate in Hz.")
    parser.add_argument("--freq", type=float, default=10.0, help="Sine frequency in Hz.")
    parser.add_argument("--duration", type=float, default=10.0, help="Output duration in seconds.")
    parser.add_argument("--amp", type=float, default=1.0, help="Sine amplitude in volts.")
    parser.add_argument("--offset", type=float, default=0.0, help="DC offset in volts.")
    args = parser.parse_args()
    run_test(args.device, args.channel, args.rate, args.freq, args.duration, args.amp, args.offset)


if __name__ == "__main__":
    main()
