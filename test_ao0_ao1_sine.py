# -*- coding: utf-8 -*-
"""Standalone ART-DAQ AO0/AO1 synchronized sine-wave test.

Outputs 1 V amplitude, 10 Hz sine waves on ao0 and ao1 for about 10 seconds.
Default device name follows the ASOCT UI setting: "Galvo".
"""

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


def build_sine_pair(sample_rate_hz, frequency_hz, duration_s, amplitude_v):
    sample_count = int(round(float(sample_rate_hz) * float(duration_s)))
    t = np.arange(sample_count, dtype=np.float64) / float(sample_rate_hz)
    sine = float(amplitude_v) * np.sin(2.0 * np.pi * float(frequency_hz) * t)
    return np.vstack([sine, sine]).astype(np.float64)


def run_test(device, sample_rate_hz, frequency_hz, duration_s, amplitude_v):
    ao_channels = f"{device}/ao0:1"
    waveform = build_sine_pair(
        sample_rate_hz=sample_rate_hz,
        frequency_hz=frequency_hz,
        duration_s=duration_s,
        amplitude_v=amplitude_v,
    )

    print(
        "Starting synchronized AO test: "
        f"channels={ao_channels}, shape={waveform.shape}, "
        f"sample_rate={sample_rate_hz:g} Hz, frequency={frequency_hz:g} Hz, "
        f"amplitude={amplitude_v:g} V, duration={duration_s:g} s"
    )

    with ni.Task("ao0_ao1_sine_test") as task:
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
            samps_per_chan=waveform.shape[-1],
        )
        writer = AnalogMultiChannelWriter(task.out_stream, auto_start=False)
        writer.verify_array_shape = False
        written = writer.write_many_sample(np.ascontiguousarray(waveform, dtype=np.float64))
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
    parser.add_argument("--freq", type=float, default=10.0, help="Sine frequency in Hz.")
    parser.add_argument("--duration", type=float, default=10.0, help="Output duration in seconds.")
    parser.add_argument("--amp", type=float, default=1.0, help="Sine amplitude in volts.")
    args = parser.parse_args()

    run_test(
        device=args.device,
        sample_rate_hz=args.rate,
        frequency_hz=args.freq,
        duration_s=args.duration,
        amplitude_v=args.amp,
    )


if __name__ == "__main__":
    main()
