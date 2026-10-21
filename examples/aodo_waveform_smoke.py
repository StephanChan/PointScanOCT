"""示例：生成 B-scan / C-scan 的振镜 AO 波形与相机行触发 DO（不需要 DAQ）。

用法：
    python examples/aodo_waveform_smoke.py
"""

import os
import sys

try:
    import octsdk
except ModuleNotFoundError:  # 直接在仓库里跑示例时，把仓库根目录加进 sys.path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import octsdk

# 0.66X 物镜实测换算（mm/V）；换成你的物镜标定值即可
MM_PER_VOLT = 2.3776


def build_config():
    return octsdk.OctConfig(
        scan=octsdk.ScanConfig(
            x_step_um=10.0, x_length_um=200.0, y_step_um=10.0, y_lines=20,
            flyback_samples=50,
        ),
        processing=octsdk.ProcessingConfig(
            raw_spectral_samples=1440, aline_avg=1, bline_avg=1
        ),
        aodo=octsdk.AodoConfig(x_mm_per_volt=MM_PER_VOLT, y_mm_per_volt=MM_PER_VOLT),
    )


def main():
    config = build_config()
    processing = config.resolved_processing()
    sample_rate = 2.0 * config.camera.line_rate_hz

    for mode in (octsdk.AcqModes.FINITE_BSCAN, octsdk.AcqModes.FINITE_CSCAN):
        spec = octsdk.build_waveform(config.scan, processing, config.aodo, mode, sample_rate)
        info = spec.describe()
        print(
            f"{mode:14s} samples={info['samples']:7d} line={info['line_samples']:6d} "
            f"duration={info['duration_s']:.3f}s "
            f"X={info['x_voltage_range'][0]:+.4f}..{info['x_voltage_range'][1]:+.4f}V "
            f"Y={info['y_voltage_range'][0]:+.4f}..{info['y_voltage_range'][1]:+.4f}V "
            f"do_pulses={info['do_pulses']}"
        )

    simulated = octsdk.AodoConfig(
        x_mm_per_volt=MM_PER_VOLT, y_mm_per_volt=MM_PER_VOLT, simulate=True
    )
    device = octsdk.open_aodo(simulated, camera_config=config.camera)
    device.open()
    report = device.configure(config.scan, processing, octsdk.AcqModes.FINITE_CSCAN)
    device.start()
    device.stop()
    device.close()
    print(
        "SIM aodo: samples={samples} do_pulses={do_pulses} rate={sample_rate_hz:g} Hz "
        "duration={duration_s:.3f}s".format(**report)
    )


if __name__ == "__main__":
    main()
