"""示例：``OctEngine`` 端到端（SIM 相机 + SIM AODO + CPU 处理 + TIFF 落盘）。

用法：
    python examples/engine_sim_smoke.py [输出目录]
"""

import os
import sys
import tempfile

import numpy as np

try:
    import octsdk
except ModuleNotFoundError:  # 直接在仓库里跑示例时，把仓库根目录加进 sys.path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import octsdk

MM_PER_VOLT = 2.3776  # 0.66X 物镜实测换算，换成你自己的标定值


def build_config(y_lines=8):
    return octsdk.OctConfig(
        scan=octsdk.ScanConfig(
            x_step_um=10.0,
            x_length_um=160.0,  # 16 个振镜位置
            y_step_um=10.0,
            y_lines=y_lines,
            flyback_samples=16,
        ),
        processing=octsdk.ProcessingConfig(
            raw_spectral_samples=256,
            aline_avg=1,
            bline_avg=1,
            depth_start=8,  # 相机输出非负计数 → 裁掉深度 0 的 DC
            amplification=1.0,
            y_notch_enabled=True,
        ),
        camera=octsdk.CameraConfig(simulate=True),
        aodo=octsdk.AodoConfig(
            simulate=True, x_mm_per_volt=MM_PER_VOLT, y_mm_per_volt=MM_PER_VOLT
        ),
        engine=octsdk.EngineConfig(package_y_lines=4),
    )


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp(prefix="octsdk_engine_")
    config = build_config()
    processing = config.resolved_processing()
    engine = octsdk.OctEngine(
        config,
        backend="cpu",
        camera_options={"pace": False, "target_depth_fraction": 0.25, "seed": 7},
    )
    with engine:
        bscan = engine.single_scan(octsdk.AcqModes.FINITE_BSCAN)
        peak = int(np.argmax(bscan.bline)) + int(processing.depth_start)
        print(
            f"B-scan : rows={bscan.rows} packages={bscan.packages} "
            f"bline={bscan.bline.shape} peak_depth={peak} "
            f"({bscan.timings['wall_s'] * 1e3:.1f} ms)"
        )

        cscan_path = os.path.join(out_dir, "cscan.tif")
        cscan = engine.single_scan(octsdk.AcqModes.FINITE_CSCAN, save_path=cscan_path)
        info = cscan.describe()
        print(
            f"C-scan : volume={info['volume_shape']} aip={info['aip_shape']} "
            f"filled_y={info['filled_y']} y_notch={info['y_notch_applied']} "
            f"({cscan.timings['wall_s'] * 1e3:.1f} ms)"
        )

        repeated = engine.repeat_scan(octsdk.AcqModes.CONTINUOUS_BLINE, max_packages=3)
        print(
            f"repeat : packages={repeated.packages} rows={repeated.rows} "
            f"bline={None if repeated.bline is None else repeated.bline.shape}"
        )

    print(f"saved  : {cscan_path} ({os.path.getsize(cscan_path)} bytes)")


if __name__ == "__main__":
    main()
