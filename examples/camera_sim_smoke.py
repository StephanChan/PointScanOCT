"""示例：SIM 相机 → OctProcessor 全链路（完全不需要硬件）。

用法：
    python examples/camera_sim_smoke.py

结构就是 M4 ``OctEngine`` 将来要做的事：相机在采集线程里交出一包原始光谱，
回调里把它交给处理层，拿到深度图。
"""

import os
import sys
import time

import numpy as np

try:
    import octsdk
except ModuleNotFoundError:  # 直接在仓库里跑示例时，把仓库根目录加进 sys.path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import octsdk

SAMPLES = 256
X_PIXELS = 64
Y_LINES = 8
PACKAGE_Y_LINES = 2


def build_config():
    return octsdk.OctConfig(
        scan=octsdk.ScanConfig(
            x_step_um=10.0, x_length_um=float(X_PIXELS) * 10.0, y_lines=Y_LINES
        ),
        processing=octsdk.ProcessingConfig(
            raw_spectral_samples=SAMPLES,
            aline_avg=1,
            depth_start=8,  # 相机输出非负计数 → 深度 0 有强 DC，按真实用法裁掉
            amplification=1.0,
        ),
        camera=octsdk.CameraConfig(simulate=True),
    )


def main():
    config = build_config()
    processing = config.resolved_processing()
    processor = octsdk.OctProcessor(processing, backend="cpu")
    processor.prepare()

    results = []

    def on_package(package):
        result = processor.process(package.spectra, package_start=package.package_start)
        results.append(result)
        peak = int(np.argmax(result.data.mean(axis=0))) + int(processing.depth_start)
        print(
            f"package start={package.package_start:5d} rows={package.rows:4d} "
            f"-> depth={result.data.shape} peak_depth={peak} "
            f"({result.timings['total_s'] * 1e3:.2f} ms)"
        )

    camera = octsdk.open_camera(
        config.camera, pace=False, target_depth_fraction=0.25, seed=3
    )
    camera.open()
    camera.configure(rows_per_bline=processing.x_pixels, spectral_samples=SAMPLES)
    camera.start(
        on_package,
        x_pixels=processing.x_pixels,
        bline_avg=processing.bline_avg,
        package_y_lines=PACKAGE_Y_LINES,
        y_lines=Y_LINES,
    )
    deadline = time.time() + 5.0
    while len(results) < 3 and time.time() < deadline:
        time.sleep(0.01)
    camera.stop()
    camera.close()

    print(f"packages={len(results)} rows={sum(item.rows for item in results)}")
    print(f"expected peak depth = {SAMPLES // 4}")
    print("camera:", {k: v for k, v in camera.describe().items() if k != "effective"})


if __name__ == "__main__":
    main()
