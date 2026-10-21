"""最小示例：不需要任何硬件，用合成光谱跑一遍 ``OctProcessor``（CPU 与 GPU 各一次）。

用法：
    python examples/processing_smoke.py
"""

import os
import sys

import numpy as np

try:
    import octsdk
except ModuleNotFoundError:  # 直接在仓库里跑示例时，把仓库根目录加进 sys.path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import octsdk


def synthetic_spectra(rows, samples, depth_index, amplitude=10.0, seed=0):
    """合成一包光谱：随机噪声 + 一条余弦调制，FFT 后应在 ``depth_index`` 处出现峰。"""
    rng = np.random.default_rng(seed)
    spectra = rng.normal(0.0, 1.0, (rows, samples)).astype(np.float32)
    axis = np.arange(samples, dtype=np.float32)
    carrier = np.cos(2.0 * np.pi * float(depth_index) * axis / float(samples))
    return spectra + amplitude * carrier[np.newaxis, :]


def main():
    samples = 256
    depth_index = samples // 4
    config = build_processing_config(samples)
    spectra = synthetic_spectra(int(config.x_pixels), samples, depth_index)

    for backend in ("cpu", "gpu"):
        try:
            processor = octsdk.OctProcessor(config, backend=backend)
        except octsdk.BackendUnavailableError as error:
            print(f"{backend}: skipped ({error})")
            continue
        processor.prepare()
        processor.process(spectra)  # 首次调用包含 CUDA 上下文初始化，先热身一次
        package = processor.process(spectra)
        peak = int(np.argmax(package.data.mean(axis=0)))
        print(
            f"{backend}: shape={package.data.shape} dtype={package.data.dtype} "
            f"peak_depth={peak} (expected {depth_index}) "
            f"time={package.timings['total_s'] * 1e3:.2f} ms"
        )

    print("cupy:", octsdk.environment_report()["cupy"])


def build_processing_config(samples):
    return octsdk.OctConfig(
        scan=octsdk.ScanConfig(x_step_um=10.0, x_length_um=40.0),
        processing=octsdk.ProcessingConfig(
            raw_spectral_samples=samples,
            x_pixels=4,
            amplification=1.0,
            output_dtype="float32",
        ),
    ).resolved_processing()


if __name__ == "__main__":
    main()
