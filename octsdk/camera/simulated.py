"""SIM 相机：不需要任何硬件，生成带点目标与噪声的 12-bit 光谱。

用途：
* 在没有相机的机器上验证包切分、回调契约、与 :class:`octsdk.OctProcessor` 的联调；
* 工厂回归/客户演示。

约定与真实相机一致：一帧 = 一条 B-line（``rows_per_bline`` 行 × ``spectral_samples``），
dtype ``uint16``，有效位深 12 bit（0–4095）。

注意：输出是**非负计数**，所以 FFT 在深度 0 一定有一个强 DC。想看到点目标的深度峰，
需要像真实采集那样用 ``ProcessingConfig.depth_start`` 把 DC 附近裁掉。
"""

import time

import numpy as np

from .base import CameraDeviceBase


class SimulatedCameraDevice(CameraDeviceBase):
    """合成光谱相机。"""

    def __init__(
        self,
        config,
        *,
        target_depth_fraction=0.25,
        target_amplitude=10.0,
        noise_sigma=1.0,
        counts_scale=200.0,
        pace=True,
        seed=0,
        status_callback=None,
    ):
        super().__init__(config, status_callback=status_callback, logger_name="camera.sim")
        if not 0.0 < float(target_depth_fraction) < 1.0:
            raise ValueError(f"target_depth_fraction must be within (0, 1): {target_depth_fraction}")
        self.target_depth_fraction = float(target_depth_fraction)
        self.target_amplitude = float(target_amplitude)
        self.noise_sigma = float(noise_sigma)
        self.counts_scale = float(counts_scale)
        self.pace = bool(pace)
        self._rng = np.random.default_rng(int(seed))
        self._carrier = None
        self._frame_index = 0
        self._axis = np.empty(0, dtype=np.float32)

    # -- 钩子 -------------------------------------------------------------- #
    def _open(self):
        return {"simulate": True, "seed": int(self._rng.bit_generator.seed_seq.entropy)}

    def _configure(self, rows_per_bline, spectral_samples):
        axis = np.arange(spectral_samples, dtype=np.float32)
        depth_index = int(round(self.target_depth_fraction * spectral_samples))
        # 余弦在 FFT 后正好落在 depth_index 处（与 K空间线性轴一致）
        self._axis = (2.0 * np.pi * depth_index * axis / float(spectral_samples)).astype(
            np.float32
        )
        self._carrier = self.target_amplitude * np.cos(self._axis)
        self._frame_index = 0
        return {
            "rows_per_bline": int(rows_per_bline),
            "spectral_samples": int(spectral_samples),
            "target_depth_index": int(depth_index),
            "line_rate_hz": float(self.config.line_rate_hz),
            "pace_frames": self.pace,
        }

    def _start_stream(self):
        self._frame_index = 0

    def _stop_stream(self):
        pass

    def _next_frame(self):
        if self.pace:
            # 一帧 = 一条 B-line，按行频给它应有的时长
            rows = max(1, int(self._rows_per_bline))
            time.sleep(rows / max(1.0, float(self.config.line_rate_hz)))
        rows = max(1, int(self._rows_per_bline))
        noise = self._rng.normal(0.0, self.noise_sigma, (rows, self._spectral_samples))
        # 每帧乘一个缓慢漂移的强度，模拟光源抖动
        drift = 1.0 + 0.02 * np.sin(2.0 * np.pi * self._frame_index / 64.0)
        signal = (noise + drift * self._carrier[np.newaxis, :]) * self.counts_scale
        self._frame_index += 1
        max_count = float((1 << int(self.config.bit_depth)) - 1)
        return np.clip(signal, 0.0, min(max_count, 65535.0)).astype(np.uint16)

    def _describe_extra(self):
        return {
            "target_depth_fraction": self.target_depth_fraction,
            "target_amplitude": self.target_amplitude,
            "noise_sigma": self.noise_sigma,
            "counts_scale": self.counts_scale,
            "pace_frames": self.pace,
        }
