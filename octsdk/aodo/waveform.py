"""振镜 AO 波形与相机行触发 DO 波形（纯函数，无 Qt / 无 DAQ 依赖）。

与主程序 ``Generaic_functions.GenAODO`` 同一套几何与电压换算，唯一区别：主程序按物镜
名字查 ``HardwareSpecs`` 的 ``angle_to_mm_ratio``，SDK 直接收 ``AodoConfig.x_mm_per_volt``
/ ``y_mm_per_volt``（mm/V），方便客户自带标定。

通道与语义：
* ``ao[0]`` = X（快轴，一条 B-line 内扫描），``ao[1]`` = Y（慢轴，C-scan 逐行）；
* ``do`` 是相机行触发，只在 X 正向扫描段为高，每个振镜位置 ``AlineAVG`` 个脉冲；
* 每个 AO 采样点对应 ``2`` 个像素时钟（主程序约定），所以
  ``line_samples = 2 × x_pixels + flyback_samples``。
"""

from dataclasses import dataclass

import numpy as np

from ..config import BLINE_MODES, AcqModes, AodoConfig, ProcessingConfig, ScanConfig
from ..errors import ConfigurationError

SUPPORTED_MODES = (
    AcqModes.FINITE_BSCAN,
    AcqModes.CONTINUOUS_BLINE,
    AcqModes.X_FREQUENCY,
    AcqModes.FINITE_CSCAN,
    AcqModes.CONTINUOUS_CSCAN,
)


@dataclass(frozen=True)
class WaveformSpec:
    """一次扫描需要的整段波形。"""

    ao: np.ndarray
    do: np.ndarray
    sample_rate_hz: float
    line_samples: int
    y_steps: int
    x_positions: int
    x_pixels: int
    mode: str

    @property
    def samples(self):
        return int(self.ao.shape[1])

    @property
    def duration_s(self):
        return self.samples / self.sample_rate_hz if self.sample_rate_hz > 0 else 0.0

    def describe(self):
        return {
            "mode": self.mode,
            "samples": self.samples,
            "line_samples": int(self.line_samples),
            "y_steps": int(self.y_steps),
            "x_positions": int(self.x_positions),
            "x_pixels": int(self.x_pixels),
            "sample_rate_hz": float(self.sample_rate_hz),
            "duration_s": float(self.duration_s),
            "x_voltage_range": [float(self.ao[0].min()), float(self.ao[0].max())],
            "y_voltage_range": [float(self.ao[1].min()), float(self.ao[1].max())],
            "do_pulses": int(np.count_nonzero(self.do)),
        }


def scan_voltage_limits(length_mm, mm_per_volt, bias_v=0.0):
    """扫描长度 → 电压区间；与主程序一致：``Vmax=(L/2)/ratio/2+bias``。"""
    if float(mm_per_volt) <= 0:
        raise ConfigurationError(f"mm_per_volt must be positive: {mm_per_volt}")
    half_span = (float(length_mm) / 2.0) / float(mm_per_volt) / 2.0
    return float(bias_v) - half_span, float(bias_v) + half_span


def camera_trigger_train(trigger_count):
    """返回 ``2 × trigger_count`` 个采样的 DO 序列（0,1,0,1,…）。"""
    trigger_count = max(1, int(trigger_count))
    waveform = np.zeros((trigger_count, 2), dtype=np.uint32)
    waveform[:, 1] = 1
    return waveform.flatten()


def cos_flyback(v_from, v_to, samples):
    """余弦回扫（位置从 ``v_from`` 平滑走到 ``v_to``）。"""
    samples = max(1, int(samples))
    theta = np.linspace(0.0, np.pi, samples, endpoint=False)
    return (float(v_from) - float(v_to)) / 2.0 * np.cos(theta) + (
        float(v_from) + float(v_to)
    ) / 2.0


def build_waveform(scan, processing, aodo, mode, sample_rate_hz=None):
    """按扫描几何生成 ``(AO, DO)`` 波形，返回 :class:`WaveformSpec`。"""
    if mode not in SUPPORTED_MODES:
        raise ConfigurationError(
            f"unsupported acquisition mode for AODO: {mode!r} (expected one of {SUPPORTED_MODES})"
        )
    if aodo.x_mm_per_volt is None or aodo.y_mm_per_volt is None:
        raise ConfigurationError(
            "AodoConfig.x_mm_per_volt / y_mm_per_volt must be set before building the waveform"
        )
    rate = float(sample_rate_hz if sample_rate_hz is not None else (aodo.sample_rate_hz or 0.0))
    if rate <= 0:
        raise ConfigurationError(
            "sample rate must be resolved (usually 2 × line_rate_hz) before building the waveform"
        )

    x_positions = int(scan.x_positions)
    aline_avg = max(1, int(processing.aline_avg))
    bline_repeat = max(1, int(processing.bline_avg))
    x_pixels = x_positions * aline_avg
    postclocks = max(1, int(scan.flyback_samples))
    y_steps = max(1, int(scan.y_lines))

    x_length_mm = float(scan.x_step_um) * x_positions / 1000.0
    y_length_mm = float(scan.y_step_um) * y_steps / 1000.0
    x_min, x_max = scan_voltage_limits(x_length_mm, aodo.x_mm_per_volt, aodo.x_voltage_bias)
    y_min, y_max = scan_voltage_limits(y_length_mm, aodo.y_mm_per_volt, aodo.y_voltage_bias)
    limit = float(aodo.ao_voltage_limit_v)
    worst = max(abs(x_min), abs(x_max), abs(y_min), abs(y_max))
    if worst > limit:
        raise ConfigurationError(
            f"scan needs {worst:.3f} V but ao_voltage_limit_v={limit:.3f} V; "
            "reduce the scan length or raise the limit"
        )

    x_forward = np.repeat(np.linspace(x_min, x_max, x_positions), 2 * aline_avg)
    x_back = cos_flyback(x_max, x_min, postclocks)
    do_forward = camera_trigger_train(x_positions * aline_avg)
    do_off = np.zeros(postclocks, dtype=np.uint32)

    x_segments = []
    y_segments = []
    do_segments = []
    if mode in BLINE_MODES:
        # 单条 B-line：Y 保持偏置，重复 BlineAVG 次
        for _ in range(bline_repeat):
            x_segments.extend([x_forward, x_back])
            y_segments.extend(
                [
                    np.full(x_forward.size, float(aodo.y_voltage_bias)),
                    np.full(postclocks, float(aodo.y_voltage_bias)),
                ]
            )
            do_segments.extend([do_forward, do_off])
    else:
        y_positions = np.linspace(y_min, y_max, y_steps)
        for row, y_pos in enumerate(y_positions):
            next_y = y_positions[row + 1] if row + 1 < y_steps else y_positions[0]
            for repeat in range(bline_repeat):
                flyback_to = next_y if repeat == bline_repeat - 1 else y_pos
                x_segments.extend([x_forward, x_back])
                y_segments.extend(
                    [
                        np.full(x_forward.size, float(y_pos)),
                        cos_flyback(y_pos, flyback_to, postclocks),
                    ]
                )
                do_segments.extend([do_forward, do_off])

    ao = np.vstack([np.concatenate(x_segments), np.concatenate(y_segments)]).astype(np.float64)
    do = np.concatenate(do_segments).astype(np.uint32)
    return WaveformSpec(
        ao=np.ascontiguousarray(ao),
        do=np.ascontiguousarray(do),
        sample_rate_hz=rate,
        line_samples=int(2 * x_pixels + postclocks),
        y_steps=int(y_steps),
        x_positions=int(x_positions),
        x_pixels=int(x_pixels),
        mode=str(mode),
    )