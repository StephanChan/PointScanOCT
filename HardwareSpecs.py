# -*- coding: utf-8 -*-
"""Central hardware specifications for the LineScanOCT control software.

Keep physical hardware constants here instead of scattering them through GUI,
waveform, and thread-control code.
"""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ObjectiveSpec:
    name: str
    angle_to_mm_ratio: float
    camera_step_divisor: float
    max_y_fov_mm: float


@dataclass(frozen=True)
class CameraSpec:
    name: str
    pixel_size_um: float
    spectral_samples_px: int

    @property
    def max_height_px(self):
        return self.spectral_samples_px


@dataclass(frozen=True)
class StageAxisSpec:
    axis_index: int
    init_speed_mm_s: float


SYSTEM_MAGNIFICATION_4X = 4


# ---- Galvo / scan optics calibration -----------------------------------------
# 电压→角度：1 度/V（振镜机械角）。若厂家规格给的是"光束角"（镜面转 φ 光束偏 2φ），
# 把这里改成 0.5，全表自动重算。
GALVO_MECH_DEG_PER_VOLT = 1.0
# 扫描透镜焦距（mm）。振镜在扫描透镜前焦面 + 200 mm tube lens 的 telecentric 结构下：
#     Δx_sample = (f_scan / M_obj) · θ_beam,   θ_beam = 2φ,   φ = K · U
#     ⇒ angle_to_mm_ratio [mm/V] = (f_scan / M) · 2 · (π/180) · K
# 实测扫描透镜 f = 85 mm；若某路用不同焦距的扫描镜，改对应常量即可。
SCAN_LENS_MM_HIGH_MAG = 85.0
SCAN_LENS_MM_LOW_MAG = 85.0


# 实测微调系数（实际扫描长度 / 设定长度）：物理换算之后再乘这个数。
# 10X：设定 90 µm、实测 78 µm ⇒ 78/90；
# 0.66X：设定 9454 µm(9.454 mm)、实测 2.5 mm ⇒ 2.5/9.454；其它物镜实测后同样填在这里即可。
GALVO_RATIO_MEASURED_FACTOR = {
    "0.66X": 2.5 / 9.454,
    "4X": 1.0,
    "5X": 1.0,
    "10X": 78.0 / 90.0,
    "20X": 1.0,
}

# （轴向像素尺寸常量统一维护在下方 AXIAL_PIXEL_SIZE_UM 处，此处不再重复定义）

def galvo_mm_per_volt(
    objective_magnification,
    scan_lens_mm=SCAN_LENS_MM_HIGH_MAG,
    mech_deg_per_volt=GALVO_MECH_DEG_PER_VOLT,
):
    """样品端位移/电压 [mm/V]（偏置约定：offset_mm = ratio · V）。"""
    magnification = float(objective_magnification)
    if magnification <= 0.0:
        raise ValueError(f"objective magnification must be positive: {objective_magnification}")
    return (
        (float(scan_lens_mm) / magnification)
        * 4.0
        * (math.pi / 180.0)
        * float(mech_deg_per_volt)
    )


OBJECTIVE_SPECS = {
    "0.66X": ObjectiveSpec(
        name="0.66X",
        angle_to_mm_ratio=galvo_mm_per_volt(0.66, SCAN_LENS_MM_LOW_MAG) * GALVO_RATIO_MEASURED_FACTOR["0.66X"],  # ≈2.3776（实测 9.454 mm→2.5 mm 修正）
        camera_step_divisor=0.66 / 4.0,
        max_y_fov_mm=5.0 * 4.0 / 0.66,
    ),
    "4X": ObjectiveSpec(
        name="4X",
        angle_to_mm_ratio=galvo_mm_per_volt(4.0, SCAN_LENS_MM_LOW_MAG),  # ≈0.3665 mm/V
        camera_step_divisor=1.0,
        max_y_fov_mm=5.0,
    ),
    "5X": ObjectiveSpec(
        name="5X",
        angle_to_mm_ratio=galvo_mm_per_volt(5.0),  # 50 mm scan lens, K=1 deg/V -> 0.3491 mm/V
        camera_step_divisor=1.25,
        max_y_fov_mm=4.0,
    ),
    "10X": ObjectiveSpec(
        name="10X",
        angle_to_mm_ratio=galvo_mm_per_volt(10.0) * GALVO_RATIO_MEASURED_FACTOR["10X"],  # ≈0.5143（实测 90 µm→78 µm 修正）
        camera_step_divisor=2.5,
        max_y_fov_mm=2.0,
    ),
    "20X": ObjectiveSpec(
        name="20X",
        angle_to_mm_ratio=galvo_mm_per_volt(20.0),  # 50 mm scan lens, K=1 deg/V -> 0.0873 mm/V
        camera_step_divisor=5.0,
        max_y_fov_mm=1.0,
    ),
}


CAMERA_SPECS = {
    "PhotonFocus": CameraSpec(name="PhotonFocus", pixel_size_um=9.0, spectral_samples_px=1100),
    "XingTu": CameraSpec(name="XingTu", pixel_size_um=6.5, spectral_samples_px=1024),
    "Daheng": CameraSpec(name="Daheng", pixel_size_um=9.0, spectral_samples_px=1600),
    "AiKe": CameraSpec(name="AiKe", pixel_size_um=7.0, spectral_samples_px=4096),
    "AK": CameraSpec(name="AK", pixel_size_um=7.0, spectral_samples_px=4096),
    "IKap": CameraSpec(name="IKap", pixel_size_um=7.0, spectral_samples_px=4096),
    "PL4KCL-200KT": CameraSpec(name="PL4KCL-200KT", pixel_size_um=7.0, spectral_samples_px=4096),
}


STAGE_AXIS_SPECS = {
    "X": StageAxisSpec(axis_index=0, init_speed_mm_s=1.0),
    "Y": StageAxisSpec(axis_index=1, init_speed_mm_s=1.0),
    "Z": StageAxisSpec(axis_index=2, init_speed_mm_s=0.1),
}


AODO_TRIGGER_OUT_PFI = "PFI3"
AODO_TRIGGER_IN_PFI = "PFI7"
AODO_DEFAULT_FRAME_RATE = 400
AODO_AO_VOLTAGE_MIN = -10.0
AODO_AO_VOLTAGE_MAX = 10.0

# 轴向（深度）采样尺度：6.6 µm 为"空气等效/光学路径"步长，即每像素对应的光程长度；
# 在水中折射率 n≈1.32~1.33 时换算成真实深度 ≈ 6.6/n ≈ 5.0 µm/px。
AXIAL_PIXEL_SIZE_UM = 6.6
WATER_REFRACTIVE_INDEX = 1.33
AXIAL_PIXEL_SIZE_UM_IN_WATER = AXIAL_PIXEL_SIZE_UM / WATER_REFRACTIVE_INDEX  # ≈4.96 µm
# 兼容旧名：历史上用 4.4，现已统一为 AXIAL_PIXEL_SIZE_UM（6.6 空气等效 / 5.0 水中深度）
DEFAULT_AXIAL_PIXEL_SIZE_UM = AXIAL_PIXEL_SIZE_UM
Axial_pixel_size = AXIAL_PIXEL_SIZE_UM  # 兼容刚加入的写法

PHOTONFOCUS_STATIC_NORMALIZATION_MEAN = 2048.0
GPU_DEFAULT_STATIC_NORMALIZATION_MEAN = 40000.0
GPU_STATIC_NORMALIZATION_EPS = 1e-3
GPU_BACKGROUND_X_NORMALIZATION_EPS = 1e-3
GPU_BACKGROUND_X_NORMALIZATION_ROOT_ORDER = 2.0
GPU_PRE_FFT_LOG_DEVIATION_THRESHOLD_PCT = 1.0
GPU_PROFILE_TIMING_DEFAULT = False
TIFF_APPEND_WRITES_DEFAULT = False
# Classic TIFF stores file offsets in 32 bit fields, so a stack file that grows past
# 4 GiB makes tifffile fail with "struct.error: argument out of range". New stack files
# are therefore created as BigTIFF (64 bit offsets). Set to False if a downstream tool
# cannot read BigTIFF - but then keep the stack files below ~4 GiB.
TIFF_BIGTIFF_DEFAULT = True


def get_objective_spec(name):
    return OBJECTIVE_SPECS.get(str(name))


def get_camera_spec(name):
    return CAMERA_SPECS.get(str(name))


def get_stage_axis_spec(axis):
    return STAGE_AXIS_SPECS[str(axis)]


def get_laser_spec(name):
    return LASER_SPECS.get(str(name))


def camera_step_size_um(camera_name, objective_name):
    camera = get_camera_spec(camera_name)
    objective = get_objective_spec(objective_name)
    if camera is None:
        raise KeyError(f"Unknown camera: {camera_name}")
    if objective is None:
        raise KeyError(f"Unknown objective: {objective_name}")
    return camera.pixel_size_um / SYSTEM_MAGNIFICATION_4X / objective.camera_step_divisor


def digital_line_mask(line_name):
    try:
        line_number = int(str(line_name).split("line")[-1])
    except (TypeError, ValueError):
        raise KeyError(f"Unsupported digital line: {line_name}")
    if line_number < 0 or line_number > 31:
        raise KeyError(f"Unsupported digital line: {line_name}")
    return 1 << line_number
