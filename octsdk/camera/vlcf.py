"""``.vlcf`` / ``.ccf`` 预设解析与路径解析（纯文件逻辑，不需要相机）。

* ``.vlcf``：板卡侧权威配置（位深、tap、ROI 宽度/偏移、行触发源…），由 IKap 板卡加载。
* ``.ccf`` ：相机侧配置（``SynchronizationMode`` 等），由相机加载；约定与 ``.vlcf`` 同名。

这三个层级的优先级（与主程序 ``ThreadCamera_AK.ResolveVlcfPath`` 一致）：
显式 ``vlcf_path`` > ``preset`` 名 > 按 ``(bit_depth, taps)`` 选预设。
"""

from pathlib import Path

from ..errors import HardwareError

# .vlcf 中决定板卡行为的字段（板卡设置的权威来源，与主程序同一份清单）。
VLCF_KEYS_OF_INTEREST = (
    "PIXEL_DEPTH",
    "TAP_NUMBER",
    "TAP_ARRANGEMENT",
    "PIXEL_CLOCK",
    "IMAGE_WIDTH",
    "IMAGE_HEIGHT",
    "IMAGE_TYPE",
    "IMAGE_OFFSET_X",
    "CC1_SOURCE",
    "INTEGRATION_TRIGGER_SOURCE",
    "INTEGRATION_METHOD",
    "SHAFT_ENCODER_CHANNEL",
    "SHAFT_ENCODER_MULTIPLY_FACTOR",
    "GENERAL_INPUT1_SAMPLE_MODE",
    "GENERALINPUT1_TRIGGER_MODE",
    "GRAB_MODE",
    "TIMEOUT",
    "INTERNAL_TRIGGER_FREQUENCY",
    "PIXEL_FORMAT_PFNC",
)

#: ``(位深, tap 数) -> 随包携带的 .vlcf``
PRESET_FILES = {
    (8, 8): "8bit8tap.vlcf",
    (8, 10): "8bit10tap.vlcf",
    (10, 8): "10bit8tap.vlcf",
    (12, 4): "12bit4tap.vlcf",
}

#: ``INTEGRATION_TRIGGER_SOURCE`` 取值 -> 名字（与 IKap 文档一致）
TRIGGER_SOURCE_NAMES = {
    0: "Internal",
    1: "GeneralInput1",
    2: "GeneralInput2",
    5: "ShaftEncoder1",
    7: "BoardSync1",
    8: "BoardSync2",
    9: "Software",
    10: "CameraSync",
}


def presets_dir():
    return Path(__file__).resolve().with_name("presets")


def available_presets():
    return sorted(path.name for path in presets_dir().glob("*.vlcf"))


def read_vlcf_settings(vlcf_path):
    """读取 ``.vlcf`` 中关心的字段，返回 ``{大写字段名: 去空白的字符串}``。"""
    settings = {}
    path = Path(str(vlcf_path or ""))
    if not path.is_file():
        return settings
    wanted = {key.upper() for key in VLCF_KEYS_OF_INTEREST}
    with open(path, "r", encoding="latin-1") as handle:
        for line in handle:
            if "=" not in line:
                continue
            name, _, value = line.partition("=")
            name = name.strip().upper()
            if name in wanted:
                settings[name] = value.strip()
    return settings


def vlcf_int(settings, key, default=None):
    """取整数值（兼容 ``9000.0`` 这类写法）。"""
    value = settings.get(str(key).upper())
    if value is None:
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def vlcf_float(settings, key, default=None):
    value = settings.get(str(key).upper())
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def preset_path(bit_depth, taps):
    """按位深与 tap 数在随包预设里查找；没有对应预设返回 ``None``。"""
    filename = PRESET_FILES.get((int(bit_depth), int(taps)))
    if filename is None:
        return None
    path = presets_dir() / filename
    return path if path.is_file() else None


def resolve_vlcf_path(config):
    """解析要加载的 ``.vlcf``；不可用时抛 :class:`HardwareError`（不静默降级）。"""
    explicit = str(getattr(config, "vlcf_path", "") or "").strip()
    if explicit:
        path = Path(explicit)
        if not path.is_file():
            raise HardwareError(f"vlcf file not found: {path}")
        return path
    preset = str(getattr(config, "preset", "") or "").strip()
    if preset:
        name = preset if preset.lower().endswith(".vlcf") else f"{preset}.vlcf"
        path = presets_dir() / name
        if not path.is_file():
            raise HardwareError(
                f"vlcf preset not found: {path}; available: {', '.join(available_presets())}"
            )
        return path
    path = preset_path(getattr(config, "bit_depth", 0), getattr(config, "taps", 0))
    if path is None:
        raise HardwareError(
            "no .vlcf for bit_depth={} taps={}; set CameraConfig.vlcf_path or CameraConfig.preset "
            "(available: {})".format(
                getattr(config, "bit_depth", None),
                getattr(config, "taps", None),
                ", ".join(available_presets()),
            )
        )
    return path


def resolve_ccf_path(vlcf_path, config=None):
    """解析相机侧 ``.ccf``：显式路径优先，否则取与 ``.vlcf`` 同名的文件（没有则 ``None``）。"""
    explicit = ""
    if config is not None:
        explicit = str(getattr(config, "ccf_path", "") or "").strip()
    if explicit:
        path = Path(explicit)
        if not path.is_file():
            raise HardwareError(f"ccf file not found: {path}")
        return path
    candidate = Path(vlcf_path).with_suffix(".ccf")
    return candidate if candidate.is_file() else None


def describe_vlcf(settings):
    """把解析出的字段整理成可读字典（写日志、``describe()`` 用）。"""
    trigger_value = vlcf_int(settings, "INTEGRATION_TRIGGER_SOURCE")
    return {
        "pixel_depth": vlcf_int(settings, "PIXEL_DEPTH"),
        "taps": vlcf_int(settings, "TAP_NUMBER"),
        "image_width": vlcf_int(settings, "IMAGE_WIDTH"),
        "image_height": vlcf_int(settings, "IMAGE_HEIGHT"),
        "offset_x": vlcf_int(settings, "IMAGE_OFFSET_X"),
        "pixel_clock": vlcf_float(settings, "PIXEL_CLOCK"),
        "grab_mode": vlcf_int(settings, "GRAB_MODE"),
        "internal_trigger_hz": vlcf_float(settings, "INTERNAL_TRIGGER_FREQUENCY"),
        "trigger_source_id": trigger_value,
        "trigger_source": (
            TRIGGER_SOURCE_NAMES.get(int(trigger_value), "?")
            if trigger_value is not None
            else None
        ),
        "pixel_format_pfnc": vlcf_int(settings, "PIXEL_FORMAT_PFNC"),
    }


def load_vlcf(config):
    """解析并读取 ``.vlcf``（同时给出配对的 ``.ccf``），返回 ``(vlcf, settings, ccf)``。"""
    vlcf_path = resolve_vlcf_path(config)
    settings = read_vlcf_settings(vlcf_path)
    if not settings:
        raise HardwareError(f"vlcf could not be parsed (empty settings): {vlcf_path}")
    return vlcf_path, settings, resolve_ccf_path(vlcf_path, config)
