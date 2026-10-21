"""SDK 异常层次。

调用方只需要 `except OctSdkError` 就能兜住所有 SDK 抛出的异常；
细分类型用于区分"配置错 / 标定缺失 / 硬件不可用 / 计算失败"。
"""


class OctSdkError(Exception):
    """SDK 所有异常的基类。"""


class ConfigurationError(OctSdkError):
    """配置不自洽（例如 AlineAVG 不能整除每 B-line 的 A-line 数）。"""


class CalibrationError(OctSdkError):
    """色散/插值/背景等标定数据缺失或与当前光谱采样数不匹配。"""


class HardwareError(OctSdkError):
    """相机或 AODO 板卡不可用、参数写入失败。"""


class BackendUnavailableError(OctSdkError):
    """请求的计算后端不可用（例如没有 cupy 却要求 gpu）。"""


class ProcessingError(OctSdkError):
    """数据形状或数值处理失败。"""
