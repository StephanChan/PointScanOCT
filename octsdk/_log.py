"""统一日志与状态回调。

SDK 内部不使用 `print`：要么走 `logging`，要么调用调用方给的 `status_callback`。
"""

import logging
from typing import Callable, Optional

LOGGER_NAME = "octsdk"

StatusCallback = Callable[[str], None]

_configured = False


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """返回 SDK 命名空间下的 logger（默认只挂 NullHandler，不污染宿主程序）。"""
    global _configured
    if not _configured:
        package_logger = logging.getLogger(LOGGER_NAME)
        if not package_logger.handlers:
            package_logger.addHandler(logging.NullHandler())
        _configured = True
    if name:
        return logging.getLogger(f"{LOGGER_NAME}.{name}")
    return logging.getLogger(LOGGER_NAME)


def configure_logging(level=logging.INFO, handler: Optional[logging.Handler] = None) -> logging.Logger:
    """给宿主程序用的便捷入口：一行把 SDK 日志接到控制台。"""
    logger = get_logger()
    if handler is None:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s"))
    handler.setLevel(level)
    logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def emit_status(callback: Optional[StatusCallback], message: str) -> None:
    """把状态消息同时发给日志与回调（回调异常不能打断采集）。"""
    if message is None:
        return
    text = str(message)
    get_logger().info(text)
    if callback is None:
        return
    try:
        callback(text)
    except Exception:  # 回调是宿主代码，出错只记日志
        get_logger("callback").warning("status callback raised", exc_info=True)
