"""运行环境探测：给客户与售后用的"一眼看清装了什么"。

只做检测与报告，不导入厂商 SDK（导入失败也不会抛异常）。
"""

import os
import platform
import shutil
import sys

IKAP_ROOT_ENV = "IKAP_ROOT"
IKAP_DEFAULT_ROOT = r"C:\Program Files\I-TEK OptoElectronics\IKapLibrary"
ARTDAQ_LIB_ENV = "ARTDAQ_PYTHON_LIB"
ARTDAQ_DEFAULT_LIB = r"C:\Program Files (x86)\ART Technology\ART-DAQ\Samples\Python\LIB"


def _module_version(name):
    try:
        module = __import__(name)
    except Exception as error:  # 缺包是正常情况，不抛
        return {"available": False, "error": str(error)}
    return {"available": True, "version": getattr(module, "__version__", "unknown")}


def _cupy_report():
    report = _module_version("cupy")
    if not report["available"]:
        return report
    try:
        import cupy

        count = int(cupy.cuda.runtime.getDeviceCount())
        devices = []
        for index in range(count):
            props = cupy.cuda.runtime.getDeviceProperties(index)
            name = props["name"]
            if isinstance(name, bytes):
                name = name.decode("utf-8", "ignore")
            devices.append({"index": index, "name": name,
                            "total_memory_gb": round(props["totalGlobalMem"] / 1024**3, 1)})
        report["devices"] = devices
    except Exception as error:
        report["cuda_error"] = str(error)
    return report


def _vendor_report(env_name, default_path, hint):
    path = os.environ.get(env_name) or default_path
    return {
        "env": env_name,
        "path": path,
        "exists": os.path.isdir(path),
        "hint": hint,
    }


def environment_report():
    """返回可直接打印/写日志的环境报告字典。"""
    return {
        "octsdk": __import__("octsdk").__version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": _module_version("numpy"),
        "scipy": _module_version("scipy"),
        "cupy": _cupy_report(),
        "ikap_sdk": _vendor_report(
            IKAP_ROOT_ENV,
            IKAP_DEFAULT_ROOT,
            "I-TEK IKap SDK（相机/DLL），需要客户自行安装，SDK 不打包",
        ),
        "artdaq_sdk": _vendor_report(
            ARTDAQ_LIB_ENV,
            ARTDAQ_DEFAULT_LIB,
            "ART-DAQ Python 库（振镜/DO），需要客户自行安装",
        ),
        "disk_free_gb": _disk_free_gb(),
    }


def _disk_free_gb():
    try:
        usage = shutil.disk_usage(os.getcwd())
        return round(usage.free / 1024**3, 1)
    except Exception:
        return None
