"""色散/插值标定数据：``intpX.bin`` / ``intpXp.bin`` / ``intpIndice.bin`` / ``dspPhase.bin``。

与主程序 ``ThreadGPU.update_Dispersion`` 的约定完全一致：

* ``intpX``    原始光谱轴（float32，长度 = 光谱采样数）
* ``intpXp``   目标线性 k 轴（float32，同样长度）
* ``intpIndice`` 插值索引表（uint16，``2 × samples``，两种排布都接受）
* ``dspPhase``  色散相位（float32，弧度），实际乘的是 ``exp(-1j·phase)``
"""

import os
from dataclasses import dataclass, field

import numpy as np

from .._log import get_logger
from ..errors import CalibrationError

CALIBRATION_FILES = (
    ("intp_x", "intpX.bin", np.float32),
    ("intp_xp", "intpXp.bin", np.float32),
    ("indice", "intpIndice.bin", np.uint16),
    ("dispersion_phase", "dspPhase.bin", np.float32),
)


@dataclass
class DispersionCalibration:
    """一次加载、两种后端共享的标定数据。"""

    intp_x: np.ndarray
    intp_xp: np.ndarray
    indice: np.ndarray
    dispersion: np.ndarray
    samples: int
    directory: str
    _backend_cache: dict = field(default_factory=dict, repr=False)

    @classmethod
    def load(cls, directory, samples):
        """从目录加载；文件缺失、尺寸不符或索引越界都抛 :class:`CalibrationError`。"""
        directory = str(directory or "").strip()
        if not directory or not os.path.isdir(directory):
            raise CalibrationError(f"calibration directory not found: {directory or '<empty>'}")
        samples = int(samples)
        missing = [
            filename
            for _attr, filename, _dtype in CALIBRATION_FILES
            if not os.path.isfile(os.path.join(directory, filename))
        ]
        if missing:
            raise CalibrationError(
                f"calibration is incomplete in {directory}: missing {', '.join(missing)}"
            )

        arrays = {}
        for attr, filename, dtype in CALIBRATION_FILES:
            path = os.path.join(directory, filename)
            raw = np.fromfile(path, dtype=dtype)
            if attr == "indice":
                raw = _reshape_index_table(raw, samples, path)
            elif raw.size != samples:
                raise CalibrationError(
                    f"{filename} has {raw.size} values, expected {samples}: {path}"
                )
            arrays[attr] = raw

        calibration = cls(
            intp_x=np.ascontiguousarray(arrays["intp_x"], dtype=np.float32),
            intp_xp=np.ascontiguousarray(arrays["intp_xp"], dtype=np.float32),
            indice=np.ascontiguousarray(arrays["indice"], dtype=np.uint16),
            dispersion=np.exp(
                -1j * np.ascontiguousarray(arrays["dispersion_phase"], dtype=np.float32)
            ).astype(np.complex64),
            samples=samples,
            directory=directory,
        )
        get_logger("calibration").info("dispersion calibration loaded from %s", directory)
        return calibration

    @property
    def usable(self):
        return (
            self.intp_x.size == self.samples
            and self.intp_xp.size == self.samples
            and self.indice.shape == (2, self.samples)
            and self.dispersion.size == self.samples
        )

    def backend_arrays(self, xp):
        """返回该后端下的只读数组（首次访问时上传，之后复用）。"""
        key = getattr(xp, "__name__", str(xp))
        cached = self._backend_cache.get(key)
        if cached is not None:
            return cached
        payload = {
            "intp_x": xp.asarray(self.intp_x, dtype=xp.float32),
            "intp_xp": xp.asarray(self.intp_xp, dtype=xp.float32),
            "indice1": xp.asarray(self.indice[0].astype(np.int32, copy=False)),
            "indice2": xp.asarray(self.indice[1].astype(np.int32, copy=False)),
            "dispersion": xp.asarray(self.dispersion, dtype=xp.complex64),
        }
        self._backend_cache = {key: payload}
        return payload

    def describe(self):
        return {
            "directory": self.directory,
            "samples": int(self.samples),
            "usable": bool(self.usable),
        }


def _reshape_index_table(raw, samples, path):
    """接受 ``2×samples`` 或 ``samples×2`` 两种排布，并校验索引不越界。"""
    expected = 2 * int(samples)
    if raw.size != expected:
        raise CalibrationError(
            f"intpIndice.bin has {raw.size} values, expected {expected}: {path}"
        )
    candidates = (("2xsamples", raw.reshape([2, samples])), ("samplesx2", raw.reshape([samples, 2]).T))
    diagnostics = []
    for layout, candidate in candidates:
        low, high = int(candidate.min()), int(candidate.max())
        diagnostics.append(f"{layout}: min={low}, max={high}")
        if low >= 0 and high <= int(samples) - 1:
            if layout != "2xsamples":
                get_logger("calibration").info("index table loaded as %s", layout)
            return candidate
    raise CalibrationError(
        "intpIndice.bin indices are out of range for samples="
        f"{samples}: " + "; ".join(diagnostics)
    )
