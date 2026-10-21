"""按包累积成 C-scan 体积（对应主程序 ``ThreadDnS.Process_Cscan`` 的累积部分）。

一包处理结果的行数 = ``y_lines_in_package × x_pixels_eff``（``x_pixels_eff`` 是光谱域
平均之后的每 B-line A-line 数）。``package_start`` 是**原始 A-line 行数**，所以 Y 起始
位置要用 ``x_pixels × bline_avg`` 换算。
"""

import numpy as np

from .config import ProcessingConfig
from .errors import ProcessingError
from .processing.shapes import effective_x_pixels


class Pipeline:
    """一次扫描的体积缓冲。"""

    def __init__(self, processing, y_lines=1, max_volume_gb=4.0):
        if not isinstance(processing, ProcessingConfig):
            raise ProcessingError(f"processing must be ProcessingConfig, got {type(processing)}")
        self.processing = processing
        self.y_lines = max(1, int(y_lines))
        self.x_pixels = effective_x_pixels(processing, processing.aline_avg)
        self.depth_pixels = processing.resolved_depth_pixels
        self.dtype = np.complex64 if processing.keep_complex else np.float32
        self.bytes_needed = (
            self.y_lines
            * self.x_pixels
            * self.depth_pixels
            * np.dtype(self.dtype).itemsize
        )
        limit = float(max_volume_gb) * 1024 ** 3
        if self.bytes_needed > limit:
            raise ProcessingError(
                f"volume {self.y_lines}×{self.x_pixels}×{self.depth_pixels} needs "
                f"{self.bytes_needed / 1024 ** 3:.2f} GiB > max_volume_gb={max_volume_gb}"
            )
        self.volume = np.zeros(
            (self.y_lines, self.x_pixels, self.depth_pixels), dtype=self.dtype
        )
        self._raw_frames_per_y = max(
            1, int(processing.x_pixels) * int(processing.bline_avg)
        )
        self.filled_y = 0
        self.packages = 0
        self.rows = 0
        self.last_bline = None

    def add_package(self, processed, package_start):
        """写入一包处理结果，返回本次实际写入的 Y 行数。"""
        data = np.asarray(processed.data)
        if data.ndim != 2:
            raise ProcessingError(f"processed data must be (rows, depth), got {data.shape}")
        if int(data.shape[1]) != self.depth_pixels:
            raise ProcessingError(
                f"depth mismatch: {int(data.shape[1])} vs {self.depth_pixels}"
            )
        rows = int(data.shape[0])
        if rows % self.x_pixels:
            raise ProcessingError(
                f"package rows must be a multiple of x_pixels: {rows} vs {self.x_pixels}"
            )
        lines_in_package = rows // self.x_pixels
        # 单行缓冲（B-scan / 单条 C-scan）总是覆盖第 0 行
        if self.y_lines == 1:
            y_start = 0
        else:
            y_start = int(package_start) // self._raw_frames_per_y
        y_end = min(y_start + lines_in_package, self.y_lines)
        taken = max(0, y_end - y_start)
        if taken:
            block = data[: taken * self.x_pixels].reshape(taken, self.x_pixels, self.depth_pixels)
            self.volume[y_start:y_end] = block
            self.filled_y = max(self.filled_y, y_end)
        self.rows += rows
        self.packages += 1
        self.last_bline = np.array(data[-1], copy=True)
        return taken

    @property
    def is_complete(self):
        return self.filled_y >= self.y_lines

    def volume_filled(self):
        return self.volume[: self.filled_y]

    def aip(self):
        """en-face 投影（沿深度取幅度最大值）。"""
        filled = self.volume_filled()
        if filled.size == 0:
            return np.zeros((0, self.x_pixels), dtype=np.float32)
        return np.max(np.abs(filled), axis=2).astype(np.float32)

    def bline(self):
        return None if self.last_bline is None else self.last_bline

    def describe(self):
        return {
            "y_lines": int(self.y_lines),
            "x_pixels": int(self.x_pixels),
            "depth_pixels": int(self.depth_pixels),
            "filled_y": int(self.filled_y),
            "packages": int(self.packages),
            "rows": int(self.rows),
            "dtype": str(np.dtype(self.dtype)),
            "bytes_needed": int(self.bytes_needed),
        }
