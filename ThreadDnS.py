# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 18:26:44 2023

@author: admin
"""

from PyQt5.QtCore import QThread
import numpy as np
import traceback
import time
import os
from pathlib import Path
from scipy import ndimage
import tifffile as TIFF
from ActionTypes import AcqTypes, DnSActions, EXIT_ACTION
from HardwareSpecs import TIFF_APPEND_WRITES_DEFAULT, TIFF_BIGTIFF_DEFAULT
from DataShape import data_shape

ALINE_MODES = ()

BLINE_MODES = (
    AcqTypes.FINITE_BSCAN,
    AcqTypes.CONTINUOUS_BLINE,
    AcqTypes.X_FREQUENCY,
)

CSCAN_MODES = (
    AcqTypes.FINITE_CSCAN,
    AcqTypes.CONTINUOUS_CSCAN,
)

DNS_ROUTINE_DIAGNOSTICS = False

# Y notch (band-stop) filter ----------------------------------------------------
# Sharp, DC-centred band-stop along the Y (B-line) axis of the rebuilt C-scan.
# The GPU only ever sees one package (CSCAN_PACKAGE_Y_LINES lines), so the slow
# axis is complete only here: the filter runs at the end of a scan (CLEAR), on the
# assembled volume, for C-scan modes only and never on raw spectra. The suppressed
# half-width is a fraction of the Y Nyquist frequency (UI: YNotchFilter /
# YNotchBandwidth). Disabled by default.
NOTCH_Y_FILTER_DEFAULT = False
NOTCH_Y_CENTER_BAND_HALF_WIDTH_FRACTION = 0.25
NOTCH_Y_FILTER_Z_CHUNK = 32


class DnSThread(QThread):
    def __init__(self):
        super().__init__()
        self.AIP = []
        self.XYVolume = []
        self.display_actions = 0
        self.active_tasks = 0
        self.tiff_append_writes = TIFF_APPEND_WRITES_DEFAULT
        self.tiff_bigtiff = TIFF_BIGTIFF_DEFAULT
        self._tiff_initialized_files = set()
        # 每个 TIFF 已写入的帧数：C-scan 每个 package 都会把整卷交回来，只追加新增
        # 的已填满行，否则文件会成倍膨胀（并最终撞上 4 GiB / 32 位偏移上限）。
        self._tiff_written_frames = {}
        self.MeanVolume = []
        # Y notch state (see NOTCH_Y_* above) plus the C-scan target needed to
        # rewrite the file once the whole volume has been filtered.
        self.notch_y_enabled = NOTCH_Y_FILTER_DEFAULT
        self.notch_y_half_width_fraction = NOTCH_Y_CENTER_BAND_HALF_WIDTH_FRACTION
        self._notch_y_weights_cache = {}
        self._notch_y_weights_gpu_cache = {}
        self._last_cscan_filename = None
        self._last_cscan_mode = None
        self._last_notified_cscan_filename = None

    def reset_accumulators(self):
        self.AIP = []
        self.XYVolume = []
        self.MeanVolume = []

    def run(self):
        self.QueueOut()

    def QueueOut(self):
        self.item = self.queue.get()
        while self.item.action != EXIT_ACTION:
            self.active_tasks += 1
            start = time.time()
            profile = dict(getattr(self.item, "profile", {}) or {})
            profile["dns_start_s"] = time.perf_counter()
            if profile.get("dns_queue_put_s") is not None:
                profile["dns_queue_wait_s"] = profile["dns_start_s"] - profile["dns_queue_put_s"]
            self.current_acq_mode = self.item.acq_mode
            try:
                if self.item.action in (
                    AcqTypes.FINITE_ALINE,
                    AcqTypes.CONTINUOUS_ALINE,
                ):
                    self.display_actions += 1
                    self.Process_aline(self.item.data, self.item.raw, self.current_acq_mode, self.item.gpu_avg_count)
                    self._emit_display(kind="aline")
                elif self.item.action in (
                    AcqTypes.FINITE_BSCAN,
                    AcqTypes.CONTINUOUS_BLINE,
                    AcqTypes.X_FREQUENCY,
                ):
                    self.Process_bline(self.item.data, self.item.raw, self.current_acq_mode, self.item.gpu_avg_count)
                    self.display_actions += 1
                    emit_start = time.perf_counter()
                    self._emit_display(kind="bline")
                    profile["dns_emit_display_s"] = time.perf_counter() - emit_start
                    profile["dns_process_display_s"] = time.perf_counter() - profile["dns_start_s"]
                    self.print_package_profile("bline", profile)
                elif self.item.action in (
                    AcqTypes.FINITE_CSCAN,
                    AcqTypes.CONTINUOUS_CSCAN,
                ):
                    self.display_actions += 1
                    self.Process_Cscan(self.item.data, self.item.raw, self.current_acq_mode, self.item.gpu_avg_count)
                    emit_start = time.perf_counter()
                    self._emit_display(kind="cscan")
                    profile["dns_emit_display_s"] = time.perf_counter() - emit_start
                    profile["dns_process_display_s"] = time.perf_counter() - profile["dns_start_s"]
                    self.print_package_profile("cscan", profile)

                elif self.item.action == DnSActions.CLEAR:
                    self.finalize_cscan_output()
                    self.reset_accumulators()
                elif self.item.action == DnSActions.DISPLAY_COUNTS:
                    self.print_display_counts(self.item.context)

                elif self.item.action == DnSActions.AGAR_TILE:
                    self.SurfFilename()
                elif self.item.action == DnSActions.WRITE_AGAR:
                    self.WriteAgar(self.item.data, self.item.context)
                else:
                    message = f"Unknown display/save command: {self.item.action}"
                    print(message)
                    self.emit_status(message)
                if DNS_ROUTINE_DIAGNOSTICS and time.time() - start > 2.0:
                    print("time for DnS:", round(time.time() - start, 3))
            except Exception as error:
                message = "Display/save processing failed. This item was skipped."
                print(message)
                self.emit_status(message)
                print(traceback.format_exc())
            finally:
                output_buffer_token = getattr(self.item, "output_buffer_token", None)
                if output_buffer_token is not None and hasattr(self, "PinnedOutputFreeQueue"):
                    self.PinnedOutputFreeQueue.put(output_buffer_token)
                self.active_tasks = max(0, self.active_tasks - 1)
            self.item = self.queue.get()

        self.emit_status("Display/save thread exited.")

    def print_package_profile(self, kind, profile):
        if not DNS_ROUTINE_DIAGNOSTICS or not profile:
            return
        labels = [
            ("camera", "camera_package_fill_s"),
            ("gpu_queue", "gpu_queue_wait_s"),
            ("gpu", "gpu_wall_s"),
            ("H2D", "gpu_load_raw_to_gpu_s"),
            ("FFT", "gpu_fft_s"),
            ("D2H", "gpu_copy_gpu_to_host_s"),
            ("DnS", "dns_process_display_s"),
        ]
        parts = []
        for label, key in labels:
            value = profile.get(key)
            if isinstance(value, (int, float)):
                parts.append(f"{label}={value * 1000.0:.2f} ms")
        if parts:
            message = f"Package timing [{kind}]: " + ", ".join(parts)
            print(message)
            self.emit_status(message)

    def is_idle(self):
        return self.queue.qsize() == 0 and self.active_tasks == 0

    def emit_status(self, message):
        if message is None:
            return
        self.ui_bridge.status_message.emit(str(message))

    def current_save_enabled(self):
        return self.ui.Save.isChecked()

    def current_aline_avg(self):
        return max(1, int(self.ui.AlineAVG.value()))

    def current_y_pixels(self):
        return max(1, int(self.ui.Ypixels.value()))

    def current_z_depth_index(self, z_pixels):
        if z_pixels <= 0:
            return 0
        if not hasattr(self.ui, "ZDepthBar"):
            return 0
        return max(0, min(int(self.ui.ZDepthBar.value()), int(z_pixels) - 1))

    def current_bline_avg(self):
        return max(1, int(self.ui.BlineAVG.value()))

    # -- Y notch (band-stop) filter ---------------------------------------- #

    def current_y_notch_enabled(self):
        widget = getattr(self.ui, "YNotchFilter", None)
        if widget is not None and hasattr(widget, "isChecked"):
            try:
                return bool(widget.isChecked())
            except Exception:
                pass
        return bool(self.notch_y_enabled)

    def current_y_notch_half_width_fraction(self):
        widget = getattr(self.ui, "YNotchBandwidth", None)
        if widget is not None and hasattr(widget, "value"):
            try:
                value = float(widget.value())
                if np.isfinite(value):
                    return max(0.0, min(value, 1.0))
            except Exception:
                pass
        return float(self.notch_y_half_width_fraction)

    def notch_y_weights(self, y_pixels):
        """Sharp DC-centred band-stop along Y, same shape as the X notch.

        Native FFT order (no shift): np.fft.fftfreq gives cycles/sample in
        [-0.5, 0.5), so Nyquist is 0.5 and the cutoff is a fraction of it.
        """
        y_pixels = int(y_pixels)
        fraction = float(self.current_y_notch_half_width_fraction())
        cache_key = (y_pixels, fraction)
        cached = self._notch_y_weights_cache.get(cache_key)
        if cached is not None:
            return cached
        freq_axis = np.fft.fftfreq(y_pixels).astype(np.float32)
        cutoff = np.float32(fraction * 0.5)
        weights = (np.abs(freq_axis) > cutoff).astype(np.float32)
        self._notch_y_weights_cache = {cache_key: weights}
        return weights

    def apply_y_notch(self, volume):
        """Band-stop a (Y, X, Z) float32 stack along Y, chunked over Z."""
        if not isinstance(volume, np.ndarray) or volume.ndim != 3:
            return volume
        y_pixels = int(volume.shape[0])
        if y_pixels < 2:
            return volume
        weights = self.notch_y_weights(y_pixels)
        chunk = max(1, min(int(volume.shape[2]), NOTCH_Y_FILTER_Z_CHUNK))
        filtered = np.empty_like(volume)
        for start in range(0, volume.shape[2], chunk):
            stop = min(start + chunk, volume.shape[2])
            spectrum = np.fft.fft(volume[:, :, start:stop], axis=0)
            spectrum *= weights[:, None, None]
            filtered[:, :, start:stop] = np.real(np.fft.ifft(spectrum, axis=0))
        return filtered

    def y_notch_weights_gpu(self, y_pixels, cupy):
        """Cached cupy copy of the Y-notch weights (same shape as notch_y_weights)."""
        y_pixels = int(y_pixels)
        fraction = float(self.current_y_notch_half_width_fraction())
        cache_key = (y_pixels, fraction)
        cached = self._notch_y_weights_gpu_cache.get(cache_key)
        if cached is not None:
            return cached
        weights_gpu = cupy.asarray(self.notch_y_weights(y_pixels))
        self._notch_y_weights_gpu_cache = {cache_key: weights_gpu}
        return weights_gpu

    def apply_y_notch_gpu(self, volume):
        """Y notch on the GPU (cuFFT). Returns None when cupy is unavailable or the
        transfer fails, so the caller can fall back to the numpy implementation.

        The transform runs along Y, keeping that axis contiguous: each Z block is
        staged as (Z_chunk, X, Y), filtered in one cuFFT call and copied back, so
        GPU memory stays bounded even for large C-scan volumes.
        """
        try:
            import cupy
        except Exception as error:
            print("cupy unavailable; Y notch falls back to CPU:", error)
            return None
        if not isinstance(volume, np.ndarray) or volume.ndim != 3:
            return None
        y_pixels = int(volume.shape[0])
        x_pixels = int(volume.shape[1])
        z_pixels = int(volume.shape[2])
        if y_pixels < 2 or z_pixels < 1:
            return None
        weights_gpu = self.y_notch_weights_gpu(y_pixels, cupy)
        # Bound the staging buffer by the free device memory: the block needs the
        # input plus a complex spectrum (~12 bytes per sample).
        try:
            free_bytes, _total = cupy.cuda.Device().mem_info
            per_slice = max(1, y_pixels * x_pixels * 12)
            z_chunk = int(max(1, min(NOTCH_Y_FILTER_Z_CHUNK, z_pixels, free_bytes // per_slice)))
        except Exception:
            z_chunk = int(max(1, min(NOTCH_Y_FILTER_Z_CHUNK, z_pixels)))
        out = np.empty_like(volume)
        try:
            stream = cupy.cuda.Stream(non_blocking=True)
            with stream:
                for start in range(0, z_pixels, z_chunk):
                    stop = min(start + z_chunk, z_pixels)
                    block_gpu = cupy.ascontiguousarray(
                        cupy.asarray(volume[:, :, start:stop]).transpose(2, 1, 0)
                    )
                    spectrum_gpu = cupy.fft.fft(block_gpu, axis=-1)
                    spectrum_gpu *= weights_gpu[cupy.newaxis, cupy.newaxis, :]
                    filtered_gpu = cupy.real(cupy.fft.ifft(spectrum_gpu, axis=-1))
                    out[:, :, start:stop] = cupy.asnumpy(filtered_gpu.transpose(2, 1, 0))
                stream.synchronize()
        except Exception as error:
            print("GPU Y notch failed; falling back to CPU:", error)
            return None
        return out

    def y_notch_applies(self, acq_mode, raw=False):
        """Y notch is for rebuilt C-scan volumes only; raw spectra are never filtered."""
        if raw or acq_mode is None:
            return False
        if acq_mode not in CSCAN_MODES:
            return False
        return self.current_y_notch_enabled()

    def filter_cscan_y_notch(self):
        """Apply the Y notch to the rebuilt volume; returns (applied, backend, filled).

        Only the filled Y rows take part in the transform, so the zero-padded tail of a
        continuous scan cannot produce edge artefacts.
        """
        mode = getattr(self, "current_acq_mode", None) or self._last_cscan_mode
        filled = self.cscan_filled_rows()
        if not self.y_notch_applies(mode, False):
            return False, None, filled
        if not isinstance(self.XYVolume, np.ndarray) or self.XYVolume.ndim != 3:
            return False, None, filled
        if filled < 2:
            return False, None, filled
        z_pixels = int(self.XYVolume.shape[2])
        source = np.ascontiguousarray(self.XYVolume[:filled])
        filtered = self.apply_y_notch_gpu(source)
        backend = "GPU"
        if filtered is None:
            filtered = self.apply_y_notch(source)
            backend = "CPU"
        self.XYVolume[:filled] = filtered
        z_idx = self.current_z_depth_index(z_pixels)
        self.AIP[:filled, :] = filtered[:, :, z_idx]
        self._emit_display(kind="cscan")
        return True, backend, filled

    def finalize_cscan_output(self):
        """End of scan: filter (Y notch, if on) and write the whole C-scan once.

        The volume is rebuilt in RAM (XYVolume) while the scan runs and only written
        here, so no partially scanned volume ever reaches the disk and the file cannot
        grow with duplicated packages.
        """
        filename = self._last_cscan_filename
        acq_mode = self._last_cscan_mode
        applied, backend, filled = self.filter_cscan_y_notch()
        if applied:
            message = (
                "Y notch applied to the C-scan (%s): half-width %.3f x Y Nyquist over "
                "%d B-line(s)"
                % (backend, self.current_y_notch_half_width_fraction(), filled)
            )
            print(message)
            self.emit_status(message)
        self._last_cscan_filename = None
        self._last_cscan_mode = None
        if not filename or filled <= 0 or not isinstance(self.XYVolume, np.ndarray):
            return
        started = time.time()
        self.reset_tiff_output(filename)
        self.write_stack_tiff(filename, self.XYVolume[:filled], filled)
        message = (
            "C-scan saved once at the end of the scan [%s]: %s (%d x %d x %d, %.1f s)"
            % (
                acq_mode or "?",
                os.path.basename(filename),
                int(filled),
                int(self.XYVolume.shape[1]),
                int(self.XYVolume.shape[2]),
                time.time() - started,
            )
        )
        print(message)
        self.emit_status(message)

    def write_stack_tiff(self, filename, stack, count=None):
        frame_count = int(stack.shape[0]) if count is None else int(count)
        if frame_count <= 0:
            return
        for ii in range(frame_count):
            self.write_tiff_frame(filename, stack[ii], append_if_exists=True)
        resolved = str(Path(filename).resolve())
        self._tiff_written_frames[resolved] = self.tiff_written_frames(filename) + frame_count

    def cscan_filled_rows(self):
        """XYVolume 里已经填过数据的行数（C-scan 是逐包填充的）。"""
        if not isinstance(self.XYVolume, np.ndarray) or self.XYVolume.ndim != 3:
            return 0
        rows = int(self.XYVolume.shape[0])
        filled = int(getattr(self, "latest_cscan_bline_index", 0) or 0) + 1
        return max(0, min(filled, rows))

    def reset_cscan_output_for_new_volume(self, y_start):
        """新的 C-scan 卷开始时从零重写同一文件，否则新卷的行会接在旧数据后面。"""
        if int(y_start) != 0 or not self.current_save_enabled():
            return
        bundle = getattr(self.item, "filename_bundle", None) or {}
        filename = bundle.get("filename")
        if filename and self.tiff_written_frames(filename) > 0:
            self.reset_tiff_output(filename)

    def notice_cscan_save_deferred(self, filename):
        """每次扫描提示一次：C-scan 文件在采集结束后一次性写出。"""
        message = (
            "C-scan saving deferred: the volume stays in RAM and is written once when "
            "the scan finishes (%s)." % os.path.basename(filename)
        )
        print(message)
        self.emit_status(message)

    @staticmethod
    def display_data(data):
        if isinstance(data, np.ndarray) and data.dtype.kind == "c":
            return np.abs(data)
        return data

    @staticmethod
    def save_data(data):
        if not (isinstance(data, np.ndarray) and data.dtype.kind == "c"):
            return data
        z_pixels = data.shape[-1]
        interleaved = np.empty(data.shape[:-1] + (z_pixels * 2,), dtype=np.float32)
        interleaved[..., :z_pixels] = np.abs(data).astype(np.float32, copy=False)
        interleaved[..., z_pixels:] = np.angle(data).astype(np.float32, copy=False)
        return interleaved

    def reset_tiff_output(self, filename):
        path = Path(filename)
        try:
            if path.exists():
                path.unlink()
        except OSError as error:
            message = f"Failed to reset TIFF output {filename}: {error}"
            print(message)
            self.emit_status(message)
        self._tiff_initialized_files.discard(str(path.resolve()))
        self._tiff_written_frames.pop(str(path.resolve()), None)

    def tiff_written_frames(self, filename):
        """已写入该 TIFF 的帧数（供 C-scan 只追加新行使用）。"""
        return int(self._tiff_written_frames.get(str(Path(filename).resolve()), 0))

    def write_tiff_frame(self, filename, image, append_if_exists=True):
        path = Path(filename)
        resolved = str(path.resolve())
        if self.tiff_append_writes:
            first_write = resolved not in self._tiff_initialized_files
            self.imwrite_tiff(filename, image, append=(append_if_exists and not first_write),
                              create_new=first_write)
            if append_if_exists:
                self._tiff_initialized_files.add(resolved)
            return

        first_write = resolved not in self._tiff_initialized_files
        if first_write:
            self.reset_tiff_output(filename)
        self.imwrite_tiff(filename, image, append=(append_if_exists and not first_write),
                          create_new=first_write)
        if append_if_exists:
            self._tiff_initialized_files.add(resolved)

    def imwrite_tiff(self, filename, image, append=False, create_new=False):
        """写一帧；新建文件时用 BigTIFF（64 位偏移），避免 4 GiB 处的 struct.error。"""
        options = {}
        if create_new and self.tiff_bigtiff:
            options["bigtiff"] = True
        TIFF.imwrite(filename, image, append=append, **options)

    def _emit_display(self, kind: str):
        bridge = getattr(self, "ui_bridge", None)
        if bridge is None:
            return

        acq_mode = self.current_acq_mode

        if kind == "aline" and hasattr(self, "Aline") and np.size(self.Aline) > 0:
            bridge.aline_ready.emit({"mode": acq_mode, "aline": np.array(self.Aline, copy=True)})

        if kind == "bline" and hasattr(self, "Bline") and np.size(self.Bline) > 0:
            bridge.bline_ready.emit(
                {
                    "mode": acq_mode,
                    "bline": np.array(self.Bline, copy=True),
                }
            )

        if (
            kind == "cscan"
            and hasattr(self, "Bline")
            and hasattr(self, "AIP")
            and np.size(self.Bline) > 0
            and np.size(self.AIP) > 0
        ):
            bridge.cscan_ready.emit(
                {
                    "mode": acq_mode,
                    "bline": np.array(self.Bline, copy=True),
                    "latest_bline_index": getattr(self, "latest_cscan_bline_index", None),
                    "aip": np.array(self.AIP, copy=True),
                    "volume": np.array(self.XYVolume, copy=True) if hasattr(self, "XYVolume") and np.size(self.XYVolume) > 0 else None,
                }
            )

    def print_display_counts(self, display_name=""):
        if DNS_ROUTINE_DIAGNOSTICS:
            print(f"{self.display_actions} {display_name} display update(s) completed.")
        self.display_actions = 0

    def Process_aline(self, data, raw=False, acq_mode=None, gpu_avg_count=1):
        display_data = self.display_data(data)
        shape = data_shape(self.ui, display_data, raw, acq_mode, gpu_avg_count)
        Zpixels = shape.z_pixels
        Xpixels = shape.x_pixels
        if display_data.shape[0] > 1:
            Ascan = np.mean(display_data, 0)
        else:
            Ascan = display_data[0]
        aline_avg = self.current_aline_avg()
        if aline_avg > 1:
            Ascan = Ascan.reshape([Xpixels // aline_avg, aline_avg, Zpixels])
            Ascan = np.mean(Ascan, 1)
            Xpixels = Xpixels // aline_avg

        self.Aline = Ascan[Xpixels // 2]
        if self.current_save_enabled():
            self.Save(data=data, raw=raw, acq_mode=acq_mode, gpu_avg_count=gpu_avg_count)

    def Process_bline(self, data, raw=False, acq_mode=None, gpu_avg_count=1):
        display_data = self.display_data(data)
        shape = data_shape(self.ui, display_data, raw, acq_mode, gpu_avg_count)
        Zpixels = shape.z_pixels
        Xpixels = shape.x_pixels
        series = display_data
        if getattr(series, "ndim", 0) == 3 and int(series.shape[1]) == 1:
            series = series[:, 0, :]
        if getattr(series, "ndim", 0) != 2:
            raise ValueError(f"Line-scan Bscan data must be [frames, z], got {getattr(display_data, 'shape', None)}")
        complete_frames = (int(series.shape[0]) // Xpixels) * Xpixels
        if complete_frames <= 0:
            raise ValueError(
                f"Not enough A-lines for one Bscan: frames={int(series.shape[0])}, Xpixels={Xpixels}"
            )
        repeat_count = max(1, complete_frames // Xpixels)
        Bline = series[:complete_frames].reshape(repeat_count, Xpixels, Zpixels).mean(axis=0)
        aline_avg = self.current_aline_avg()
        if aline_avg > 1:
            Bline = Bline.reshape([Xpixels // aline_avg, aline_avg, Zpixels])
            Bline = np.mean(Bline, 1)
            Xpixels = Xpixels // aline_avg
        self.Bline = np.transpose(Bline)

        if self.current_save_enabled():
            self.Save(data=data, raw=raw, acq_mode=acq_mode, gpu_avg_count=gpu_avg_count)

    def Process_Cscan(self, data, raw=False, acq_mode=None, gpu_avg_count=1):
        display_data = self.display_data(data)
        shape = data_shape(self.ui, display_data, raw, acq_mode, gpu_avg_count)
        Zpixels = shape.z_pixels
        Xpixels = shape.x_pixels
        package_total_frames = getattr(self.item, "package_total_frames", None)
        package_start = int(getattr(self.item, "package_start", 0) or 0)

        bline_avg = self.current_bline_avg()
        series = display_data
        if getattr(series, "ndim", 0) == 3 and int(series.shape[1]) == 1:
            series = series[:, 0, :]
        if getattr(series, "ndim", 0) != 2:
            raise ValueError(f"Line-scan Cscan data must be [frames, z], got {getattr(display_data, 'shape', None)}")
        frames_per_y = max(1, Xpixels * bline_avg)
        package_y_pixels = int(series.shape[0]) // frames_per_y
        complete_frames = package_y_pixels * frames_per_y
        if complete_frames <= 0:
            raise ValueError(
                f"Not enough A-lines for one Cscan row: frames={int(series.shape[0])}, "
                f"frames_per_y={frames_per_y}"
            )
        Cscan = series[:complete_frames].reshape(package_y_pixels, bline_avg, Xpixels, Zpixels)
        Cscan = np.mean(Cscan, 1)
        y_start = package_start // frames_per_y

        Ypixels = package_y_pixels
        if package_total_frames is not None:
            Ypixels = max(package_y_pixels, int(package_total_frames) // frames_per_y)

        aline_avg = self.current_aline_avg()
        if aline_avg > 1:
            Cscan = Cscan.reshape([package_y_pixels, Xpixels // aline_avg, aline_avg, Zpixels])
            Cscan = np.mean(Cscan, 2)
            Xpixels = Xpixels // aline_avg

        z_idx = self.current_z_depth_index(Zpixels)
        needs_new_volume = (
            package_total_frames is None
            or not isinstance(self.XYVolume, np.ndarray)
            or self.XYVolume.shape != (Ypixels, Xpixels, Zpixels)
            or y_start == 0
        )
        if needs_new_volume:
            self.XYVolume = np.zeros((Ypixels, Xpixels, Zpixels), dtype=np.float32)
            self.AIP = np.zeros((Ypixels, Xpixels), dtype=np.float32)
            self.reset_cscan_output_for_new_volume(y_start)

        y_end = min(y_start + package_y_pixels, Ypixels)
        valid_count = max(0, y_end - y_start)
        if valid_count > 0:
            self.XYVolume[y_start:y_end, :, :] = Cscan[:valid_count]
            self.AIP[y_start:y_end, :] = Cscan[:valid_count, :, z_idx]
            bline_index = y_end - 1
            self.latest_cscan_bline_index = bline_index
            self.Bline = np.transpose(self.XYVolume[bline_index, :, :]).copy()
        else:
            self.latest_cscan_bline_index = y_start
            self.Bline = np.transpose(Cscan[0, :, :]).copy()

        if self.current_save_enabled():
            self.Save(data=data, raw=raw, acq_mode=acq_mode, gpu_avg_count=gpu_avg_count)

    def Save(self, data=None, raw=False, acq_mode=None, gpu_avg_count=1):
        if data is None:
            data = []
        if getattr(self.item, "skip_save", False):
            return
        if acq_mode == AcqTypes.X_FREQUENCY:
            # Diagnostic mode: the X-frequency spectrum is displayed live and never
            # written to disk (no filename bundle is built for it either).
            return
        shape = data_shape(self.ui, data, raw, acq_mode, gpu_avg_count)
        data_to_save = self.save_data(data)
        Xpixels = shape.x_pixels
        Yrpt = shape.repeat_count
        bundle = self.item.filename_bundle or {}
        if acq_mode in BLINE_MODES:
            filename = bundle.get("filename")
            if not filename:
                raise RuntimeError("Missing bline filename bundle.")
            if not hasattr(self, "Bline") or np.size(self.Bline) <= 0:
                return
            self.write_stack_tiff(filename, np.asarray(self.Bline)[None, :, :], 1)

        elif acq_mode in CSCAN_MODES:
            filename = bundle.get("filename")
            if not filename:
                raise RuntimeError("Missing cscan filename bundle.")
            if not isinstance(self.XYVolume, np.ndarray) or np.size(self.XYVolume) <= 0:
                return
            self._last_cscan_filename = filename
            self._last_cscan_mode = acq_mode
            if filename != self._last_notified_cscan_filename:
                # Nothing is written while scanning: the whole C-scan is rebuilt in RAM
                # (XYVolume) and written once when the acquisition finishes
                # (DnSActions.CLEAR -> finalize_cscan_output).
                self._last_notified_cscan_filename = filename
                self.notice_cscan_save_deferred(filename)
            return

    def WriteData(self, data, filename):
        filePath = self.ui.DIR.toPlainText()
        filePath = filePath + "/" + filename
        start = time.time()
        with open(filePath, "wb") as fp:
            data.tofile(fp)
        if time.time() - start > 1:
            message = "Saving took " + str(round(time.time() - start, 3)) + " s."
            print(message)

    def SurfFilename(self):
        pass

    def WriteAgar(self, data, context):
        Ystep, Xstep = context
        slice_num = self.ui.SliceN.value() if hasattr(self.ui, "SliceN") else 1
        filename = "slice-" + str(slice_num) + "-agarTiles X-" + str(Xstep) + "-by Y-" + str(Ystep) + "-.bin"
        filePath = self.ui.DIR.toPlainText()
        filePath = filePath + "/" + filename
        with open(filePath, "wb") as fp:
            data.tofile(fp)

