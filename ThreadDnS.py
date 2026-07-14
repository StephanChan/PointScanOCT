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
from HardwareSpecs import TIFF_APPEND_WRITES_DEFAULT
from DataShape import data_shape

ALINE_MODES = ()

BLINE_MODES = (
    AcqTypes.FINITE_BLINE,
    AcqTypes.CONTINUOUS_BLINE,
    AcqTypes.CONTINUOUS_KX,
)

CSCAN_MODES = (
    AcqTypes.FINITE_CSCAN,
    AcqTypes.CONTINUOUS_CSCAN,
)

DNS_ROUTINE_DIAGNOSTICS = False


class DnSThread(QThread):
    def __init__(self):
        super().__init__()
        self.AIP = []
        self.XYVolume = []
        self.display_actions = 0
        self.active_tasks = 0
        self.tiff_append_writes = TIFF_APPEND_WRITES_DEFAULT
        self._tiff_initialized_files = set()
        self.MeanVolume = []

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
                    AcqTypes.FINITE_BLINE,
                    AcqTypes.CONTINUOUS_BLINE,
                    AcqTypes.CONTINUOUS_KX,
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
        if getattr(self, "current_acq_mode", None) == AcqTypes.CONTINUOUS_KX:
            return 1
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

    def write_stack_tiff(self, filename, stack, count=None):
        frame_count = int(stack.shape[0]) if count is None else int(count)
        for ii in range(frame_count):
            self.write_tiff_frame(filename, stack[ii], append_if_exists=True)

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

    def write_tiff_frame(self, filename, image, append_if_exists=True):
        path = Path(filename)
        resolved = str(path.resolve())
        if self.tiff_append_writes:
            TIFF.imwrite(filename, image, append=append_if_exists)
            if append_if_exists:
                self._tiff_initialized_files.add(resolved)
            return

        first_write = resolved not in self._tiff_initialized_files
        if first_write:
            self.reset_tiff_output(filename)
        TIFF.imwrite(filename, image, append=(append_if_exists and not first_write))
        if append_if_exists:
            self._tiff_initialized_files.add(resolved)

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
        Bline = display_data[0]
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
        if raw and bline_avg > 1:
            package_y_pixels = int(display_data.shape[0]) // bline_avg
            Cscan = display_data[:package_y_pixels * bline_avg].reshape(
                [package_y_pixels, bline_avg, Xpixels, Zpixels]
            )
            Cscan = np.mean(Cscan, 1)
            y_start = package_start // bline_avg
        else:
            Cscan = display_data.copy()
            package_y_pixels = int(Cscan.shape[0])
            y_start = package_start // max(1, int(gpu_avg_count))

        Ypixels = package_y_pixels
        if package_total_frames is not None:
            divisor = bline_avg if raw else max(1, int(gpu_avg_count))
            Ypixels = max(package_y_pixels, int(package_total_frames) // divisor)

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
        shape = data_shape(self.ui, data, raw, acq_mode, gpu_avg_count)
        data_to_save = self.save_data(data)
        Xpixels = shape.x_pixels
        Yrpt = shape.repeat_count
        bundle = self.item.filename_bundle or {}
        if acq_mode in BLINE_MODES:
            filename = bundle.get("filename")
            if not filename:
                raise RuntimeError("Missing bline filename bundle.")
            self.write_stack_tiff(filename, data_to_save, Yrpt)

        elif acq_mode in CSCAN_MODES:
            filename = bundle.get("filename")
            if not filename:
                raise RuntimeError("Missing cscan filename bundle.")
            count = int(data_to_save.shape[0])
            self.write_stack_tiff(filename, data_to_save, count)

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

