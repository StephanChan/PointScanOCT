# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 16:50:25 2023

@author: admin
"""
from PyQt5.QtCore import QThread
import cupy
import numpy as np
from ActionFields import DnSActionField
import os
import time
import traceback
from queue import Empty
from ActionTypes import AcqTypes, DnSActions, EXIT_ACTION, GPUActions
from CameraUi import effective_camera_sample_count
from HardwareSpecs import get_camera_spec

# GPU performance -------------------------------------------------------------
GPU_RELEASE_MEMORY_EACH_FFT = False
GPU_FFT_CHUNK_FRAMES = 32
GPU_OVERLAP_TRANSFER = True

# Off-axis holography X-sideband filtering -----------------------------------
HOLO_X_FILTER_ENABLED = True
HOLO_REFERENCE_TILT_DEG = 1.3
HOLO_CENTER_WAVELENGTH_NM = 840.0
HOLO_WAVELENGTH_RANGE_NM = 40.0
HOLO_SIDEBAND_SIGN = +1
HOLO_SIDEBAND_HALF_WIDTH_FRACTION = 0.65
HOLO_SIDEBAND_TAPER_FRACTION = 0.30
HOLO_X_FILTER_CENTER_FRACTION_OVERRIDE = None
HOLO_REFERENCE_CARRIER_DEMODULATION = True
HOLO_CONJUGATE_POSITIVE_SIDEBAND = True
HOLO_KEEP_HIGH_FREQUENCY_TAIL = False
HOLO_SUBTRACT_COMMON_MODE_AFTER_DEMOD = False
HOLO_PRINT_FILTER_INFO = True
HOLO_X_CONTIGUOUS_FFT = True

# Depth display ---------------------------------------------------------------
DEPTH_CENTER_ZERO_DEPTH_IN_DISPLAY = True

# ContinuousKX display --------------------------------------------------------
CONTINUOUS_KX_DISPLAY_SCALE = "linear"
CONTINUOUS_KX_NORMALIZE_PER_FRAME = True

# GPU profiling ---------------------------------------------------------------
# Diagnostic mode: synchronizes CUDA stage boundaries so individual timings
# are meaningful. Disable after profiling to restore maximum stream overlap.
GPU_PROFILE_TIMING_ENABLED = False
GPU_PROFILE_TIMING_PRINT_CHUNKS = False
PINNED_OUTPUT_POOL_SIZE = 2
GPU_ROUTINE_DIAGNOSTICS = False

# Static/background normalization ---------------------------------------------
NORMALIZATION_EPS = 1e-3
BACKGROUND_X_NORMALIZATION_ROOT_ORDER = 2.0

# Diagnostics -----------------------------------------------------------------

class GPUThread(QThread):
    def __init__(self):
        super().__init__()
        self.exit_message = "GPU processing thread exited."
        self.FFT_actions = 0
        self.bg_sub = False
        self.background_gpu = None
        self.background_x_normalization = None
        self.background_x_normalization_gpu = None
        self.intpX_gpu = None
        self.intpXp_gpu = None
        self.indice1_gpu = None
        self.indice2_gpu = None
        self.dispersion_gpu = None
        self.release_gpu_memory_each_fft = GPU_RELEASE_MEMORY_EACH_FFT
        self.gpu_chunk_frames = GPU_FFT_CHUNK_FRAMES
        self.gpu_overlap_transfer = GPU_OVERLAP_TRANSFER
        self.holo_x_filter_enabled = HOLO_X_FILTER_ENABLED
        self.holo_reference_tilt_deg = HOLO_REFERENCE_TILT_DEG
        self.holo_center_wavelength_nm = HOLO_CENTER_WAVELENGTH_NM
        self.holo_wavelength_range_nm = HOLO_WAVELENGTH_RANGE_NM
        self.holo_sideband_sign = HOLO_SIDEBAND_SIGN
        self.holo_sideband_half_width_fraction = HOLO_SIDEBAND_HALF_WIDTH_FRACTION
        self.holo_sideband_taper_fraction = HOLO_SIDEBAND_TAPER_FRACTION
        self.holo_x_filter_center_fraction_override = HOLO_X_FILTER_CENTER_FRACTION_OVERRIDE
        self.holo_reference_carrier_demodulation = HOLO_REFERENCE_CARRIER_DEMODULATION
        self.holo_conjugate_positive_sideband = HOLO_CONJUGATE_POSITIVE_SIDEBAND
        self.holo_keep_high_frequency_tail = HOLO_KEEP_HIGH_FREQUENCY_TAIL
        self.holo_subtract_common_mode_after_demod = HOLO_SUBTRACT_COMMON_MODE_AFTER_DEMOD
        self.holo_print_filter_info = HOLO_PRINT_FILTER_INFO
        self.holo_x_contiguous_fft = HOLO_X_CONTIGUOUS_FFT
        self.depth_center_zero_depth_in_display = DEPTH_CENTER_ZERO_DEPTH_IN_DISPLAY
        self.holo_filter_info_printed = False
        self.kx_display_scale = CONTINUOUS_KX_DISPLAY_SCALE
        self.kx_normalize_per_frame = CONTINUOUS_KX_NORMALIZE_PER_FRAME
        self.gpu_profile_timing_enabled = GPU_PROFILE_TIMING_ENABLED
        self.gpu_profile_timing_print_chunks = GPU_PROFILE_TIMING_PRINT_CHUNKS
        self.background_x_normalization_eps = NORMALIZATION_EPS
        self.background_x_normalization_root_order = BACKGROUND_X_NORMALIZATION_ROOT_ORDER
        self.yp_gpu_buffer = None
        self.yp_gpu_stream_buffers = [None, None]
        self.raw_gpu_buffer = None
        self.raw_gpu_stream_buffers = [None, None]
        self.float_gpu_buffer = None
        self.float_gpu_stream_buffers = [None, None]
        self.gpu_streams = None
        self.holo_filter_cache = {}
        self.holo_carrier_cache = {}
        self.pinned_output_pool_size = PINNED_OUTPUT_POOL_SIZE
        self.pinned_output_buffers = {}
        self.next_pinned_output_token = 0
        self._leased_output_token = None
        self.active_tasks = 0
        self.return_processed_to_weaver = False
        self.interp = False

    def definterp(self):
        self.interp_kernel = cupy.RawKernel(r'''
            extern "C" __global__
            void interp1d(long long NAlines, long long NSamples, float* x, float* xp, float* y, unsigned short* indice1, unsigned short* indice2, float* yp){
                const int blockID = blockIdx.x + blockIdx.y * gridDim.x;
                const int threadID = blockID * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x;
                const int numThreads = blockDim.x * blockDim.y * gridDim.x * gridDim.y;

                long long int i;
                for(i=threadID; i<NAlines * NSamples; i += numThreads){
                        int sampx = i%NSamples;
                        long long int sampy0 = i / NSamples * NSamples;
                        float xt = xp[sampx];
                        float x0 = x[indice1[sampx]];
                        float x1 = x[indice2[sampx]];
                        float y0 = y[sampy0+indice1[sampx]];
                        float y1 = y[sampy0+indice2[sampx]];
                        yp[i] = (y0+(xt-x0)*(y1-y0)/(x1-x0+0.00001));
                        }
            }
            ''', 'interp1d')

    def run(self):
        self.definterp()
        self.update_Dispersion()
        self.update_background()
        self.QueueOut()

    def QueueOut(self):
        self.item = self.queue.get()
        while self.item.action != EXIT_ACTION:
            t1 = time.time()
            self.active_tasks += 1
            try:
                if self.item.action == GPUActions.GPU:
                    self.cudaFFT(self.item.DnS_action, self.item.acq_mode, self.item.memory_slot, self.item.context)
                    self.FFT_actions += 1
                elif self.item.action == GPUActions.CPU:
                    self.fft_cpu(self.item.DnS_action, self.item.acq_mode, self.item.memory_slot, self.item.context)
                    self.FFT_actions += 1
                elif self.item.action == GPUActions.CLEAR:
                    self.DnSQueue.put(DnSActionField(DnSActions.CLEAR))
                elif self.item.action == GPUActions.UPDATE_DISPERSION:
                    self.update_Dispersion()
                elif self.item.action == GPUActions.UPDATE_BACKGROUND:
                    self.update_background()
                elif self.item.action == GPUActions.DISPLAY_FFT_ACTIONS:
                    self.display_FFT_actions()
                elif self.item.action == GPUActions.DISPLAY_COUNTS:
                    an_action = DnSActionField(
                        DnSActions.DISPLAY_COUNTS,
                        context=self.item.context,
                    )
                    self.DnSQueue.put(an_action)
                else:
                    message = f"Unknown GPU command: {self.item.action}"
                    print(message)
                    self.emit_status(message)
            except Exception:
                message = "FFT processing failed. This frame was skipped."
                print(message)
                self.emit_status(message)
                print(traceback.format_exc())
            finally:
                if self._leased_output_token is not None:
                    self.PinnedOutputFreeQueue.put(self._leased_output_token)
                    self._leased_output_token = None
                if getattr(self.item, "action", None) in (GPUActions.GPU, GPUActions.CPU):
                    self.release_memory_slot(getattr(self.item, "memory_slot", None))
                self.active_tasks = max(0, self.active_tasks - 1)
                if GPU_ROUTINE_DIAGNOSTICS:
                    print("GPU thread took ", round(time.time() - t1, 2), " seconds for action: ", self.item.action)
            self.item = self.queue.get()
        self.emit_status(self.exit_message)

    def is_idle(self):
        return self.queue.qsize() == 0 and self.active_tasks == 0


    def emit_status(self, message):
        if message is None:
            return
        self.ui_bridge.status_message.emit(str(message))

    def current_bline_avg(self):
        return max(1, int(self.ui.BlineAVG.value()))


    def current_nsamples(self):
        return effective_camera_sample_count(self.ui)


    def current_depth_start(self):
        return int(self.ui.DepthStart.value())


    def current_depth_range(self):
        return int(self.ui.DepthRange.value())


    def current_fft_result_mode(self):
        return str(self.ui.FFTresults.currentText())


    def current_kx_display_enabled(self):
        item_mode = getattr(getattr(self, "item", None), "acq_mode", None)
        return item_mode == AcqTypes.CONTINUOUS_KX or self.ui.FFTresults.currentText() == "KX"


    def current_fft_output_range(self, samples, pixel_range):
        if self.current_kx_display_enabled():
            return int(samples)
        return pixel_range




    def kx_display_target_max(self):
        target = getattr(self, "kx_display_scale", "linear")
        if target == "linear":
            return 65025
        return 255

    def scale_kx_display_cpu(self, kx_data):
        target = getattr(self, "kx_display_scale", "linear")
        normalize = getattr(self, "kx_normalize_per_frame", False)
        if normalize and hasattr(self, "ui") and hasattr(self.ui, "XZmax"):
            target_max = float(self.ui.XZmax.value())
        else:
            target_max = float(self.kx_display_target_max())
        if target == "log":
            magnitude = np.abs(kx_data)
            magnitude = np.clip(magnitude, 1e-12, None)
            log_mag = np.log(magnitude)
            log_mag -= np.min(log_mag, axis=(1, 2), keepdims=True)
            scale = target_max / (np.max(log_mag, axis=(1, 2), keepdims=True) + 1e-12)
            return (log_mag * scale).astype(np.float32)
        elif target == "sqrt":
            magnitude = np.abs(kx_data)
            sqrt_mag = np.sqrt(magnitude)
            scale = target_max / (np.max(sqrt_mag, axis=(1, 2), keepdims=True) + 1e-12)
            return (sqrt_mag * scale).astype(np.float32)
        else:
            magnitude = np.abs(kx_data)
            scale = target_max / (np.max(magnitude, axis=(1, 2), keepdims=True) + 1e-12)
            return (magnitude * scale).astype(np.float32)

    def scale_kx_display_gpu(self, kx_data):
        target = getattr(self, "kx_display_scale", "linear")
        normalize = getattr(self, "kx_normalize_per_frame", False)
        if normalize and hasattr(self, "ui") and hasattr(self.ui, "XZmax"):
            target_max = float(self.ui.XZmax.value())
        else:
            target_max = float(self.kx_display_target_max())
        if target == "log":
            magnitude = cupy.abs(kx_data)
            magnitude = cupy.clip(magnitude, 1e-12, None)
            log_mag = cupy.log(magnitude)
            log_mag -= cupy.min(log_mag, axis=(1, 2), keepdims=True)
            scale = target_max / (cupy.max(log_mag, axis=(1, 2), keepdims=True) + 1e-12)
            return (log_mag * scale).astype(cupy.float32)
        elif target == "sqrt":
            magnitude = cupy.abs(kx_data)
            sqrt_mag = cupy.sqrt(magnitude)
            scale = target_max / (cupy.max(sqrt_mag, axis=(1, 2), keepdims=True) + 1e-12)
            return (sqrt_mag * scale).astype(cupy.float32)
        else:
            magnitude = cupy.abs(kx_data)
            scale = target_max / (cupy.max(magnitude, axis=(1, 2), keepdims=True) + 1e-12)
            return (magnitude * scale).astype(cupy.float32)

    def current_depth_centered_enabled(self):
        if self.current_kx_display_enabled():
            return False
        return bool(self.depth_center_zero_depth_in_display)

    def depth_crop_start(self, total_samples, pixel_start, pixel_range):
        if self.current_depth_centered_enabled():
            center = total_samples // 2
            half = pixel_range // 2
            return min(max(0, center - half), max(0, total_samples - pixel_range))
        return min(max(0, int(pixel_start)), max(0, int(total_samples) - int(pixel_range)))

    def select_fft_depth_result(self, fft_data, pixel_start, pixel_range, xp):
        samples = int(fft_data.shape[-1])
        start = self.depth_crop_start(samples, pixel_start, pixel_range)
        end = min(start + pixel_range, samples)
        if self.current_depth_centered_enabled():
            # Select the requested window in fftshift order without rearranging
            # the complete depth FFT. fftshift is a roll by samples//2.
            shifted_indices = xp.arange(start, end, dtype=xp.int64)
            native_indices = (shifted_indices - samples // 2) % samples
            return xp.take(fft_data, native_indices, axis=-1)
        return fft_data[..., start:end]

    def current_gpu_avg_count(self, frame_count=None):
        if self.current_kx_display_enabled():
            return 1
        configured = self.current_bline_avg()
        if frame_count is None:
            return configured
        frame_count = max(0, int(frame_count))
        if configured <= 1 or frame_count < configured:
            return 1
        return configured

    @staticmethod
    def averaged_frame_count(frame_count, avg_count):
        return int(frame_count) // max(1, int(avg_count))

    @staticmethod
    def allocate_pinned_array(shape, dtype=np.float32):
        dtype = np.dtype(dtype)
        nbytes = int(np.prod(shape)) * dtype.itemsize
        try:
            owner = cupy.cuda.alloc_pinned_memory(nbytes)
            return np.frombuffer(owner, dtype=dtype, count=int(np.prod(shape))).reshape(shape)
        except Exception as error:
            print(f"Pinned output allocation failed; using pageable NumPy memory: {error}")
            return np.empty(shape, dtype=dtype)

    def drain_returned_pinned_buffers(self):
        returned = 0
        while True:
            try:
                token = self.PinnedOutputFreeQueue.get_nowait()
            except Empty:
                break
            record = self.pinned_output_buffers.get(token)
            if record is not None:
                record["in_use"] = False
                returned += 1
        return returned

    def acquire_pinned_output_buffer(self, shape, dtype=np.float32):
        shape = tuple(int(value) for value in shape)
        dtype = np.dtype(dtype)
        self.drain_returned_pinned_buffers()

        def matching_free_record():
            candidates = []
            for token, record in self.pinned_output_buffers.items():
                allocated_shape = record["shape"]
                compatible = (
                    not record["in_use"]
                    and record["dtype"] == dtype
                    and len(allocated_shape) == len(shape)
                    and allocated_shape[1:] == shape[1:]
                    and allocated_shape[0] >= shape[0]
                )
                if compatible:
                    candidates.append((allocated_shape[0], token, record))
            if candidates:
                _, token, record = min(candidates, key=lambda candidate: candidate[0])
                return token, record
            return None, None

        def requested_view(record):
            return record["array"][:shape[0], ...]

        token, record = matching_free_record()
        if record is not None:
            record["in_use"] = True
            return requested_view(record), token, 0.0

        while len(self.pinned_output_buffers) >= self.pinned_output_pool_size:
            reusable_token = next(
                (key for key, value in self.pinned_output_buffers.items() if not value["in_use"]),
                None,
            )
            if reusable_token is not None:
                del self.pinned_output_buffers[reusable_token]
                break
            returned_token = self.PinnedOutputFreeQueue.get()
            returned_record = self.pinned_output_buffers.get(returned_token)
            if returned_record is not None:
                returned_record["in_use"] = False
            token, record = matching_free_record()
            if record is not None:
                record["in_use"] = True
                return requested_view(record), token, 0.0

        allocation_start = time.perf_counter()
        array = self.allocate_pinned_array(shape, dtype=dtype)
        allocation_seconds = time.perf_counter() - allocation_start
        token = self.next_pinned_output_token
        self.next_pinned_output_token += 1
        self.pinned_output_buffers[token] = {
            "array": array,
            "shape": shape,
            "dtype": dtype,
            "in_use": True,
        }
        return array, token, allocation_seconds

    def gpu_timing_start(self, stream=None):
        if self.gpu_profile_timing_enabled and stream is not None:
            stream.synchronize()
        return time.perf_counter()

    def gpu_timing_end(self, timing, label, start, stream=None):
        if self.gpu_profile_timing_enabled and stream is not None:
            stream.synchronize()
        elapsed = time.perf_counter() - start
        if self.gpu_profile_timing_enabled:
            prev = timing.get(label, 0.0)
            timing[label] = prev + elapsed

    def gpu_timing_synchronize(self, stream=None):
        if stream is not None:
            stream.synchronize()
        else:
            cupy.cuda.get_current_stream().synchronize()

    def format_gpu_timing_summary(self, timing):
        if not timing:
            return ""
        wall = timing.get("_wall_seconds", 0.0)
        parts = [f"wall={wall*1000:.1f}ms"]
        stage_order = (
            "prepare_request",
            "pinned_buffer_allocate",
            "pinned_buffer_wait",
            "load_raw_to_gpu",
            "convert_to_float",
            "saved_background_subtraction",
            "reshape_alines",
            "interpolation",
            "off_axis_x_filter",
            "fft",
            "select_depth_result",
            "abs_magnitude",
            "reshape_depth",
            "post_fft_scaling",
            "background_x_normalization",
            "post_abs_averaging",
            "copy_gpu_to_host",
            "release_gpu_memory",
        )
        stage_total = 0.0
        for key in stage_order:
            value = timing.get(key)
            if not isinstance(value, (int, float)):
                continue
            stage_total += value
            parts.append(f"{key}={value*1000:.1f}ms")
        unaccounted = max(0.0, wall - stage_total)
        parts.append(f"stage_total={stage_total*1000:.1f}ms")
        parts.append(f"unaccounted={unaccounted*1000:.1f}ms")
        return ", ".join(parts)

    def print_gpu_timing_summary(
        self,
        timing,
        mode,
        acq_mode,
        input_shape,
        input_dtype,
        output_shape,
        frames,
        chunk_frames,
        chunks,
    ):
        if not self.gpu_profile_timing_enabled:
            return
        summary = self.format_gpu_timing_summary(timing)
        input_bytes = int(frames) * int(np.prod(input_shape[1:])) * np.dtype(input_dtype).itemsize
        output_bytes = int(np.prod(output_shape)) * np.dtype(np.float32).itemsize
        wall = max(float(timing.get("_wall_seconds", 0.0)), 1e-12)
        print(
            "GPU profile "
            f"[result={mode}, acq={acq_mode}] "
            f"input={tuple(input_shape)} valid_frames={frames} dtype={input_dtype} "
            f"output={tuple(output_shape)} chunk_frames={chunk_frames} chunks={chunks} "
            f"input={input_bytes / 1048576.0:.1f}MiB "
            f"output={output_bytes / 1048576.0:.1f}MiB "
            f"input_rate={input_bytes / wall / 1048576.0:.1f}MiB/s: "
            f"{summary}"
        )

    def release_memory_slot(self, memory_slot):
        if memory_slot is None or not hasattr(self, "MemoryFreeQueue"):
            return
        self.MemoryFreeQueue.put(memory_slot)

    def valid_input_frame_count(self, memory_slot):
        if memory_slot is None or memory_slot >= len(self.Memory):
            return 0
        slot = self.Memory[memory_slot]
        requested = getattr(getattr(self, "item", None), "package_frame_count", None)
        if requested is None:
            return int(slot.shape[0])
        return min(int(slot.shape[0]), max(0, int(requested)))


    def cudaFFT(self, DnS_action, acq_mode, memory_slot, context):
        timing = {"_wall_start": time.perf_counter()}
        package_profile = dict(getattr(self.item, "profile", {}) or {})
        package_profile["gpu_start_s"] = timing["_wall_start"]
        if package_profile.get("gpu_queue_put_s") is not None:
            package_profile["gpu_queue_wait_s"] = timing["_wall_start"] - package_profile["gpu_queue_put_s"]
        request_start = self.gpu_timing_start()
        samples = self.current_nsamples()
        Pixel_start = self.current_depth_start()
        Pixel_range = self.current_depth_range()
        output_range = self.current_fft_output_range(samples, Pixel_range)
        slot_shape = self.Memory[memory_slot].shape
        valid_input_frames = self.valid_input_frame_count(memory_slot)
        gpu_avg_count = self.current_gpu_avg_count(valid_input_frames)
        effective_frames = self.averaged_frame_count(valid_input_frames, gpu_avg_count) * gpu_avg_count
        if effective_frames <= 0:
            raise ValueError(f"No complete frame group available for GPU averaging by {gpu_avg_count}.")
        if effective_frames != valid_input_frames:
            print(
                f"GPU averaging dropped {valid_input_frames - effective_frames} incomplete trailing frame(s) "
                f"for BlineAVG={gpu_avg_count}."
            )
        chunk_frames = self.gpu_fft_chunk_frames(effective_frames, gpu_avg_count)
        background_reference_gpu = self.determine_background_gpu(memory_slot)
        output_frames = self.averaged_frame_count(effective_frames, gpu_avg_count)
        self.gpu_timing_end(timing, "prepare_request", request_start)

        pinned_start = time.perf_counter()
        self.data_CPU, output_buffer_token, allocation_seconds = self.acquire_pinned_output_buffer(
            (output_frames, slot_shape[1], output_range), np.float32
        )
        pinned_elapsed = time.perf_counter() - pinned_start
        timing["pinned_buffer_allocate"] = timing.get("pinned_buffer_allocate", 0.0) + allocation_seconds
        timing["pinned_buffer_wait"] = timing.get("pinned_buffer_wait", 0.0) + max(
            0.0, pinned_elapsed - allocation_seconds
        )
        self._leased_output_token = output_buffer_token
        self.cudaFFT_chunked_overlapped(
            memory_slot,
            samples,
            Pixel_start,
            Pixel_range,
            chunk_frames,
            gpu_avg_count,
            background_reference_gpu,
            timing,
        )
        del background_reference_gpu
        if self.release_gpu_memory_each_fft:
            release_start = self.gpu_timing_start()
            self.release_gpu_memory()
            self.gpu_timing_end(timing, "release_gpu_memory", release_start)
        timing["_wall_seconds"] = time.perf_counter() - timing["_wall_start"]
        package_profile["gpu_wall_s"] = timing["_wall_seconds"]
        for key, value in timing.items():
            if not key.startswith("_") and isinstance(value, (int, float)):
                package_profile["gpu_{}_s".format(key)] = float(value)
        an_action = DnSActionField(
            DnS_action,
            acq_mode=acq_mode,
            data=self.data_CPU,
            raw=False,
            context=context,
            gpu_avg_count=gpu_avg_count,
            filename_bundle=self.item.filename_bundle,
            skip_save=self.item.skip_save,
            package_start=getattr(self.item, "package_start", 0),
            package_frame_count=getattr(self.item, "package_frame_count", None),
            package_total_frames=getattr(self.item, "package_total_frames", None),
            is_final=getattr(self.item, "is_final", True),
            profile=package_profile,
            output_buffer_token=output_buffer_token,
        )
        an_action.profile["dns_queue_put_s"] = time.perf_counter()
        self.DnSQueue.put(an_action)
        self._leased_output_token = None
        if getattr(self, "return_processed_to_weaver", False):
            self.GPU2weaverQueue.put(self.data_CPU)
        self.print_gpu_timing_summary(
            timing,
            self.current_fft_result_mode(),
            acq_mode,
            slot_shape,
            self.Memory[memory_slot].dtype,
            self.data_CPU.shape,
            effective_frames,
            chunk_frames,
            int(timing.get("_chunks", 0)),
        )

    def fft_cpu(self, DnS_action, acq_mode, memory_slot, context):
        cpu_start_s = time.perf_counter()
        package_profile = dict(getattr(self.item, "profile", {}) or {})
        samples = self.current_nsamples()
        pixel_start = self.current_depth_start()
        pixel_range = self.current_depth_range()
        valid_input_frames = self.valid_input_frame_count(memory_slot)
        gpu_avg_count = self.current_gpu_avg_count(valid_input_frames)
        effective_frames = self.averaged_frame_count(valid_input_frames, gpu_avg_count) * gpu_avg_count
        self.data_CPU = self.Memory[memory_slot][:effective_frames].astype(np.float32, copy=True)
        processed_shape = self.data_CPU.shape
        background_reference_cpu = self.determine_background_cpu(memory_slot)
        self.apply_saved_background_subtraction_cpu(self.data_CPU, background_reference_cpu)
        alines = processed_shape[0] * processed_shape[1]
        self.data_CPU = self.data_CPU.reshape([alines, samples])
        if self.interp:
            self.data_CPU = self.interpolate_cpu(self.data_CPU)
        if self.current_kx_display_enabled():
            self.data_CPU = self.data_CPU.reshape(processed_shape[0], processed_shape[1], samples)
            self.data_CPU = np.fft.fftshift(np.fft.fft(self.data_CPU, axis=1), axes=1)
            self.data_CPU = self.scale_kx_display_cpu(self.data_CPU)
        else:
            if self.interp:
                self.data_CPU = np.fft.fft(self.data_CPU * self.dispersion, axis=1) / samples
            else:
                self.data_CPU = np.fft.fft(self.data_CPU, axis=1) / samples
            self.data_CPU = self.select_fft_depth_result(self.data_CPU, pixel_start, pixel_range, np)
            self.data_CPU = np.abs(self.data_CPU, dtype=np.float32)
            self.data_CPU = self.data_CPU.reshape(processed_shape[0], processed_shape[1], pixel_range)
        if not self.current_kx_display_enabled():
            self.data_CPU *= np.float32(self.AMPLIFICATION)
            self.apply_background_x_normalization_cpu(self.data_CPU)
            if gpu_avg_count > 1:
                self.data_CPU = self.data_CPU.reshape(
                    effective_frames // gpu_avg_count,
                    gpu_avg_count,
                    processed_shape[1],
                    pixel_range,
                ).mean(axis=1, dtype=np.float32)
        an_action = DnSActionField(
            DnS_action,
            acq_mode=acq_mode,
            data=self.data_CPU,
            raw=False,
            context=context,
            gpu_avg_count=gpu_avg_count,
            filename_bundle=self.item.filename_bundle,
            skip_save=self.item.skip_save,
            package_start=getattr(self.item, "package_start", 0),
            package_frame_count=getattr(self.item, "package_frame_count", None),
            package_total_frames=getattr(self.item, "package_total_frames", None),
            is_final=getattr(self.item, "is_final", True),
            profile=package_profile,
        )
        package_profile["cpu_wall_s"] = time.perf_counter() - cpu_start_s
        an_action.profile["cpu_wall_s"] = package_profile["cpu_wall_s"]
        an_action.profile["dns_queue_put_s"] = time.perf_counter()
        self.DnSQueue.put(an_action)
        if getattr(self, "return_processed_to_weaver", False):
            self.GPU2weaverQueue.put(self.data_CPU)

    def gpu_fft_chunk_frames(self, total_frames, avg_count=1):
        total_frames = max(1, int(total_frames))
        avg_count = max(1, int(avg_count))
        requested = max(avg_count, int(self.gpu_chunk_frames))
        aligned = max(avg_count, (requested // avg_count) * avg_count)
        return min(total_frames, aligned)

    def load_raw_chunk_to_gpu(self, raw_chunk, slot=None, stream=None):
        raw_gpu = self.gpu_raw_buffer(raw_chunk.shape, raw_chunk.dtype, slot=slot)
        if stream is None:
            raw_gpu.set(raw_chunk)
        else:
            try:
                raw_gpu.set(raw_chunk, stream=stream)
            except TypeError:
                raw_gpu.set(raw_chunk)
        return raw_gpu



    def prepare_float_chunk(self, raw_gpu, slot=None):
        y_gpu = self.gpu_float_buffer(raw_gpu.shape, slot=slot)
        y_gpu[...] = raw_gpu
        return y_gpu

    def apply_saved_background_subtraction_gpu(self, y_gpu, background_reference_gpu=None):
        if background_reference_gpu is not None:
            y_gpu -= background_reference_gpu
            return True

    def apply_saved_background_subtraction_cpu(self, data_cpu, background_reference_cpu=None):
        if background_reference_cpu is not None:
            data_cpu -= background_reference_cpu
            return True

    def apply_background_x_normalization_cpu(self, data_cpu):
        chunk_shape = data_cpu.shape
        if self.background_x_normalization is None:
            return False
        if self.background_x_normalization.size != chunk_shape[1]:
            print(
                "Background X normalization mismatch. Skipped: ",
                self.background_x_normalization.size,
                "data X pixels:",
                chunk_shape[1],
            )
            return False
        data_cpu /= self.background_x_normalization[np.newaxis, :, np.newaxis]

    def interpolate_cpu(self, data_cpu):
        idx0 = self.indice[0, :].astype(np.intp, copy=False)
        idx1 = self.indice[1, :].astype(np.intp, copy=False)
        x0 = self.intpX[idx0]
        x1 = self.intpX[idx1]
        xt = self.intpXp
        y0 = data_cpu[:, idx0]
        y1 = data_cpu[:, idx1]
        denominator = x1 - x0
        denominator = np.where(np.abs(denominator) < 1e-5, np.float32(1e-5), denominator)
        return y0 + (xt - x0) * (y1 - y0) / denominator

    def apply_background_x_normalization_gpu(self, y_gpu):
        if self.background_x_normalization_gpu is None:
            return False
        chunk_shape = y_gpu.shape
        ref = self.background_x_normalization_gpu
        if ref.size != chunk_shape[1]:
            return False
        y_gpu /= ref[cupy.newaxis, :, cupy.newaxis]

    def cudaFFT_chunked_overlapped(
        self,
        memory_slot,
        samples,
        Pixel_start,
        Pixel_range,
        chunk_frames,
        gpu_avg_count,
        background_reference_gpu,
        timing=None,
    ):
        if timing is None:
            timing = {}
        total_input_frames = self.averaged_frame_count(
            self.valid_input_frame_count(memory_slot), gpu_avg_count
        ) * gpu_avg_count
        streams = self.gpu_overlap_streams()
        slot_refs = [None, None]

        chunk_index = 0
        input_start = 0
        output_start = 0
        while input_start < total_input_frames:
            slot = chunk_index % 2
            if slot_refs[slot] is not None:
                slot_refs[slot]["stream"].synchronize()
                slot_refs[slot] = None

            input_end = min(input_start + chunk_frames, total_input_frames)
            chunk = self.Memory[memory_slot][input_start:input_end, :, :]
            output_count = self.averaged_frame_count(input_end - input_start, gpu_avg_count)
            output_end = output_start + output_count
            host_out = self.data_CPU[output_start:output_end, :, :]

            slot_refs[slot] = self.cudaFFT_chunk_async(
                chunk,
                samples,
                Pixel_start,
                Pixel_range,
                gpu_avg_count,
                background_reference_gpu,
                streams[slot],
                slot,
                host_out,
                input_start,
                timing,
            )
            timing["_chunks"] = timing.get("_chunks", 0) + 1
            input_start = input_end
            output_start = output_end
            chunk_index += 1

        for slot in range(2):
            if slot_refs[slot] is not None:
                slot_refs[slot]["stream"].synchronize()
                slot_refs[slot] = None

    def cudaFFT_chunk_async(
        self,
        raw_chunk,
        samples,
        Pixel_start,
        Pixel_range,
        gpu_avg_count,
        background_reference_gpu,
        stream,
        slot,
        host_out,
        frame_offset=0,
        timing=None,
    ):
        if timing is None:
            timing = {}
        with stream:
            data_gpu, keep_alive = self.process_chunk_gpu(
                raw_chunk,
                samples,
                Pixel_start,
                Pixel_range,
                gpu_avg_count,
                background_reference_gpu,
                slot=slot,
                stream=stream,
                timing=timing,
            )
            keep_alive.append(data_gpu)
            copy_start = self.gpu_timing_start(stream)
            self.copy_gpu_to_host_async(data_gpu, host_out, stream)
            self.gpu_timing_end(timing, "copy_gpu_to_host", copy_start, stream)
        return {
            "stream": stream,
            "keep_alive": keep_alive,
        }

    def current_holo_x_filter_enabled(self):
        return bool(self.holo_x_filter_enabled)


    def current_camera_pixel_size_m(self):
        camera = get_camera_spec(self.ui.Camera.currentText())
        if camera is None:
            return 9.0e-6
        return float(camera.pixel_size_um) * 1e-6





    def holo_center_wavelength_m(self):
        return float(self.holo_center_wavelength_nm) * 1e-9


    def holo_wavelength_range_m(self):
        return float(self.holo_wavelength_range_nm) * 1e-9


    def holo_reference_tilt_rad(self):
        return float(self.holo_reference_tilt_deg) * np.pi / 180.0


    def holo_wavenumber_axis_gpu(self, samples):
        center = self.holo_center_wavelength_m()
        span = self.holo_wavelength_range_m()
        lambda_start = center - span / 2.0
        lambda_end = center + span / 2.0
        k_start = 2.0 * np.pi / lambda_end
        k_end = 2.0 * np.pi / lambda_start
        return cupy.linspace(k_start, k_end, int(samples), dtype=cupy.float32)

    def holo_filter_weights_gpu(self, x_pixels):
        x_pixels = int(x_pixels)
        pixel_size_m = self.current_camera_pixel_size_m()
        cache_key = (
            x_pixels, pixel_size_m, float(self.holo_reference_tilt_deg),
            float(self.holo_center_wavelength_nm), int(self.holo_sideband_sign),
            float(self.holo_sideband_half_width_fraction),
            float(self.holo_sideband_taper_fraction),
            self.holo_x_filter_center_fraction_override,
            bool(self.holo_keep_high_frequency_tail),
        )
        cached = self.holo_filter_cache.get(cache_key)
        if cached is not None:
            return cached
        # Native FFT order eliminates fftshift/ifftshift around the X filter.
        kx_axis = cupy.fft.fftfreq(x_pixels, d=pixel_size_m).astype(cupy.float32)
        kx_axis *= cupy.float32(2.0 * np.pi)
        kx_nyquist = np.pi / pixel_size_m
        sign = 1.0 if int(self.holo_sideband_sign) >= 0 else -1.0
        override = self.holo_x_filter_center_fraction_override
        if override is None:
            center_abs = 2.0 * np.pi * abs(np.sin(self.holo_reference_tilt_rad())) / self.holo_center_wavelength_m()
            center = sign * center_abs
        else:
            center = float(override) * kx_nyquist
            center_abs = abs(center)
        half_width = max(center_abs * float(self.holo_sideband_half_width_fraction), kx_nyquist / max(x_pixels, 1))
        taper = np.clip(float(self.holo_sideband_taper_fraction), 0.0, 0.95) * half_width
        weights = cupy.zeros((x_pixels,), dtype=cupy.float32)
        if self.holo_keep_high_frequency_tail:
            signed_kx = cupy.float32(sign) * kx_axis
            lower_edge = center_abs - half_width
            if taper > 0:
                flat = signed_kx >= lower_edge + taper
                roll = (signed_kx > lower_edge) & (signed_kx < lower_edge + taper)
                weights[flat] = cupy.float32(1.0)
                weights[roll] = cupy.float32(0.5) * (
                    cupy.float32(1.0) - cupy.cos(cupy.float32(np.pi) * (signed_kx[roll] - lower_edge) / taper)
                )
            else:
                weights[signed_kx > lower_edge] = cupy.float32(1.0)
        else:
            dist = cupy.abs(kx_axis - cupy.float32(center))
            if taper > 0:
                flat = dist <= (half_width - taper)
                roll = (dist > (half_width - taper)) & (dist < half_width)
                weights[flat] = cupy.float32(1.0)
                weights[roll] = cupy.float32(0.5) * (
                    cupy.float32(1.0) + cupy.cos(cupy.float32(np.pi) * (dist[roll] - (half_width - taper)) / taper)
                )
            else:
                weights[dist < half_width] = cupy.float32(1.0)
        if self.holo_print_filter_info and not self.holo_filter_info_printed:
            self.holo_filter_info_printed = True
            print(
                "Off-axis X filter: "
                f"enabled={self.holo_x_filter_enabled}, sign={int(sign)}, "
                f"center/Nyquist={center / kx_nyquist:.3f}, "
                f"half_width/Nyquist={half_width / kx_nyquist:.3f}, "
                f"taper_fraction={float(self.holo_sideband_taper_fraction):.2f}, "
                f"pixel={pixel_size_m * 1e6:.2f} um"
            )
        self.holo_filter_cache = {cache_key: weights}
        return weights

    def holo_carrier_gpu(self, x_pixels, samples):
        pixel_size_m = self.current_camera_pixel_size_m()
        cache_key = (
            int(x_pixels), int(samples), pixel_size_m,
            float(self.holo_reference_tilt_deg),
            float(self.holo_center_wavelength_nm),
            float(self.holo_wavelength_range_nm), int(self.holo_sideband_sign),
        )
        cached = self.holo_carrier_cache.get(cache_key)
        if cached is not None:
            return cached
        x_axis = cupy.arange(int(x_pixels), dtype=cupy.float32)
        x_axis -= cupy.float32((int(x_pixels) - 1) / 2.0)
        x_axis *= cupy.float32(pixel_size_m)
        k_axis = self.holo_wavenumber_axis_gpu(samples)
        sign = 1.0 if int(self.holo_sideband_sign) >= 0 else -1.0
        carrier = cupy.exp(
            -1j
            * cupy.float32(sign * np.sin(self.holo_reference_tilt_rad()))
            * k_axis[cupy.newaxis, :]
            * x_axis[:, cupy.newaxis]
        ).astype(cupy.complex64, copy=False)
        self.holo_carrier_cache = {cache_key: carrier}
        return carrier

    def apply_holo_x_filter_gpu(self, spectral_gpu, frames, x_pixels, samples):
        field_gpu = spectral_gpu.reshape(int(frames), int(x_pixels), int(samples)).astype(cupy.complex64, copy=False)
        weights_gpu = self.holo_filter_weights_gpu(x_pixels)
        if self.holo_x_contiguous_fft:
            # cuFFT is substantially happier when X is the contiguous axis.
            field_x_last = cupy.ascontiguousarray(field_gpu.transpose(0, 2, 1))
            filtered_kx_gpu = cupy.fft.fft(field_x_last, axis=-1)
            filtered_kx_gpu *= weights_gpu[cupy.newaxis, cupy.newaxis, :]
            field_gpu = cupy.fft.ifft(filtered_kx_gpu, axis=-1).transpose(0, 2, 1)
        else:
            filtered_kx_gpu = cupy.fft.fft(field_gpu, axis=1)
            filtered_kx_gpu *= weights_gpu[cupy.newaxis, :, cupy.newaxis]
            field_gpu = cupy.fft.ifft(filtered_kx_gpu, axis=1)
        if self.holo_reference_carrier_demodulation:
            carrier = self.holo_carrier_gpu(x_pixels, samples)
            field_gpu *= carrier[cupy.newaxis, :, :]
            if self.holo_conjugate_positive_sideband:
                field_gpu = cupy.conj(field_gpu)
        if self.holo_subtract_common_mode_after_demod:
            field_gpu = field_gpu - cupy.median(field_gpu, axis=1, keepdims=True)
        return field_gpu.reshape(int(frames) * int(x_pixels), int(samples))

    def gpu_interpolation_buffer(self, shape, slot=None):
        if slot is None:
            if self.yp_gpu_buffer is None or self.yp_gpu_buffer.shape != shape:
                self.yp_gpu_buffer = cupy.empty(shape, dtype=cupy.float32)
            return self.yp_gpu_buffer
        if self.yp_gpu_stream_buffers[slot] is None or self.yp_gpu_stream_buffers[slot].shape != shape:
            self.yp_gpu_stream_buffers[slot] = cupy.empty(shape, dtype=cupy.float32)
        return self.yp_gpu_stream_buffers[slot]

    def process_chunk_gpu(
        self,
        raw_chunk,
        samples,
        Pixel_start,
        Pixel_range,
        gpu_avg_count,
        background_reference_gpu,
        slot=None,
        stream=None,
        timing=None,
    ):
        if timing is None:
            timing = {}
        chunk_shape = raw_chunk.shape
        output_frames = chunk_shape[0]
        keep_alive = []
        chunk_start = time.perf_counter() if self.gpu_profile_timing_print_chunks else None

        step_start = self.gpu_timing_start(stream)
        raw_gpu = self.load_raw_chunk_to_gpu(raw_chunk, slot=slot, stream=stream)
        self.gpu_timing_end(timing, "load_raw_to_gpu", step_start, stream)
        keep_alive.append(raw_gpu)

        step_start = self.gpu_timing_start(stream)
        y_gpu = self.prepare_float_chunk(raw_gpu, slot=slot)
        self.gpu_timing_end(timing, "convert_to_float", step_start, stream)
        keep_alive.append(y_gpu)

        step_start = self.gpu_timing_start(stream)
        self.apply_saved_background_subtraction_gpu(y_gpu, background_reference_gpu)
        self.gpu_timing_end(timing, "saved_background_subtraction", step_start, stream)

        step_start = self.gpu_timing_start(stream)
        alines = y_gpu.shape[0] * y_gpu.shape[1]
        y_gpu = y_gpu.reshape([alines, samples])
        self.gpu_timing_end(timing, "reshape_alines", step_start, stream)
        keep_alive.append(y_gpu)

        step_start = self.gpu_timing_start(stream)
        if self.interp:
            self.ensure_dispersion_gpu_cache()
            yp_gpu = self.gpu_interpolation_buffer(y_gpu.shape, slot=slot)
            self.interp_kernel(
                (8, 8),
                (16, 16),
                (
                    alines,
                    samples,
                    self.intpX_gpu,
                    self.intpXp_gpu,
                    y_gpu,
                    self.indice1_gpu,
                    self.indice2_gpu,
                    yp_gpu,
                ),
            )
        else:
            yp_gpu = y_gpu
        self.gpu_timing_end(timing, "interpolation", step_start, stream)
        keep_alive.append(yp_gpu)

        if self.current_holo_x_filter_enabled() and not self.current_kx_display_enabled():
            step_start = self.gpu_timing_start(stream)
            yp_gpu = self.apply_holo_x_filter_gpu(yp_gpu, output_frames, chunk_shape[1], samples)
            self.gpu_timing_end(timing, "off_axis_x_filter", step_start, stream)
            keep_alive.append(yp_gpu)

        if self.current_kx_display_enabled():
            step_start = self.gpu_timing_start(stream)
            kx_gpu = cupy.fft.fftshift(
                cupy.fft.fft(
                    yp_gpu.reshape(output_frames, chunk_shape[1], samples),
                    axis=1,
                ),
                axes=1,
            )
            data_gpu = self.scale_kx_display_gpu(kx_gpu)
            self.gpu_timing_end(timing, "fft", step_start, stream)
        else:
            step_start = self.gpu_timing_start(stream)
            if self.interp:
                data_gpu = cupy.fft.fft(yp_gpu * self.dispersion_gpu, axis=1) / samples
            else:
                data_gpu = cupy.fft.fft(yp_gpu, axis=1) / samples
            self.gpu_timing_end(timing, "fft", step_start, stream)

            step_start = self.gpu_timing_start(stream)
            data_gpu = self.select_fft_depth_result(data_gpu, Pixel_start, Pixel_range, cupy)
            self.gpu_timing_end(timing, "select_depth_result", step_start, stream)

            step_start = self.gpu_timing_start(stream)
            data_gpu = cupy.abs(data_gpu)
            self.gpu_timing_end(timing, "abs_magnitude", step_start, stream)

        step_start = self.gpu_timing_start(stream)
        output_range = self.current_fft_output_range(samples, Pixel_range)
        data_gpu = data_gpu.reshape(output_frames, chunk_shape[1], output_range)
        self.gpu_timing_end(timing, "reshape_depth", step_start, stream)

        step_start = self.gpu_timing_start(stream)
        if not self.current_kx_display_enabled():
            data_gpu *= cupy.float32(self.AMPLIFICATION)
        self.gpu_timing_end(timing, "post_fft_scaling", step_start, stream)

        if not self.current_kx_display_enabled():
            step_start = self.gpu_timing_start(stream)
            self.apply_background_x_normalization_gpu(data_gpu)
            self.gpu_timing_end(timing, "background_x_normalization", step_start, stream)

        if gpu_avg_count > 1:
            step_start = self.gpu_timing_start(stream)
            complete_frames = (output_frames // gpu_avg_count) * gpu_avg_count
            data_gpu = data_gpu[:complete_frames].reshape(
                complete_frames // gpu_avg_count,
                gpu_avg_count,
                chunk_shape[1],
                output_range,
            ).mean(axis=1, dtype=cupy.float32)
            output_frames = complete_frames // gpu_avg_count
            self.gpu_timing_end(timing, "post_abs_averaging", step_start, stream)

        if self.gpu_profile_timing_print_chunks and chunk_start is not None:
            self.gpu_timing_synchronize(stream)
            print(
                "GPU chunk timing [{}], frames={}, slot={}: {:.2f} ms".format(
                    self.current_fft_result_mode(),
                    output_frames,
                    slot,
                    (time.perf_counter() - chunk_start) * 1000.0,
                )
            )
        return data_gpu, keep_alive

    def copy_gpu_to_host_async(self, data_gpu, host_out, stream):
        try:
            cupy.asnumpy(data_gpu, out=host_out, stream=stream, blocking=False)
        except TypeError:
            cupy.asnumpy(data_gpu, out=host_out, stream=stream)

    def gpu_overlap_streams(self):
        if self.gpu_streams is None and self.gpu_overlap_transfer:
            self.gpu_streams = [cupy.cuda.Stream(non_blocking=True) for _ in range(2)]
        if self.gpu_streams is None:
            self.gpu_streams = [cupy.cuda.Stream.null, cupy.cuda.Stream.null]
        return self.gpu_streams

    def gpu_raw_buffer(self, shape, dtype, slot=None):
        if slot is None:
            if self.raw_gpu_buffer is None or self.raw_gpu_buffer.shape != shape or self.raw_gpu_buffer.dtype != dtype:
                self.raw_gpu_buffer = cupy.empty(shape, dtype=dtype)
            return self.raw_gpu_buffer
        if self.raw_gpu_stream_buffers[slot] is None or self.raw_gpu_stream_buffers[slot].shape != shape or self.raw_gpu_stream_buffers[slot].dtype != dtype:
            self.raw_gpu_stream_buffers[slot] = cupy.empty(shape, dtype=dtype)
        return self.raw_gpu_stream_buffers[slot]

    def gpu_float_buffer(self, shape, slot=None):
        if slot is None:
            if self.float_gpu_buffer is None or self.float_gpu_buffer.shape != shape:
                self.float_gpu_buffer = cupy.empty(shape, dtype=cupy.float32)
            return self.float_gpu_buffer
        if self.float_gpu_stream_buffers[slot] is None or self.float_gpu_stream_buffers[slot].shape != shape:
            self.float_gpu_stream_buffers[slot] = cupy.empty(shape, dtype=cupy.float32)
        return self.float_gpu_stream_buffers[slot]

    def determine_background_gpu(self, memory_slot):
        shape = self.Memory[memory_slot].shape
        if self.bg_sub and self.background.shape == shape[1:]:
            if self.background_gpu is None or self.background_gpu.shape != self.background.shape:
                self.background_gpu = cupy.asarray(self.background, dtype=cupy.float32)
            return self.background_gpu
        return None






    def determine_background_cpu(self, memory_slot):
        shape = self.Memory[memory_slot].shape
        if self.bg_sub and self.background.shape == shape[1:]:
            return np.asarray(self.background[np.newaxis, :, :], dtype=np.float32)
        return None

    def ensure_dispersion_gpu_cache(self):
        if self.dispersion_gpu is None:
            self.update_Dispersion()
        intpX_gpu = getattr(self, "intpX_gpu", None)
        if intpX_gpu is None:
            self.update_Dispersion()

    def update_Dispersion(self):
        samples = self.current_nsamples()
        interp_path = self.ui.InD_DIR.text()
        self.intpX = None
        self.intpXp = None
        self.indice1 = None
        self.indice2 = None
        self.indice = None
        self.dispersion = None
        self.intpX_gpu = None
        self.intpXp_gpu = None
        self.indice1_gpu = None
        self.indice2_gpu = None
        self.dispersion_gpu = None
        if os.path.isfile(interp_path + "/intpX.bin"):
            self.intpX = np.fromfile(interp_path + "/intpX.bin", dtype=np.float32)
            self.intpX_gpu = cupy.asarray(self.intpX, dtype=cupy.float32)
        if os.path.isfile(interp_path + "/intpXp.bin"):
            self.intpXp = np.fromfile(interp_path + "/intpXp.bin", dtype=np.float32)
            self.intpXp_gpu = cupy.asarray(self.intpXp, dtype=cupy.float32)
        if os.path.isfile(interp_path + "/indice1.bin"):
            self.indice1 = np.fromfile(interp_path + "/indice1.bin", dtype=np.uint16)
            self.indice1_gpu = cupy.asarray(self.indice1, dtype=cupy.ushort)
        if os.path.isfile(interp_path + "/indice2.bin"):
            self.indice2 = np.fromfile(interp_path + "/indice2.bin", dtype=np.uint16)
            self.indice2_gpu = cupy.asarray(self.indice2, dtype=cupy.ushort)
        if self.indice1 is not None and self.indice2 is not None:
            self.indice = np.vstack([self.indice1, self.indice2])
        if os.path.isfile(interp_path + "/dspPhase.bin"):
            self.dispersion = np.fromfile(interp_path + "/dspPhase.bin", dtype=np.complex64)
            self.dispersion_gpu = cupy.asarray(self.dispersion, dtype=cupy.complex64)
        arrays = (self.intpX, self.intpXp, self.indice1, self.indice2, self.dispersion)
        self.interp = all(array is not None and array.size == samples for array in arrays)
        if not self.interp and any(array is not None for array in arrays):
            message = (
                "Interpolation/dispersion calibration is incomplete or does not match "
                f"the current spectral sample count ({samples}). Processing without calibration."
            )
            print(message)
            self.emit_status(message)

    def update_background(self):
        samples = self.current_nsamples()
        Xpixels = self.ui.AlinesPerBline.value()
        background_path = self.ui.BG_DIR.text()
        if os.path.isfile(background_path):
            try:
                self.background = np.float32(np.fromfile(background_path, dtype=np.float32)).reshape([Xpixels, samples])
                self.update_background_x_normalization()
                self.background_gpu = cupy.asarray(self.background, dtype=cupy.float32)
                if self.background_x_normalization is not None:
                    self.background_x_normalization_gpu = cupy.asarray(
                        self.background_x_normalization,
                        dtype=cupy.float32,
                    )
                message = "Background file loaded."
                self.emit_status(message)
                print(message)
                self.bg_sub = True
            except (OSError, ValueError) as error:
                self.bg_sub = False
                self.background = np.zeros([Xpixels, samples])
                self.background_gpu = None
                self.background_x_normalization = None
                self.background_x_normalization_gpu = None
                message = f"Background file could not be loaded. Using zero background. {error}"
                self.emit_status(message)
                print(message)
            except Exception as error:
                self.bg_sub = False
                self.background = np.zeros([Xpixels, samples])
                self.background_gpu = None
                self.background_x_normalization = None
                self.background_x_normalization_gpu = None
                message = f"Background load failed unexpectedly. Using zero background. {error}"
                self.emit_status(message)
                print(message)
        else:
            self.bg_sub = False
            self.background = np.zeros([Xpixels, samples])
            self.background_gpu = None
            self.background_x_normalization = None
            self.background_x_normalization_gpu = None
            message = "No background file selected. Using zero background."
            self.emit_status(message)
            print(message)

    def update_background_x_normalization(self):
        if self.background is None or self.background.size == 0:
            self.background_x_normalization = None
            self.background_x_normalization_gpu = None
            return False
        x_profile = np.mean(self.background, axis=1, dtype=np.float32)
        profile_mean = float(np.mean(x_profile))
        if not np.isfinite(profile_mean) or profile_mean <= self.background_x_normalization_eps:
            self.background_x_normalization = None
            self.background_x_normalization_gpu = None
            return False
        normalized = x_profile / np.float32(profile_mean)
        bad = (~np.isfinite(normalized)) | (np.abs(normalized) <= np.float32(self.background_x_normalization_eps))
        if np.any(bad):
            normalized[bad] = np.float32(1.0)
        root_order = float(self.background_x_normalization_root_order)
        if np.isfinite(root_order) and root_order > 1.0:
            normalized = np.power(normalized, np.float32(1.0 / root_order), dtype=np.float32)
        self.background_x_normalization = np.asarray(normalized, dtype=np.float32)
        self.background_x_normalization_gpu = None

    def display_FFT_actions(self):
        if GPU_ROUTINE_DIAGNOSTICS:
            print(f"{self.FFT_actions} FFT request(s) processed.")
        self.FFT_actions = 0

    def release_gpu_memory(self):
        self.yp_gpu_buffer = None
        self.raw_gpu_buffer = None
        self.float_gpu_buffer = None
        self.yp_gpu_stream_buffers = [None, None]
        self.raw_gpu_stream_buffers = [None, None]
        self.float_gpu_stream_buffers = [None, None]
        self.gpu_streams = None
        self.holo_filter_cache = {}
        self.holo_carrier_cache = {}













