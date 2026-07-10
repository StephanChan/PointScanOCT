# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 16:50:25 2023

@author: admin
"""
from PyQt5.QtCore import  QThread
from scipy.ndimage import gaussian_filter, uniform_filter1d
import cupy
import numpy as np
from ActionFields import DnSActionField
import os
import time
import traceback
from ActionTypes import AcqTypes, DnSActions, EXIT_ACTION, GPUActions
from CameraUi import effective_camera_sample_count
from HardwareSpecs import get_camera_spec
from scipy.ndimage import zoom, uniform_filter

# =============================================================================
# GPU / dynamic processing tuning knobs
# =============================================================================
# Keep the commonly tuned acquisition-processing settings here so they are easy
# to find after long breaks. These values are copied into each GPUThread object
# during initialization.

# GPU performance -------------------------------------------------------------
# Set True to free temporary CuPy memory after each FFT action. This can reduce
# GPU memory pressure, but may slow repeated processing because buffers are
# recreated more often.
GPU_RELEASE_MEMORY_EACH_FFT = False

# Number of frames processed in each GPU FFT chunk. Increase this if the GPU has
# enough memory and you want fewer chunk launches; decrease it if you hit memory
# limits during large stacks.
GPU_FFT_CHUNK_FRAMES = 8

# Set True to use two CUDA streams and overlap host-to-GPU transfer with GPU
# calculation. Usually leave enabled unless debugging transfer/order issues.
GPU_OVERLAP_TRANSFER = True

# Spectral baseline subtraction window in camera samples. The raw spectrum is
# smoothed along each A-line with this uniform-filter window, then subtracted
# before interpolation/dispersion/FFT. Must be an odd integer; values above 513
# are clamped because of the current CUDA kernel halo loading pattern.
GPU_SPECTRAL_BASELINE_WINDOW_SIZE = 35

# Off-axis holography X-sideband filtering -----------------------------------
# Set True to enable: FFT along X, keep one spatial sideband, IFFT back, then
# remove the reference carrier before the axial FFT. Leave False for normal OCT.
HOLO_X_FILTER_ENABLED = False

# Physical reference tilt model. The sideband center is calculated as
# kx = 2*pi*sin(theta)/lambda0 unless HOLO_X_FILTER_CENTER_FRACTION_OVERRIDE
# is set below.
HOLO_REFERENCE_TILT_DEG = 1.3
HOLO_CENTER_WAVELENGTH_NM = 840.0
HOLO_WAVELENGTH_RANGE_NM = 40.0

# +1 keeps the positive-kx sideband; -1 keeps the negative-kx sideband.
HOLO_SIDEBAND_SIGN = +1

# Sideband window width/taper as fractions of the sideband center magnitude.
HOLO_SIDEBAND_HALF_WIDTH_FRACTION = 0.65
HOLO_SIDEBAND_TAPER_FRACTION = 0.30

# Optional direct tuning: set to a signed fraction of kx Nyquist, e.g. +0.35.
# None means use HOLO_REFERENCE_TILT_DEG and camera pixel size.
HOLO_X_FILTER_CENTER_FRACTION_OVERRIDE = None

# Demodulate exp(-i*k*x*sin(theta)) after IFFT. For measured/manual carrier
# tuning, leave this enabled only when HOLO_REFERENCE_TILT_DEG is correct.
HOLO_REFERENCE_CARRIER_DEMODULATION = True

# With the sign convention used here, the positive sideband is often R*conj(E),
# so conjugating it restores the sample axial phase sign.
HOLO_CONJUGATE_POSITIVE_SIDEBAND = True

# Set True to keep all high frequencies beyond the low edge instead of using a
# finite band around the sideband center. Usually False is cleaner.
HOLO_KEEP_HIGH_FREQUENCY_TAIL = False

# Set True to subtract the median spectrum over X after demodulation.
HOLO_SUBTRACT_COMMON_MODE_AFTER_DEMOD = False

# Set True to print the chosen carrier/window once when the first chunk runs.
HOLO_PRINT_FILTER_INFO = True

# Set True to print GPU processing timing for each FFT request. Timing forces
# CUDA synchronization around each profiled step, so use it for diagnostics and
# set it back to False for maximum acquisition throughput.
GPU_PROFILE_TIMING_ENABLED = False

# Set True to print one timing line per processed GPU chunk. False prints only
# one compact summary per FFT request.
GPU_PROFILE_TIMING_PRINT_CHUNKS = False

# Dynamic processing is disabled in ASOCT; OCT intensity is the only output mode.
DYNAMIC_TEMPORAL_LOWPASS_ENABLED = False

# Temporal uniform-filter window size in frames. Only used when
# DYNAMIC_TEMPORAL_LOWPASS_ENABLED is True. Larger values suppress faster
# temporal fluctuations; 1 is equivalent to no temporal filtering.
DYNAMIC_TEMPORAL_LOWPASS_WINDOW_SIZE = 1

# Spatial Gaussian smoothing sigma applied to the final dynamic image. Set 0 to
# disable final image smoothing.
DYNAMIC_GAUSSIAN_SMOOTHING = False

# Multiplicative display/output scaling for dynamic images. This does not change
# the underlying dynamic algorithm; it only scales the final numeric image.
DYNAMIC_MAGNIFICATION = 1

# Set True to remove the strongest shared temporal/spatial modes before dynamic
# frequency analysis. This is applied after per-pixel temporal mean subtraction,
# so it removes coherent fluctuations rather than the static mean image.
DYNAMIC_SVD_FILTER_ENABLED = False

# Number of dominant SVD modes to remove when DYNAMIC_SVD_FILTER_ENABLED is True.
DYNAMIC_SVD_REMOVE_COMPONENTS = 5

# HSV frequency-dynamic rendering. H encodes mean temporal frequency, S encodes
# spectral bandwidth, and V encodes dynamic standard deviation.
DYNAMIC_HUE_FREQUENCY_RANGE_HZ = (2, 23.0)
DYNAMIC_SATURATION_BANDWIDTH_RANGE_HZ = (0.0, 8)
DYNAMIC_VALUE_DYNAMIC_RANGE = (0.0, 1000.0)
DYNAMIC_VALUE_GAMMA = 1.0

# Static/background normalization --------------------------------------------
# Shared small denominator protection for background X normalization and dynamic
# normalization. Usually leave this tiny; increase only if weak-signal pixels
# create unstable normalization artifacts.
NORMALIZATION_EPS = 1e-3

# Root order for background X normalization. 2.0 means square-root style scaling;
# changing this changes how strongly the X background profile is flattened.
BACKGROUND_X_NORMALIZATION_ROOT_ORDER = 2.0

# Diagnostics ----------------------------------------------------------------
# Diagnostic threshold, in percent, for logging unusual pre-FFT signal changes.
# This is for warning/inspection and should not change the processed data.
PRE_FFT_LOG_DEVIATION_THRESHOLD_PCT = 1.0

# Diagnostic threshold, in percent, for logging unusual dynamic-input changes.
# This is for warning/inspection and should not change the processed data.
DYNAMIC_INPUT_LOG_DEVIATION_THRESHOLD_PCT = 1.0


class GPUThread(QThread):
    def __init__(self):
        super().__init__()
        # Windowing remains configurable here if a dedicated preprocessing stage is added later.
        self.exit_message = 'GPU processing thread exited.'
        self.FFT_actions = 0 # count how many FFT actions have taken place
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
        self.gpu_spectral_baseline_window_size = GPU_SPECTRAL_BASELINE_WINDOW_SIZE
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
        self.holo_filter_info_printed = False
        self.gpu_profile_timing_enabled = GPU_PROFILE_TIMING_ENABLED
        self.gpu_profile_timing_print_chunks = GPU_PROFILE_TIMING_PRINT_CHUNKS
        self.background_x_normalization_eps = NORMALIZATION_EPS
        self.background_x_normalization_root_order = BACKGROUND_X_NORMALIZATION_ROOT_ORDER
        self.dynamic_normalization_eps = NORMALIZATION_EPS
        self.dynamic_temporal_lowpass_enabled = DYNAMIC_TEMPORAL_LOWPASS_ENABLED
        self.dynamic_uniform_filter_size = DYNAMIC_TEMPORAL_LOWPASS_WINDOW_SIZE
        self.dynamic_gaussian_smoothing = DYNAMIC_GAUSSIAN_SMOOTHING
        self.pre_fft_log_deviation_threshold_pct = PRE_FFT_LOG_DEVIATION_THRESHOLD_PCT
        self.dynamic_input_log_deviation_threshold_pct = DYNAMIC_INPUT_LOG_DEVIATION_THRESHOLD_PCT
        self.dynMagnification = DYNAMIC_MAGNIFICATION
        self.dynamic_svd_filter_enabled = DYNAMIC_SVD_FILTER_ENABLED
        self.dynamic_svd_remove_components = DYNAMIC_SVD_REMOVE_COMPONENTS
        self.dynamic_hue_frequency_range_hz = DYNAMIC_HUE_FREQUENCY_RANGE_HZ
        self.dynamic_saturation_bandwidth_range_hz = DYNAMIC_SATURATION_BANDWIDTH_RANGE_HZ
        self.dynamic_value_dynamic_range = DYNAMIC_VALUE_DYNAMIC_RANGE
        self.dynamic_value_gamma = DYNAMIC_VALUE_GAMMA
        self.yp_gpu_buffer = None
        self.yp_gpu_stream_buffers = [None, None]
        self.raw_gpu_buffer = None
        self.raw_gpu_stream_buffers = [None, None]
        self.float_gpu_buffer = None
        self.float_gpu_stream_buffers = [None, None]
        self.highpass_gpu_buffer = None
        self.highpass_gpu_stream_buffers = [None, None]
        self.gpu_streams = None
        # Reusable GPU buffer for pre-FFT averaging output.
        self.gpu_pre_avg_gpu_buffer = None
        self.active_tasks = 0
        self.return_processed_to_weaver = False

    def defwin(self):
        self.winfunc = cupy.ElementwiseKernel(
            'float32 x, complex64 y',
            'complex64 z',
            'z=x*y',
            'winfunc')

    def definterp(self):
        # define interpolation kernel
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
                        //if (i==0){
                        //        printf("sampx is %d, sampy0 is %llu\n",sampx, sampy0);
                         //       printf("indice1 is %hu, indice2 is %hu\n", indice1[sampx], indice2[sampx]);
                        //        printf("xt: %.5f, x0: %.5f, x1: %.5f, y0: %.5f, y1: %.5f\n",xt,x0,x1,y0,y1);
                        //        }
                        yp[i] = (y0+(xt-x0)*(y1-y0)/(x1-x0+0.00001));

                        }
            }
            ''','interp1d')
        self.spectral_baseline_subtraction_kernel = cupy.RawKernel(r'''
            extern "C" __global__
            void subtract_uniform_baseline(const float* src, float* dst, int lines, int samples, int radius, float inv_size){
                extern __shared__ float tile[];

                int line = blockIdx.y;
                int sample_start = blockIdx.x * blockDim.x;
                int tx = threadIdx.x;
                int sample = sample_start + tx;
                long long base = (long long)line * samples;

                if(line >= lines){
                    return;
                }

                int center_index = sample_start + tx;
                while(center_index < 0 || center_index >= samples){
                    if(center_index < 0){
                        center_index = -center_index - 1;
                    }
                    if(center_index >= samples){
                        center_index = 2 * samples - center_index - 1;
                    }
                }
                tile[tx + radius] = src[base + center_index];

                if(tx < radius){
                    int left_index = sample_start + tx - radius;
                    int right_index = sample_start + blockDim.x + tx;

                    while(left_index < 0 || left_index >= samples){
                        if(left_index < 0){
                            left_index = -left_index - 1;
                        }
                        if(left_index >= samples){
                            left_index = 2 * samples - left_index - 1;
                        }
                    }
                    while(right_index < 0 || right_index >= samples){
                        if(right_index < 0){
                            right_index = -right_index - 1;
                        }
                        if(right_index >= samples){
                            right_index = 2 * samples - right_index - 1;
                        }
                    }
                    tile[tx] = src[base + left_index];
                    tile[tx + blockDim.x + radius] = src[base + right_index];
                }

                __syncthreads();

                if(sample < samples){
                    float sum = 0.0f;
                    int tile_center = tx + radius;
                    for(int offset = -radius; offset <= radius; ++offset){
                        sum += tile[tile_center + offset];
                    }
                    dst[base + sample] = src[base + sample] - sum * inv_size;
                }
            }
            ''','subtract_uniform_baseline')
    def run(self):
        self.defwin()
        self.definterp()
        self.update_Dispersion()
        self.update_background()
        # self.update_FFTlength()
        self.QueueOut()

    def QueueOut(self):
        self.item = self.queue.get()
        while self.item.action != EXIT_ACTION:
            t1=time.time()
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
            except Exception as error:
                message = "FFT processing failed. This frame was skipped."
                print(message)
                self.emit_status(message)
                # self.ui.PrintOut.append(message)
                print(traceback.format_exc())
            
            finally:
                if getattr(self.item, "action", None) in (GPUActions.GPU, GPUActions.CPU):
                    self.release_memory_slot(getattr(self.item, "memory_slot", None))
                self.active_tasks = max(0, self.active_tasks - 1)
            if time.time()-t1>1:
                print('GPU thread took ', round(time.time()-t1,2), ' seconds for action: ', self.item.action)
            self.item = self.queue.get()
            # print('GPU queue size:', self.queue.qsize())
        self.emit_status(self.exit_message)

    def is_idle(self):
        return self.queue.qsize() == 0 and self.active_tasks == 0

    def emit_status(self, message):
        if message is None:
            return
        self.ui_bridge.status_message.emit(str(message))

    def current_dynamic_enabled(self):
        return False

    def current_bline_avg(self):
        return max(1, int(self.ui.BlineAVG.value()))

    def current_camera_frame_rate_hz(self):
        camera_name = self.ui.Camera.currentText()
        if camera_name == "Daheng" and hasattr(self.ui, "FrameRate_DH"):
            return float(self.ui.FrameRate_DH.value())
        if camera_name == "PhotonFocus" and hasattr(self.ui, "FrameRate_PF"):
            return float(self.ui.FrameRate_PF.value())
        return 1.0

    def current_dynamic_frame_rate_hz(self, pre_avg_count=1):
        pre_avg_count = max(1, int(pre_avg_count))
        return self.current_camera_frame_rate_hz() / float(pre_avg_count)

    def current_nsamples(self):
        return effective_camera_sample_count(self.ui)

    def current_depth_start(self):
        return int(self.ui.DepthStart.value())

    def current_depth_range(self):
        return int(self.ui.DepthRange.value())

    def current_save_enabled(self):
        return self.ui.Save.isChecked()

    def current_y_pixels(self):
        return max(1, int(self.ui.Ypixels.value()))

    def current_fft_result_mode(self):
        if getattr(self.item, "acq_mode", None) == AcqTypes.CONTINUOUS_KX:
            return "KX"
        return "AMP"

    def current_kx_display_enabled(self):
        return getattr(self.item, "acq_mode", None) == AcqTypes.CONTINUOUS_KX

    def current_fft_output_range(self, samples, pixel_range):
        if self.current_kx_display_enabled():
            return int(samples)
        return int(pixel_range)

    def select_fft_depth_result(self, fft_data, pixel_start, pixel_range, xp):
        depth_data = fft_data[:, pixel_start:pixel_start + pixel_range]
        return xp.absolute(depth_data)

    def should_run_realtime_dynamic(self):
        return False

    def current_dynamic_temporal_filter_size(self):
        if not bool(self.dynamic_temporal_lowpass_enabled):
            return 1
        return max(1, int(self.dynamic_uniform_filter_size))

    def current_spectral_baseline_window_size(self):
        window = min(513, max(1, int(self.gpu_spectral_baseline_window_size)))
        if window % 2 == 0:
            window = window + 1 if window < 513 else window - 1
        return window

    def dynamic_log_threshold_for_stage(self, stage_label):
        if stage_label == "pre_fft_pre_normalization":
            return float(self.pre_fft_log_deviation_threshold_pct)
        if stage_label in {"dynamic_processing_input", "offline_dynamic_processing_input"}:
            return float(self.dynamic_input_log_deviation_threshold_pct)
        return float(self.dynamic_input_log_deviation_threshold_pct)

    def dynamic_deviation_entries(self, frame_means, stage_label, frame_offset=0):
        values = np.asarray(frame_means, dtype=np.float32).reshape(-1)
        if values.size <= 1:
            return []
        reference = float(np.mean(values))
        if not np.isfinite(reference) or abs(reference) <= 1e-12:
            return []
        threshold_pct = self.dynamic_log_threshold_for_stage(stage_label)
        entries = []
        for local_index, mean_value in enumerate(values):
            deviation_pct = (float(mean_value) - reference) / reference * 100.0
            if abs(deviation_pct) >= threshold_pct:
                entries.append(
                    {
                        "stage": stage_label,
                        "frame_index": int(frame_offset + local_index),
                        "mean_intensity": float(mean_value),
                        "reference_mean": reference,
                        "deviation_pct": float(deviation_pct),
                    }
                )
        return entries

    def current_log_filename(self):
        if not self.current_save_enabled():
            return None
        bundle = getattr(self.item, "filename_bundle", None) or {}
        return (
            bundle.get("log_filename")
            or bundle.get("dynamic_filename")
            or bundle.get("filename")
        )

    def write_deviation_log_entries(self, frame_means, stage_label, filename, frame_offset=0, y_slice_index=None):
        if filename is None:
            return
        entries = self.dynamic_deviation_entries(frame_means, stage_label, frame_offset=frame_offset)
        for entry in entries:
            y_slice_prefix = (
                f"Y slice index={int(y_slice_index)}, "
                if y_slice_index is not None
                else ""
            )
            message = (
                f"{entry['stage']}: stack mean intensity={entry['reference_mean']:.3f}, "
                f"{y_slice_prefix}"
                f"outlier frame number={entry['frame_index']}, "
                f"outlier intensity={entry['mean_intensity']:.3f}, "
                f"percentage difference={entry['deviation_pct']:.2f}%, "
                f"file={filename}"
            )
            self.log.dynamic_write(message)

    def gpu_timing_start(self, stream=None):
        if not self.gpu_profile_timing_enabled:
            return None
        self.gpu_timing_synchronize(stream)
        return time.perf_counter()

    def gpu_timing_end(self, timing, label, start, stream=None):
        if start is None:
            return
        self.gpu_timing_synchronize(stream)
        timing[label] = timing.get(label, 0.0) + (time.perf_counter() - start)

    def gpu_timing_synchronize(self, stream=None):
        if stream is not None:
            stream.synchronize()
        else:
            cupy.cuda.Stream.null.synchronize()

    def format_gpu_timing_summary(self, timing):
        ordered_labels = [
            "prepare_request",
            "prepare_dynamic_gpu_stack",
            "load_raw_to_gpu",
            "convert_to_float",
            "pre_fft_average",
            "pre_fft_mean",
            "saved_background_subtraction",
            "spectral_highpass",
            "reshape_alines",
            "interpolation",
            "off_axis_x_filter",
            "fft",
            "select_depth_result",
            "reshape_depth",
            "post_fft_scaling",
            "background_x_normalization",
            "copy_chunk_to_dynamic_gpu",
            "copy_gpu_to_host",
            "chunk_wait_and_log",
            "dynamic_data_to_gpu",
            "dynamic_amplitude_center",
            "dynamic_amplitude_frequency_power",
            "dynamic_complex_center",
            "dynamic_complex_frequency_power",
            "dynamic_frequency_fold_power",
            "dynamic_frequency_hsv_metrics",
            "dynamic_frequency_hsv_rgb",
            "dynamic_result_to_cpu",
            "dynamic_gaussian_smoothing",
            "release_gpu_memory",
        ]
        parts = []
        for label in ordered_labels:
            if label in timing:
                parts.append(f"{label}={timing[label] * 1000.0:.2f} ms")
        for label, value in timing.items():
            if label not in ordered_labels and not label.startswith("_"):
                parts.append(f"{label}={value * 1000.0:.2f} ms")
        return ", ".join(parts)

    def print_gpu_timing_summary(self, timing, mode, frames, chunks):
        if not self.gpu_profile_timing_enabled:
            return
        total = sum(value for label, value in timing.items() if not label.startswith("_") and isinstance(value, (int, float)))
        wall_seconds = timing.get("_wall_seconds")
        message = (
            f"GPU timing [{mode}, frames={frames}, chunks={chunks}]: "
            f"total_profiled={total * 1000.0:.2f} ms"
        )
        if wall_seconds is not None:
            message += f", wall={wall_seconds * 1000.0:.2f} ms"
        details = self.format_gpu_timing_summary(timing)
        if details:
            message = f"{message}; {details}"
        print(message)
        self.emit_status(message)

    def release_memory_slot(self, memory_slot):
        if memory_slot is None or not hasattr(self, "MemoryFreeQueue"):
            return
        self.MemoryFreeQueue.put(memory_slot)

    def valid_input_frame_count(self, memory_slot):
        slot_frames = int(self.Memory[memory_slot].shape[0])
        requested = getattr(self.item, "package_frame_count", None)
        if requested is None:
            return slot_frames
        return max(1, min(slot_frames, int(requested)))

    def cudaFFT(self, DnS_action, acq_mode, memory_slot, context):
        timing = {"_wall_start": time.perf_counter()}
        package_profile = dict(getattr(self.item, "profile", {}) or {})
        package_profile["gpu_start_s"] = timing["_wall_start"]
        if package_profile.get("gpu_queue_put_s") is not None:
            package_profile["gpu_queue_wait_s"] = timing["_wall_start"] - package_profile["gpu_queue_put_s"]
        request_start = self.gpu_timing_start()
        # get samples per Aline
        samples = self.current_nsamples()
        # get depth pixels after FFT
        Pixel_start = self.current_depth_start()
        Pixel_range = self.current_depth_range()
        output_range = self.current_fft_output_range(samples, Pixel_range)
        slot_shape = self.Memory[memory_slot].shape
        valid_input_frames = self.valid_input_frame_count(memory_slot)
        shape = (valid_input_frames, slot_shape[1], slot_shape[2])
        pre_avg_count, effective_frames = self.pre_avg_plan(shape[0])

        # print('GPU data size: ', shape, ' memory_slot: ', memory_slot)
        # print('data shape', shape)
        # print('GPU receives:',self.data_CPU[0,0,0:10])
        chunk_frames = self.gpu_fft_chunk_frames(effective_frames)
        background_reference_gpu = self.determine_background_gpu(memory_slot)
        output_dtype = np.float32
        self.data_CPU = np.empty((effective_frames, shape[1], output_range), dtype=output_dtype)
        dynamic_gpu_stack = None
        if self.should_run_realtime_dynamic():
            stack_start = self.gpu_timing_start()
            dynamic_gpu_dtype = cupy.complex64 if output_dtype == np.complex64 else cupy.float32
            dynamic_gpu_stack = cupy.empty(self.data_CPU.shape, dtype=dynamic_gpu_dtype)
            self.gpu_timing_end(timing, "prepare_dynamic_gpu_stack", stack_start)
        log_filename = self.current_log_filename()
        y_slice_index = None
        self.gpu_timing_end(timing, "prepare_request", request_start)
        self.cudaFFT_chunked_overlapped(
            memory_slot,
            samples,
            Pixel_start,
            Pixel_range,
            chunk_frames,
            background_reference_gpu,
            pre_avg_count,
            log_filename,
            y_slice_index,
            timing,
            dynamic_gpu_stack,
        )
        del background_reference_gpu
        # print('data_CPU shape', self.data_CPU.shape)
        # print('data_CPU:', self.data_CPU[0,0,0:15])
        if self.should_run_realtime_dynamic():
            Dyn = self.compute_realtime_dynamic_gpu(
                dynamic_gpu_stack,
                timing,
                frame_rate_hz=self.current_dynamic_frame_rate_hz(pre_avg_count),
            )
        else:
            Dyn = []
        del dynamic_gpu_stack
        if self.release_gpu_memory_each_fft:
            release_start = self.gpu_timing_start()
            self.release_gpu_memory()
            self.gpu_timing_end(timing, "release_gpu_memory", release_start)
        if self.should_run_realtime_dynamic():
            self.write_deviation_log_entries(
                np.mean(np.abs(self.data_CPU), axis=(1, 2)),
                "dynamic_processing_input",
                log_filename,
                frame_offset=0,
                y_slice_index=y_slice_index,
            )
        timing["_wall_seconds"] = time.perf_counter() - timing["_wall_start"]
        package_profile["gpu_wall_s"] = timing["_wall_seconds"]
        for key, value in timing.items():
            if not key.startswith("_") and isinstance(value, (int, float)):
                package_profile[f"gpu_{key}_s"] = float(value)
        # display and save data, data type is float32
        an_action = DnSActionField(
            DnS_action,
            acq_mode=acq_mode,
            data=self.data_CPU,
            raw=False,
            dynamic=Dyn,
            context=context,
            gpu_avg_count=pre_avg_count,
            dynamic_bline_idx=self.item.dynamic_bline_idx,
            filename_bundle=self.item.filename_bundle,
            skip_save=self.item.skip_save,
            package_start=getattr(self.item, "package_start", 0),
            package_frame_count=getattr(self.item, "package_frame_count", None),
            package_total_frames=getattr(self.item, "package_total_frames", None),
            is_final=getattr(self.item, "is_final", True),
            profile=package_profile,
        )
        an_action.profile["dns_queue_put_s"] = time.perf_counter()
        self.DnSQueue.put(an_action)

        # print('send for display')
        if getattr(self, "return_processed_to_weaver", False):
            self.GPU2weaverQueue.put(self.data_CPU)
            # print('GPU data to weaver')
        self.print_gpu_timing_summary(
            timing,
            self.current_fft_result_mode(),
            effective_frames,
            int(timing.get("_chunks", 0)),
        )

    def fft_cpu(self, DnS_action, acq_mode, memory_slot, context):
        cpu_start_s = time.perf_counter()
        package_profile = dict(getattr(self.item, "profile", {}) or {})
        samples = self.current_nsamples()
        pixel_start = self.current_depth_start()
        pixel_range = self.current_depth_range()

        valid_input_frames = self.valid_input_frame_count(memory_slot)
        self.data_CPU = self.Memory[memory_slot][:valid_input_frames].astype(np.float32, copy=True)
        shape = self.data_CPU.shape
        pre_avg_count, _ = self.pre_avg_plan(shape[0])

        if pre_avg_count > 1:
            complete_frames = (self.data_CPU.shape[0] // pre_avg_count) * pre_avg_count
            if complete_frames >= pre_avg_count:
                self.data_CPU = self.data_CPU[:complete_frames]
                new_frame_count = complete_frames // pre_avg_count
                self.data_CPU = self.data_CPU.reshape(new_frame_count, pre_avg_count, shape[1], shape[2]).mean(axis=1)
            else:
                pre_avg_count = 1

        processed_shape = self.data_CPU.shape
        log_filename = self.current_log_filename()
        y_slice_index = None
        self.write_deviation_log_entries(
            np.mean(self.data_CPU, axis=(1, 2)),
            "pre_fft_pre_normalization",
            log_filename,
            frame_offset=0,
            y_slice_index=y_slice_index,
        )

        background_reference_cpu = self.determine_background_cpu(memory_slot)
        self.apply_saved_background_subtraction_cpu(self.data_CPU, background_reference_cpu)
        baseline = uniform_filter1d(
            self.data_CPU,
            size=self.current_spectral_baseline_window_size(),
            axis=2,
        )
        self.data_CPU -= baseline
        del baseline

        alines = processed_shape[0] * processed_shape[1]
        self.data_CPU = self.data_CPU.reshape([alines, samples])

        if self.interp:
            self.data_CPU = self.interpolate_cpu(self.data_CPU)

        if self.current_kx_display_enabled():
            self.data_CPU = self.data_CPU.reshape(processed_shape[0], processed_shape[1], samples)
            self.data_CPU = np.fft.fftshift(np.fft.fft(self.data_CPU, axis=1), axes=1)
            self.data_CPU = np.log1p(np.absolute(self.data_CPU)).astype(np.float32, copy=False)
        else:
            if self.interp:
                self.data_CPU = np.fft.fft(self.data_CPU * self.dispersion, axis=1) / samples
            else:
                self.data_CPU = np.fft.fft(self.data_CPU, axis=1) / samples
            self.data_CPU = self.select_fft_depth_result(self.data_CPU, pixel_start, pixel_range, np)
            self.data_CPU = np.float32(self.data_CPU)
            self.data_CPU = self.data_CPU.reshape(processed_shape[0], processed_shape[1], pixel_range)
        if self.current_dynamic_enabled():
            self.apply_post_fft_dynamic_normalization_cpu(self.data_CPU)
        else:
            self.data_CPU *= np.float32(self.AMPLIFICATION)
        if not self.current_kx_display_enabled():
            self.apply_background_x_normalization_cpu(self.data_CPU)

        if self.should_run_realtime_dynamic():
            dyn = self.compute_realtime_dynamic_cpu(
                frame_rate_hz=self.current_dynamic_frame_rate_hz(pre_avg_count),
            )
        else:
            dyn = []

        if self.should_run_realtime_dynamic():
            self.write_deviation_log_entries(
                np.mean(np.abs(self.data_CPU), axis=(1, 2)),
                "dynamic_processing_input",
                log_filename,
                frame_offset=0,
                y_slice_index=y_slice_index,
            )

        an_action = DnSActionField(
            DnS_action,
            acq_mode=acq_mode,
            data=self.data_CPU,
            raw=False,
            dynamic=dyn,
            context=context,
            gpu_avg_count=pre_avg_count,
            dynamic_bline_idx=self.item.dynamic_bline_idx,
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

    def gpu_fft_chunk_frames(self, total_frames):
        return min(total_frames, max(1, int(self.gpu_chunk_frames)))

    def pre_avg_plan(self, raw_frame_count):
        pre_avg_count = 1
        pre_avg_factor = self.pre_avg_factor()
        if pre_avg_factor > 1:
            complete_frames = (raw_frame_count // pre_avg_factor) * pre_avg_factor
            if complete_frames >= pre_avg_factor:
                return pre_avg_factor, complete_frames // pre_avg_factor
        return pre_avg_count, raw_frame_count

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
        return False

    def normalize_dynamic_frames(self, y_gpu):
        """
        Normalize each frame by its own mean before dynamic subtraction.

        Dynamic raw data are arranged as (frame, Y/X pixel, spectral pixel) in
        this processing path, so the per-frame light-source intensity estimate
        is the mean over the second and third dimensions. A small EPS is added
        to the denominator, matching compute_realtime_dynamic_gpu(), to avoid excessive
        gain when the reference intensity is very small.
        """
        frame_mean = cupy.mean(y_gpu, axis=(1, 2), keepdims=True)
        y_gpu /= frame_mean + cupy.float32(self.dynamic_normalization_eps)
        return y_gpu

    def normalize_dynamic_frames_cpu(self, data_cpu):
        frame_mean = np.mean(data_cpu, axis=(1, 2), keepdims=True)
        data_cpu /= frame_mean + np.float32(self.dynamic_normalization_eps)
        return data_cpu

    def apply_saved_background_subtraction_cpu(self, data_cpu, background_reference_cpu=None):
        if background_reference_cpu is not None:
            data_cpu -= background_reference_cpu
            return True
        return False

    def apply_post_fft_dynamic_normalization_cpu(self, data_cpu):
        frame_mean = np.mean(np.abs(data_cpu), axis=(1, 2), keepdims=True, dtype=np.float32)
        global_mean = np.mean(frame_mean, dtype=np.float32)
        data_cpu /= frame_mean + np.float32(self.dynamic_normalization_eps)
        data_cpu *= global_mean
        data_cpu *= np.float32(self.AMPLIFICATION)
        return data_cpu

    def apply_post_fft_dynamic_normalization_gpu(self, data_gpu):
        frame_mean = cupy.mean(cupy.absolute(data_gpu), axis=(1, 2), keepdims=True)
        global_mean = cupy.mean(frame_mean, dtype=cupy.float32)
        data_gpu /= frame_mean + cupy.float32(self.dynamic_normalization_eps)
        data_gpu *= global_mean
        data_gpu *= cupy.float32(self.AMPLIFICATION)
        return data_gpu

    def apply_background_x_normalization_cpu(self, data_cpu):
        chunk_shape = data_cpu.shape
        if self.background_x_normalization is None:
            return False
        if self.background_x_normalization.size != chunk_shape[1]:
            print(
                'Background X normalization mismatch. Skipped: ',
                self.background_x_normalization.size,
                'data X pixels:',
                chunk_shape[1],
            )
            return False
        data_cpu /= self.background_x_normalization[np.newaxis, :, np.newaxis]
        return True

    def interpolate_cpu(self, data_cpu):
        idx0 = self.indice[0, :].astype(np.intp, copy=False)
        idx1 = self.indice[1, :].astype(np.intp, copy=False)
        x0 = self.intpX[idx0]
        x1 = self.intpX[idx1]
        xt = self.intpXp
        y0 = data_cpu[:, idx0]
        y1 = data_cpu[:, idx1]
        return np.float32(y0 + (xt - x0) * (y1 - y0) / (x1 - x0 + 0.00001))

    def compute_realtime_dynamic_cpu(self, frame_rate_hz=None):
        if frame_rate_hz is None:
            frame_rate_hz = self.current_dynamic_frame_rate_hz()
        if isinstance(self.data_CPU, np.ndarray) and self.data_CPU.dtype.kind == 'c':
            data_cpu = np.asarray(self.data_CPU, dtype=np.complex64)
            if data_cpu.ndim != 3 or data_cpu.shape[0] < 2:
                return []
            metrics = self.compute_complex_frequency_hsv_cpu(data_cpu, frame_rate_hz)
        else:
            data_cpu = np.asarray(self.data_CPU, dtype=np.float32)
            if data_cpu.ndim != 3 or data_cpu.shape[0] < 2:
                return []
            metrics = self.compute_amplitude_frequency_hsv_cpu(data_cpu, frame_rate_hz)

        hsv = np.asarray(metrics["hsv"], dtype=np.float32)
        if self.dynamic_gaussian_smoothing:
            dynamic_std = hsv[..., 2]
            dynamic_std = gaussian_filter(dynamic_std, sigma=(1, 1))
            hsv[..., 2] = dynamic_std
        return {
            "hsv": hsv,
        }

    def frequency_axis_cpu(self, frames, frame_rate_hz):
        return np.fft.fftfreq(
            int(frames),
            d=1.0 / float(frame_rate_hz),
        ).astype(np.float32)

    def apply_dynamic_svd_filter_cpu(self, centered_cpu):
        if not self.dynamic_svd_filter_enabled:
            return centered_cpu

        frames = int(centered_cpu.shape[0])
        remove_components = int(self.dynamic_svd_remove_components)
        if remove_components <= 0:
            raise RuntimeError("Dynamic SVD filtering is enabled but DYNAMIC_SVD_REMOVE_COMPONENTS <= 0.")
        if remove_components >= frames:
            raise RuntimeError(
                f"Dynamic SVD remove components ({remove_components}) must be smaller than frame count ({frames})."
            )

        original_shape = centered_cpu.shape
        matrix_cpu = centered_cpu.reshape(frames, -1)
        pixel_count = max(1, int(matrix_cpu.shape[1]))
        covariance_cpu = (matrix_cpu @ matrix_cpu.conj().T) / np.float32(pixel_count)
        _, eigenvectors_cpu = np.linalg.eigh(covariance_cpu)
        dominant_modes_cpu = eigenvectors_cpu[:, -remove_components:]
        filtered_matrix_cpu = matrix_cpu - dominant_modes_cpu @ (dominant_modes_cpu.conj().T @ matrix_cpu)
        return filtered_matrix_cpu.reshape(original_shape)

    def fold_power_to_abs_frequency_cpu(self, power_cpu, frequencies_hz_cpu):
        abs_frequencies_hz_cpu = np.abs(frequencies_hz_cpu)
        unique_abs_frequencies_hz_cpu = np.unique(abs_frequencies_hz_cpu)
        folded_power_cpu = np.zeros(
            (unique_abs_frequencies_hz_cpu.size,) + power_cpu.shape[1:],
            dtype=np.float32,
        )
        for idx in range(int(unique_abs_frequencies_hz_cpu.size)):
            mask_cpu = np.isclose(abs_frequencies_hz_cpu, unique_abs_frequencies_hz_cpu[idx])
            folded_power_cpu[idx, :, :] = np.sum(power_cpu[mask_cpu, :, :], axis=0, dtype=np.float32)
        return unique_abs_frequencies_hz_cpu.astype(np.float32), folded_power_cpu

    def frequency_hsv_metrics_from_power_cpu(self, power_cpu, frequencies_hz_cpu):
        abs_frequency_hz_cpu, folded_power_cpu = self.fold_power_to_abs_frequency_cpu(
            power_cpu,
            frequencies_hz_cpu,
        )

        total_power_cpu = np.sum(folded_power_cpu, axis=0, dtype=np.float32)
        total_power_safe_cpu = np.maximum(total_power_cpu, np.float32(1e-12))
        frequency_axis_cpu = abs_frequency_hz_cpu[:, np.newaxis, np.newaxis]
        mean_frequency_hz_cpu = (
            np.sum(folded_power_cpu * frequency_axis_cpu, axis=0, dtype=np.float32)
            / total_power_safe_cpu
        ).astype(np.float32, copy=False)
        centered_frequency2_cpu = (frequency_axis_cpu - mean_frequency_hz_cpu[np.newaxis, :, :]) ** 2
        bandwidth_hz_cpu = np.sqrt(
            np.maximum(
                np.sum(folded_power_cpu * centered_frequency2_cpu, axis=0, dtype=np.float32)
                / total_power_safe_cpu,
                np.float32(0.0),
            )
        ).astype(np.float32, copy=False)
        dynamic_std_cpu = np.sqrt(np.maximum(total_power_cpu, np.float32(0.0))).astype(np.float32, copy=False)
        dynamic_std_cpu *= np.float32(self.dynMagnification)
        hsv_source_cpu = np.stack(
            [mean_frequency_hz_cpu, bandwidth_hz_cpu, dynamic_std_cpu],
            axis=-1,
        ).astype(np.float32, copy=False)

        return {
            "mean_frequency_hz": mean_frequency_hz_cpu,
            "bandwidth_hz": bandwidth_hz_cpu,
            "dynamic_std": dynamic_std_cpu,
            "hsv": hsv_source_cpu,
        }

    def compute_amplitude_frequency_hsv_cpu(self, data_cpu, frame_rate_hz):
        frames = int(data_cpu.shape[0])
        if frames < 2:
            raise RuntimeError("Amplitude HSV dynamic processing needs at least 2 frames.")
        centered_cpu = data_cpu.astype(np.float32, copy=False) - np.mean(data_cpu, axis=0, keepdims=True)
        centered_cpu = self.apply_dynamic_svd_filter_cpu(centered_cpu)
        spectrum_cpu = np.fft.fft(centered_cpu, axis=0) / np.float32(frames)
        power_cpu = (np.abs(spectrum_cpu) ** 2).astype(np.float32, copy=False)
        frequencies_hz_cpu = self.frequency_axis_cpu(frames, frame_rate_hz)
        return self.frequency_hsv_metrics_from_power_cpu(power_cpu, frequencies_hz_cpu)

    def compute_complex_frequency_hsv_cpu(self, data_cpu, frame_rate_hz):
        frames = int(data_cpu.shape[0])
        if frames < 2:
            raise RuntimeError("Complex HSV dynamic processing needs at least 2 frames.")
        centered_cpu = data_cpu.astype(np.complex64, copy=False) - np.mean(data_cpu, axis=0, keepdims=True)
        centered_cpu = self.apply_dynamic_svd_filter_cpu(centered_cpu)
        spectrum_cpu = np.fft.fft(centered_cpu, axis=0) / np.float32(frames)
        power_cpu = (np.abs(spectrum_cpu) ** 2).astype(np.float32, copy=False)
        frequencies_hz_cpu = self.frequency_axis_cpu(frames, frame_rate_hz)
        return self.frequency_hsv_metrics_from_power_cpu(power_cpu, frequencies_hz_cpu)


    def apply_background_x_normalization_gpu(self, y_gpu):
        chunk_shape = y_gpu.shape
        if self.background_x_normalization is None:
            return False
        if self.background_x_normalization.size != chunk_shape[1]:
            print(
                'Background X normalization mismatch. Skipped: ',
                self.background_x_normalization.size,
                'data X pixels:',
                chunk_shape[1],
            )
            return False
        if (
            self.background_x_normalization_gpu is None
            or self.background_x_normalization_gpu.shape != self.background_x_normalization.shape
        ):
            self.background_x_normalization_gpu = cupy.asarray(
                self.background_x_normalization,
                dtype=cupy.float32,
            )
        y_gpu /= self.background_x_normalization_gpu[cupy.newaxis, :, cupy.newaxis]
        return True

    def cudaFFT_chunked_overlapped(
        self,
        memory_slot,
        samples,
        Pixel_start,
        Pixel_range,
        chunk_frames,
        background_reference_gpu,
        pre_avg_count=1,
        log_filename=None,
        y_slice_index=None,
        timing=None,
        dynamic_gpu_stack=None,
    ):
        if timing is None:
            timing = {}
        total_output_frames = self.data_CPU.shape[0]
        streams = self.gpu_overlap_streams()
        slot_refs = [None, None]

        chunk_index = 0
        output_start = 0
        while output_start < total_output_frames:
            slot = chunk_index % 2
            if slot_refs[slot] is not None:
                wait_start = self.gpu_timing_start()
                slot_refs[slot]['stream'].synchronize()
                self.write_deviation_log_entries(
                    cupy.asnumpy(slot_refs[slot]['pre_fft_means_gpu']),
                    "pre_fft_pre_normalization",
                    log_filename,
                    frame_offset=slot_refs[slot]['frame_offset'],
                    y_slice_index=y_slice_index,
                )
                self.gpu_timing_end(timing, "chunk_wait_and_log", wait_start)
                slot_refs[slot] = None

            output_end = min(output_start + chunk_frames, total_output_frames)
            output_count = output_end - output_start
            input_start = output_start * pre_avg_count
            input_end = input_start + output_count * pre_avg_count
            chunk = self.Memory[memory_slot][input_start:input_end, :, :]
            host_out = self.data_CPU[output_start:output_end, :, :]

            slot_refs[slot] = self.cudaFFT_chunk_async(
                chunk,
                samples,
                Pixel_start,
                Pixel_range,
                background_reference_gpu,
                streams[slot],
                slot,
                host_out,
                pre_avg_count,
                output_start,
                log_filename,
                y_slice_index,
                timing,
                dynamic_gpu_stack,
            )
            timing["_chunks"] = timing.get("_chunks", 0) + 1

            output_start = output_end
            chunk_index += 1

        for slot in range(2):
            if slot_refs[slot] is not None:
                wait_start = self.gpu_timing_start()
                slot_refs[slot]['stream'].synchronize()
                self.write_deviation_log_entries(
                    cupy.asnumpy(slot_refs[slot]['pre_fft_means_gpu']),
                    "pre_fft_pre_normalization",
                    log_filename,
                    frame_offset=slot_refs[slot]['frame_offset'],
                    y_slice_index=y_slice_index,
                )
                self.gpu_timing_end(timing, "chunk_wait_and_log", wait_start)
                slot_refs[slot] = None

    def cudaFFT_chunk_async(
        self,
        raw_chunk,
        samples,
        Pixel_start,
        Pixel_range,
        background_reference_gpu,
        stream,
        slot,
        host_out,
        pre_avg_count=1,
        frame_offset=0,
        log_filename=None,
        y_slice_index=None,
        timing=None,
        dynamic_gpu_stack=None,
    ):
        if timing is None:
            timing = {}
        with stream:
            data_gpu, pre_fft_means_gpu, keep_alive = self.process_chunk_gpu(
                raw_chunk,
                samples,
                Pixel_start,
                Pixel_range,
                background_reference_gpu,
                pre_avg_count=pre_avg_count,
                slot=slot,
                stream=stream,
                timing=timing,
            )
            keep_alive.append(data_gpu)

            if dynamic_gpu_stack is not None:
                dynamic_copy_start = self.gpu_timing_start(stream)
                output_end = frame_offset + data_gpu.shape[0]
                dynamic_gpu_stack[frame_offset:output_end, :, :] = data_gpu
                self.gpu_timing_end(timing, "copy_chunk_to_dynamic_gpu", dynamic_copy_start, stream)

            copy_start = self.gpu_timing_start(stream)
            self.copy_gpu_to_host_async(data_gpu, host_out, stream)
            self.gpu_timing_end(timing, "copy_gpu_to_host", copy_start, stream)

        return {
            'stream': stream,
            'keep_alive': keep_alive,
            'pre_fft_means_gpu': pre_fft_means_gpu,
            'frame_offset': frame_offset,
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
        kx_axis = cupy.fft.fftshift(cupy.fft.fftfreq(x_pixels, d=pixel_size_m)).astype(cupy.float32)
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
        return weights

    def apply_holo_x_filter_gpu(self, spectral_gpu, frames, x_pixels, samples):
        field_gpu = spectral_gpu.reshape(int(frames), int(x_pixels), int(samples)).astype(cupy.complex64, copy=False)
        filtered_kx_gpu = cupy.fft.fftshift(cupy.fft.fft(field_gpu, axis=1), axes=1)
        weights_gpu = self.holo_filter_weights_gpu(x_pixels)
        filtered_kx_gpu *= weights_gpu[cupy.newaxis, :, cupy.newaxis]
        field_gpu = cupy.fft.ifft(cupy.fft.ifftshift(filtered_kx_gpu, axes=1), axis=1)

        if self.holo_reference_carrier_demodulation:
            pixel_size_m = self.current_camera_pixel_size_m()
            x_axis = (cupy.arange(int(x_pixels), dtype=cupy.float32) - cupy.float32((int(x_pixels) - 1) / 2.0))
            x_axis *= cupy.float32(pixel_size_m)
            k_axis = self.holo_wavenumber_axis_gpu(samples)
            sign = 1.0 if int(self.holo_sideband_sign) >= 0 else -1.0
            sin_tilt = np.sin(self.holo_reference_tilt_rad())
            carrier = cupy.exp(
                -1j
                * cupy.float32(sign * sin_tilt)
                * k_axis[cupy.newaxis, cupy.newaxis, :]
                * x_axis[cupy.newaxis, :, cupy.newaxis]
            )
            field_gpu *= carrier.astype(cupy.complex64, copy=False)
            if int(self.holo_sideband_sign) > 0 and self.holo_conjugate_positive_sideband:
                field_gpu = cupy.conj(field_gpu)

        if self.holo_subtract_common_mode_after_demod:
            field_gpu = field_gpu - cupy.median(field_gpu, axis=1, keepdims=True)

        return field_gpu.reshape(int(frames) * int(x_pixels), int(samples))

    def process_chunk_gpu(
        self,
        raw_chunk,
        samples,
        Pixel_start,
        Pixel_range,
        background_reference_gpu,
        pre_avg_count=1,
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

        if pre_avg_count > 1:
            step_start = self.gpu_timing_start(stream)
            y_gpu, output_frames = self.apply_pre_avg_filter(y_gpu, pre_avg_count, slot=slot)
            self.gpu_timing_end(timing, "pre_fft_average", step_start, stream)

        step_start = self.gpu_timing_start(stream)
        pre_fft_means_gpu = cupy.mean(y_gpu, axis=(1, 2))
        self.gpu_timing_end(timing, "pre_fft_mean", step_start, stream)
        keep_alive.append(pre_fft_means_gpu)

        step_start = self.gpu_timing_start(stream)
        self.apply_saved_background_subtraction_gpu(
            y_gpu,
            background_reference_gpu,
        )
        self.gpu_timing_end(timing, "saved_background_subtraction", step_start, stream)

        step_start = self.gpu_timing_start(stream)
        y_gpu = self.apply_highpass_filter(y_gpu, slot=slot)
        self.gpu_timing_end(timing, "spectral_highpass", step_start, stream)
        keep_alive.append(y_gpu)

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
            data_gpu = cupy.log1p(cupy.absolute(kx_gpu)).astype(cupy.float32, copy=False)
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
        output_range = self.current_fft_output_range(samples, Pixel_range)
        data_gpu = data_gpu.reshape(output_frames, chunk_shape[1], output_range)
        self.gpu_timing_end(timing, "reshape_depth", step_start, stream)

        step_start = self.gpu_timing_start(stream)
        if self.current_dynamic_enabled():
            self.apply_post_fft_dynamic_normalization_gpu(data_gpu)
        else:
            data_gpu *= cupy.float32(self.AMPLIFICATION)
        self.gpu_timing_end(timing, "post_fft_scaling", step_start, stream)

        if not self.current_kx_display_enabled():
            step_start = self.gpu_timing_start(stream)
            self.apply_background_x_normalization_gpu(data_gpu)
            self.gpu_timing_end(timing, "background_x_normalization", step_start, stream)

        if self.gpu_profile_timing_print_chunks and chunk_start is not None:
            self.gpu_timing_synchronize(stream)
            print(
                f"GPU chunk timing [{self.current_fft_result_mode()}, "
                f"frames={output_frames}, slot={slot}]: "
                f"{(time.perf_counter() - chunk_start) * 1000.0:.2f} ms"
            )

        return data_gpu, pre_fft_means_gpu, keep_alive

    def copy_gpu_to_host_async(self, data_gpu, host_out, stream):
        try:
            cupy.asnumpy(data_gpu, out=host_out, stream=stream, blocking=False)
        except TypeError:
            cupy.asnumpy(data_gpu, out=host_out, stream=stream)

    def gpu_overlap_streams(self):
        if self.gpu_streams is None:
            self.gpu_streams = [cupy.cuda.Stream(non_blocking=True), cupy.cuda.Stream(non_blocking=True)]
        return self.gpu_streams

    def gpu_raw_buffer(self, shape, dtype, slot=None):
        dtype = np.dtype(dtype)
        if slot is None:
            if (
                self.raw_gpu_buffer is None
                or self.raw_gpu_buffer.shape != shape
                or self.raw_gpu_buffer.dtype != dtype
            ):
                self.raw_gpu_buffer = cupy.empty(shape, dtype=dtype)
            return self.raw_gpu_buffer

        if (
            self.raw_gpu_stream_buffers[slot] is None
            or self.raw_gpu_stream_buffers[slot].shape != shape
            or self.raw_gpu_stream_buffers[slot].dtype != dtype
        ):
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

    def apply_highpass_filter(self, data_gpu, slot=None):
        out_gpu = self.gpu_highpass_buffer(data_gpu.shape, slot=slot)
        threads = 256
        lines = int(data_gpu.shape[0] * data_gpu.shape[1])
        samples = int(data_gpu.shape[2])
        window = self.current_spectral_baseline_window_size()
        radius = window // 2
        blocks_x = max(1, (samples + threads - 1) // threads)
        self.spectral_baseline_subtraction_kernel(
            (blocks_x, lines),
            (threads,),
            (
                data_gpu,
                out_gpu,
                lines,
                samples,
                int(radius),
                np.float32(1.0 / window),
            ),
            shared_mem=(threads + 2 * radius) * 4,
        )
        return out_gpu

    def gpu_highpass_buffer(self, shape, slot=None):
        if slot is None:
            if self.highpass_gpu_buffer is None or self.highpass_gpu_buffer.shape != shape:
                self.highpass_gpu_buffer = cupy.empty(shape, dtype=cupy.float32)
            return self.highpass_gpu_buffer

        if self.highpass_gpu_stream_buffers[slot] is None or self.highpass_gpu_stream_buffers[slot].shape != shape:
            self.highpass_gpu_stream_buffers[slot] = cupy.empty(shape, dtype=cupy.float32)
        return self.highpass_gpu_stream_buffers[slot]

    def gpu_interpolation_buffer(self, shape, slot=None):
        if slot is None:
            if self.yp_gpu_buffer is None or self.yp_gpu_buffer.shape != shape:
                self.yp_gpu_buffer = cupy.empty(shape, dtype=cupy.float32)
            return self.yp_gpu_buffer

        if self.yp_gpu_stream_buffers[slot] is None or self.yp_gpu_stream_buffers[slot].shape != shape:
            self.yp_gpu_stream_buffers[slot] = cupy.empty(shape, dtype=cupy.float32)
        return self.yp_gpu_stream_buffers[slot]

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
        if self.intpX_gpu is None or self.intpX_gpu.shape != self.intpX.shape:
            self.intpX_gpu = cupy.asarray(self.intpX, dtype=cupy.float32)
        if self.intpXp_gpu is None or self.intpXp_gpu.shape != self.intpXp.shape:
            self.intpXp_gpu = cupy.asarray(self.intpXp, dtype=cupy.float32)
        if self.indice1_gpu is None or self.indice1_gpu.shape != self.indice[0, :].shape:
            self.indice1_gpu = cupy.asarray(self.indice[0, :], dtype=cupy.uint16)
        if self.indice2_gpu is None or self.indice2_gpu.shape != self.indice[1, :].shape:
            self.indice2_gpu = cupy.asarray(self.indice[1, :], dtype=cupy.uint16)
        if self.dispersion_gpu is None or self.dispersion_gpu.shape != self.dispersion.shape:
            self.dispersion_gpu = cupy.asarray(self.dispersion, dtype=cupy.complex64)

    def clear_dispersion_gpu_cache(self):
        self.intpX_gpu = None
        self.intpXp_gpu = None
        self.indice1_gpu = None
        self.indice2_gpu = None
        self.dispersion_gpu = None

    def release_gpu_memory(self):
        cupy.get_default_memory_pool().free_all_blocks()
        pinned_pool = getattr(cupy, "get_default_pinned_memory_pool", None)
        if pinned_pool is not None:
            pinned_pool().free_all_blocks()

    def load_interpolation_indices(self, filename, samples):
        raw = np.fromfile(filename, dtype=np.uint16)
        expected = 2 * samples
        if raw.size != expected:
            raise ValueError(
                f"intpIndice.bin has {raw.size} values, expected {expected} "
                f"for {samples} samples."
            )

        max_valid = self.intpX.size - 1
        candidates = [
            ("2xsamples", raw.reshape([2, samples])),
            ("samplesx2", raw.reshape([samples, 2]).T),
        ]
        diagnostics = []
        for layout_name, candidate in candidates:
            min_index = int(candidate.min())
            max_index = int(candidate.max())
            diagnostics.append(f"{layout_name}: min={min_index}, max={max_index}")
            if min_index >= 0 and max_index <= max_valid:
                if layout_name != "2xsamples":
                    print(f"Interpolation index table loaded as {layout_name}.")
                return candidate

        raise ValueError(
            "Interpolation index table is out of range for intpX. "
            f"intpX length={self.intpX.size}, valid index=0-{max_valid}; "
            + "; ".join(diagnostics)
        )


    def update_Dispersion(self):
        # get samples per Aline
        samples = self.current_nsamples()
        # print('GPU dispersion samples: ',samples)

        # self.window = np.float32(np.hanning(samples))
        # update dispersion and window
        dispersion_path = self.ui.InD_DIR.text()
        # print(dispersion_path+'/dspPhase.bin')
        if os.path.isfile(dispersion_path+'/dspPhase.bin'):
            try:
                self.interp = True
                self.intpX  = np.float32(np.fromfile(dispersion_path+'/intpX.bin', dtype=np.float32))
                self.intpXp  = np.float32(np.fromfile(dispersion_path+'/intpXp.bin', dtype=np.float32))
                if self.intpX.size != samples or self.intpXp.size != samples:
                    raise ValueError(
                        f"Interpolation arrays do not match effective NSamples={samples}: "
                        f"len(intpX)={self.intpX.size}, len(intpXp)={self.intpXp.size}."
                    )
                self.indice = self.load_interpolation_indices(dispersion_path+'/intpIndice.bin', samples)
                dispersion_raw = np.fromfile(dispersion_path+'/dspPhase.bin', dtype=np.float32)
                if dispersion_raw.size != samples:
                    raise ValueError(
                        f"dspPhase.bin has {dispersion_raw.size} values, expected {samples}."
                    )
                self.dispersion = np.float32(dispersion_raw).reshape([1, samples])
                self.dispersion = np.complex64(np.exp(-1j*self.dispersion))
                self.clear_dispersion_gpu_cache()
                self.ensure_dispersion_gpu_cache()
                message = "Dispersion compensation loaded."
                self.emit_status(message)
                print(message)
            except ValueError as error:
                self.interp = False
                self.clear_dispersion_gpu_cache()
                message = f"Dispersion compensation file is invalid. Interpolation is disabled. {error}"
                self.emit_status(message)
                print(message)
            except OSError as error:
                self.interp = False
                self.clear_dispersion_gpu_cache()
                message = f"Dispersion compensation file could not be read. Interpolation is disabled. {error}"
                self.emit_status(message)
                print(message)
            except Exception as error:
                self.interp = False
                self.clear_dispersion_gpu_cache()
                message = f"Dispersion compensation load failed unexpectedly. Interpolation is disabled. {error}"
                self.emit_status(message)
                print(message)
        else:
            self.interp = False
            self.clear_dispersion_gpu_cache()
            # self.intpX  = np.float32(np.linspace(0,1,samples))
            # self.intpXp  = np.float32(np.linspace(0,1,samples))
            # self.indice = np.uint16(np.linspace(0,samples-1,samples)).reshape([samples,1])
            # self.indice = np.tile(self.indice,[1,2])
            # self.dispersion = np.complex64(np.ones(samples)).reshape([1,samples])
            message = "No dispersion compensation file found. Interpolation is disabled."
            self.emit_status(message)
            print(message)

    def update_background(self):
        # get samples per Aline
        samples = self.current_nsamples()
        # print('GPU dispersion samples: ',samples)
        Xpixels = self.ui.AlinesPerBline.value()
        # self.window = np.float32(np.hanning(samples))
        # update dispersion and window
        background_path = self.ui.BG_DIR.text()
        # print(dispersion_path+'/dspPhase.bin')
        if os.path.isfile(background_path):
            try:
                self.background  = np.float32(np.fromfile(background_path, dtype=np.float32)).reshape([Xpixels,samples])
                self.update_background_x_normalization()
                self.background_gpu = cupy.asarray(self.background, dtype=cupy.float32)
                if self.background_x_normalization is not None:
                    self.background_x_normalization_gpu = cupy.asarray(
                        self.background_x_normalization,
                        dtype=cupy.float32,
                    )
                # print(self.background.shape)

                # plt.figure()
                # plt.imshow(self.background)
                # plt.show()
                message = "Background file loaded."
                self.emit_status(message)
                print(message)
                self.bg_sub = True
            except ValueError as error:
                self.bg_sub = False
                self.background = np.zeros([Xpixels, samples])
                self.background_gpu = None
                self.background_x_normalization = None
                self.background_x_normalization_gpu = None
                message = f"Background file is invalid. Using zero background. {error}"
                self.emit_status(message)
                print(message)
            except OSError as error:
                self.bg_sub = False
                self.background = np.zeros([Xpixels, samples])
                self.background_gpu = None
                self.background_x_normalization = None
                self.background_x_normalization_gpu = None
                message = f"Background file could not be read. Using zero background. {error}"
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
        if (
            not np.isfinite(profile_mean)
            or profile_mean <= self.background_x_normalization_eps
        ):
            self.background_x_normalization = None
            self.background_x_normalization_gpu = None
            return False

        normalized = x_profile / np.float32(profile_mean)
        bad = (
            ~np.isfinite(normalized)
            | (np.abs(normalized) <= np.float32(self.background_x_normalization_eps))
        )
        if np.any(bad):
            normalized[bad] = np.float32(1.0)
        root_order = float(self.background_x_normalization_root_order)
        if np.isfinite(root_order) and root_order > 1.0:
            normalized = np.power(normalized, np.float32(1.0 / root_order), dtype=np.float32)
        self.background_x_normalization = np.asarray(normalized, dtype=np.float32)
        self.background_x_normalization_gpu = None
        return True


    def update_FFTlength(self):
        self.length_FFT = 2
        # get samples per Aline
        samples = self.current_nsamples()
        # print('GPU dispersion samples: ',samples)
        while self.length_FFT < samples:
            self.length_FFT *=2

    def display_FFT_actions(self):
        message = f"{self.FFT_actions} FFT request(s) processed."
        print(message)
        # self.ui.PrintOut.append(message)
        self.FFT_actions = 0

    def compute_realtime_dynamic_gpu(self, data_gpu=None, timing=None, frame_rate_hz=None):
        if timing is None:
            timing = {}
        if data_gpu is None:
            step_start = self.gpu_timing_start()
            if isinstance(self.data_CPU, np.ndarray) and self.data_CPU.dtype.kind == 'c':
                data_gpu = cupy.asarray(self.data_CPU, dtype=cupy.complex64)
            else:
                data_gpu = cupy.asarray(self.data_CPU, dtype=cupy.float32)
            self.gpu_timing_end(timing, "dynamic_data_to_gpu", step_start)

        if frame_rate_hz is None:
            frame_rate_hz = self.current_dynamic_frame_rate_hz()

        if data_gpu.dtype.kind == 'c':
            metrics_gpu = self.compute_complex_frequency_hsv_gpu(
                data_gpu,
                frame_rate_hz,
                timing=timing,
            )
        else:
            metrics_gpu = self.compute_amplitude_frequency_hsv_gpu(
                data_gpu,
                frame_rate_hz,
                timing=timing,
        )

        step_start = self.gpu_timing_start()
        hsv = cupy.asnumpy(metrics_gpu["hsv"])
        self.gpu_timing_end(timing, "dynamic_result_to_cpu", step_start)
        if self.dynamic_gaussian_smoothing:
            step_start = self.gpu_timing_start()
            dynamic_std = hsv[..., 2]
            dynamic_std = gaussian_filter(dynamic_std, sigma=(1, 1))
            hsv[..., 2] = dynamic_std
            self.gpu_timing_end(timing, "dynamic_gaussian_smoothing", step_start)
        return {
            "hsv": np.asarray(hsv, dtype=np.float32),
        }

    def prepare_stack_for_dynamic_processing(self, stack):
        stack_array = np.asarray(stack)
        if stack_array.dtype.kind == 'c':
            return np.asarray(stack_array, dtype=np.complex64)
        if (
            False
            and stack_array.ndim == 3
            and stack_array.shape[-1] % 2 == 0
        ):
            z_pixels = stack_array.shape[-1] // 2
            amplitude = np.asarray(stack_array[..., :z_pixels], dtype=np.float32)
            phase = np.asarray(stack_array[..., z_pixels:], dtype=np.float32)
            return (amplitude * np.exp(1j * phase)).astype(np.complex64, copy=False)
        return np.asarray(stack_array, dtype=np.float32)

    def compute_dynamic_and_mean_from_stack_gpu(self, stack):
        stack_prepared = self.prepare_stack_for_dynamic_processing(stack)
        mean_intensity = np.mean(np.abs(stack_prepared), axis=0, dtype=np.float32)
        if stack_prepared.dtype.kind == 'c':
            metrics_gpu = self.compute_complex_frequency_hsv_gpu(
                cupy.asarray(stack_prepared, dtype=cupy.complex64),
                self.current_dynamic_frame_rate_hz(),
            )
        else:
            metrics_gpu = self.compute_amplitude_frequency_hsv_gpu(
                cupy.asarray(stack_prepared, dtype=cupy.float32),
                self.current_dynamic_frame_rate_hz(),
            )
        dynamic = cupy.asnumpy(metrics_gpu["dynamic_std"])
        if self.dynamic_gaussian_smoothing:
            dynamic = gaussian_filter(dynamic, sigma=(1, 1))
        return np.asarray(dynamic, dtype=np.float32), np.asarray(mean_intensity, dtype=np.float32)

    def compute_dynamic_and_mean_from_stack(self, stack):
        return self.compute_dynamic_and_mean_from_stack_gpu(stack)

    def frequency_axis_gpu(self, frames, frame_rate_hz):
        return cupy.fft.fftfreq(
            int(frames),
            d=1.0 / float(frame_rate_hz),
        ).astype(cupy.float32)

    def apply_dynamic_svd_filter_gpu(self, centered_gpu, timing=None, label="dynamic"):
        if not self.dynamic_svd_filter_enabled:
            return centered_gpu
        if timing is None:
            timing = {}

        frames = int(centered_gpu.shape[0])
        remove_components = int(self.dynamic_svd_remove_components)
        if remove_components <= 0:
            raise RuntimeError("Dynamic SVD filtering is enabled but DYNAMIC_SVD_REMOVE_COMPONENTS <= 0.")
        if remove_components >= frames:
            raise RuntimeError(
                f"Dynamic SVD remove components ({remove_components}) must be smaller than frame count ({frames})."
            )

        step_start = self.gpu_timing_start()
        original_shape = centered_gpu.shape
        matrix_gpu = centered_gpu.reshape(frames, -1)
        pixel_count = max(1, int(matrix_gpu.shape[1]))
        covariance_gpu = (matrix_gpu @ matrix_gpu.conj().T) / cupy.float32(pixel_count)
        eigenvalues_gpu, eigenvectors_gpu = cupy.linalg.eigh(covariance_gpu)
        dominant_modes_gpu = eigenvectors_gpu[:, -remove_components:]
        filtered_matrix_gpu = matrix_gpu - dominant_modes_gpu @ (dominant_modes_gpu.conj().T @ matrix_gpu)
        filtered_gpu = filtered_matrix_gpu.reshape(original_shape)
        self.gpu_timing_end(timing, f"{label}_svd_filter_remove_{remove_components}", step_start)
        return filtered_gpu

    def fold_power_to_abs_frequency_gpu(self, power_gpu, frequencies_hz_gpu, timing=None):
        if timing is None:
            timing = {}
        step_start = self.gpu_timing_start()
        abs_frequencies_hz_gpu = cupy.absolute(frequencies_hz_gpu)
        unique_abs_frequencies_hz_gpu = cupy.unique(abs_frequencies_hz_gpu)
        folded_power_gpu = cupy.zeros(
            (unique_abs_frequencies_hz_gpu.size,) + power_gpu.shape[1:],
            dtype=cupy.float32,
        )
        for idx in range(int(unique_abs_frequencies_hz_gpu.size)):
            mask_gpu = cupy.isclose(abs_frequencies_hz_gpu, unique_abs_frequencies_hz_gpu[idx])
            folded_power_gpu[idx, :, :] = cupy.sum(power_gpu[mask_gpu, :, :], axis=0, dtype=cupy.float32)
        self.gpu_timing_end(timing, "dynamic_frequency_fold_power", step_start)
        return unique_abs_frequencies_hz_gpu.astype(cupy.float32), folded_power_gpu

    def normalize_dynamic_metric_gpu(self, image_gpu, value_range):
        low_value, high_value = float(value_range[0]), float(value_range[1])
        if high_value <= low_value:
            raise ValueError(f"Invalid dynamic HSV normalization range: {value_range}")
        normalized_gpu = (image_gpu.astype(cupy.float32, copy=False) - cupy.float32(low_value)) / cupy.float32(high_value - low_value)
        return cupy.clip(normalized_gpu, cupy.float32(0.0), cupy.float32(1.0))

    def hsv_to_rgb_gpu(self, hue_gpu, saturation_gpu, value_gpu):
        hue_gpu = cupy.mod(hue_gpu, cupy.float32(1.0))
        saturation_gpu = cupy.clip(saturation_gpu, cupy.float32(0.0), cupy.float32(1.0))
        value_gpu = cupy.clip(value_gpu, cupy.float32(0.0), cupy.float32(1.0))

        h6_gpu = hue_gpu * cupy.float32(6.0)
        i_gpu = cupy.floor(h6_gpu).astype(cupy.int32)
        f_gpu = h6_gpu - i_gpu.astype(cupy.float32)
        p_gpu = value_gpu * (cupy.float32(1.0) - saturation_gpu)
        q_gpu = value_gpu * (cupy.float32(1.0) - saturation_gpu * f_gpu)
        t_gpu = value_gpu * (cupy.float32(1.0) - saturation_gpu * (cupy.float32(1.0) - f_gpu))
        i_mod_gpu = cupy.mod(i_gpu, 6)

        r_gpu = cupy.empty_like(value_gpu)
        g_gpu = cupy.empty_like(value_gpu)
        b_gpu = cupy.empty_like(value_gpu)

        mask_gpu = i_mod_gpu == 0
        r_gpu[mask_gpu] = value_gpu[mask_gpu]
        g_gpu[mask_gpu] = t_gpu[mask_gpu]
        b_gpu[mask_gpu] = p_gpu[mask_gpu]

        mask_gpu = i_mod_gpu == 1
        r_gpu[mask_gpu] = q_gpu[mask_gpu]
        g_gpu[mask_gpu] = value_gpu[mask_gpu]
        b_gpu[mask_gpu] = p_gpu[mask_gpu]

        mask_gpu = i_mod_gpu == 2
        r_gpu[mask_gpu] = p_gpu[mask_gpu]
        g_gpu[mask_gpu] = value_gpu[mask_gpu]
        b_gpu[mask_gpu] = t_gpu[mask_gpu]

        mask_gpu = i_mod_gpu == 3
        r_gpu[mask_gpu] = p_gpu[mask_gpu]
        g_gpu[mask_gpu] = q_gpu[mask_gpu]
        b_gpu[mask_gpu] = value_gpu[mask_gpu]

        mask_gpu = i_mod_gpu == 4
        r_gpu[mask_gpu] = t_gpu[mask_gpu]
        g_gpu[mask_gpu] = p_gpu[mask_gpu]
        b_gpu[mask_gpu] = value_gpu[mask_gpu]

        mask_gpu = i_mod_gpu == 5
        r_gpu[mask_gpu] = value_gpu[mask_gpu]
        g_gpu[mask_gpu] = p_gpu[mask_gpu]
        b_gpu[mask_gpu] = q_gpu[mask_gpu]

        rgb_gpu = cupy.stack([r_gpu, g_gpu, b_gpu], axis=-1)
        return cupy.clip(cupy.rint(rgb_gpu * cupy.float32(255.0)), 0, 255).astype(cupy.uint8)

    def frequency_hsv_metrics_from_power_gpu(self, power_gpu, frequencies_hz_gpu, timing=None):
        if timing is None:
            timing = {}
        abs_frequency_hz_gpu, folded_power_gpu = self.fold_power_to_abs_frequency_gpu(
            power_gpu,
            frequencies_hz_gpu,
            timing=timing,
        )

        step_start = self.gpu_timing_start()
        total_power_gpu = cupy.sum(folded_power_gpu, axis=0, dtype=cupy.float32)
        total_power_safe_gpu = cupy.maximum(total_power_gpu, cupy.float32(1e-12))
        frequency_axis_gpu = abs_frequency_hz_gpu[:, cupy.newaxis, cupy.newaxis]
        mean_frequency_hz_gpu = (
            cupy.sum(folded_power_gpu * frequency_axis_gpu, axis=0, dtype=cupy.float32)
            / total_power_safe_gpu
        ).astype(cupy.float32, copy=False)
        centered_frequency2_gpu = (frequency_axis_gpu - mean_frequency_hz_gpu[cupy.newaxis, :, :]) ** 2
        bandwidth_hz_gpu = cupy.sqrt(
            cupy.maximum(
                cupy.sum(folded_power_gpu * centered_frequency2_gpu, axis=0, dtype=cupy.float32)
                / total_power_safe_gpu,
                cupy.float32(0.0),
            )
        ).astype(cupy.float32, copy=False)
        dynamic_std_gpu = cupy.sqrt(cupy.maximum(total_power_gpu, cupy.float32(0.0))).astype(cupy.float32, copy=False)
        dynamic_std_gpu *= cupy.float32(self.dynMagnification)
        self.gpu_timing_end(timing, "dynamic_frequency_hsv_metrics", step_start)

        step_start = self.gpu_timing_start()
        hue_gpu = self.normalize_dynamic_metric_gpu(
            mean_frequency_hz_gpu,
            self.dynamic_hue_frequency_range_hz,
        )
        saturation_gpu = self.normalize_dynamic_metric_gpu(
            bandwidth_hz_gpu,
            self.dynamic_saturation_bandwidth_range_hz,
        )
        value_gpu = self.normalize_dynamic_metric_gpu(
            dynamic_std_gpu,
            self.dynamic_value_dynamic_range,
        )
        gamma = float(self.dynamic_value_gamma)
        if np.isfinite(gamma) and gamma > 0.0 and abs(gamma - 1.0) > 1e-6:
            value_gpu = cupy.power(value_gpu, cupy.float32(1.0 / gamma))
        hsv_source_gpu = cupy.stack(
            [mean_frequency_hz_gpu, bandwidth_hz_gpu, dynamic_std_gpu],
            axis=-1,
        )
        self.gpu_timing_end(timing, "dynamic_frequency_hsv_source", step_start)

        return {
            "mean_frequency_hz": mean_frequency_hz_gpu,
            "bandwidth_hz": bandwidth_hz_gpu,
            "dynamic_std": dynamic_std_gpu,
            "hsv": hsv_source_gpu,
        }

    def compute_amplitude_frequency_hsv_gpu(self, data_gpu, frame_rate_hz, timing=None):
        if timing is None:
            timing = {}
        frames = int(data_gpu.shape[0])
        if frames < 2:
            raise RuntimeError("Amplitude HSV dynamic processing needs at least 2 frames.")
        step_start = self.gpu_timing_start()
        centered_gpu = data_gpu.astype(cupy.float32, copy=False) - cupy.mean(data_gpu, axis=0, keepdims=True)
        self.gpu_timing_end(timing, "dynamic_amplitude_center", step_start)
        centered_gpu = self.apply_dynamic_svd_filter_gpu(centered_gpu, timing=timing, label="dynamic_amplitude")

        step_start = self.gpu_timing_start()
        spectrum_gpu = cupy.fft.fft(centered_gpu, axis=0) / cupy.float32(frames)
        power_gpu = (cupy.absolute(spectrum_gpu) ** 2).astype(cupy.float32, copy=False)
        self.gpu_timing_end(timing, "dynamic_amplitude_frequency_power", step_start)

        frequencies_hz_gpu = self.frequency_axis_gpu(frames, frame_rate_hz)
        return self.frequency_hsv_metrics_from_power_gpu(power_gpu, frequencies_hz_gpu, timing=timing)

    def compute_complex_frequency_hsv_gpu(self, data_gpu, frame_rate_hz, timing=None):
        if timing is None:
            timing = {}
        frames = int(data_gpu.shape[0])
        if frames < 2:
            raise RuntimeError("Complex HSV dynamic processing needs at least 2 frames.")
        step_start = self.gpu_timing_start()
        centered_gpu = data_gpu.astype(cupy.complex64, copy=False) - cupy.mean(data_gpu, axis=0, keepdims=True)
        self.gpu_timing_end(timing, "dynamic_complex_center", step_start)
        centered_gpu = self.apply_dynamic_svd_filter_gpu(centered_gpu, timing=timing, label="dynamic_complex")

        step_start = self.gpu_timing_start()
        spectrum_gpu = cupy.fft.fft(centered_gpu, axis=0) / cupy.float32(frames)
        power_gpu = (cupy.absolute(spectrum_gpu) ** 2).astype(cupy.float32, copy=False)
        self.gpu_timing_end(timing, "dynamic_complex_frequency_power", step_start)

        frequencies_hz_gpu = self.frequency_axis_gpu(frames, frame_rate_hz)
        return self.frequency_hsv_metrics_from_power_gpu(power_gpu, frequencies_hz_gpu, timing=timing)

    def pre_avg_factor(self):
        return self.current_bline_avg()

    def gpu_pre_avg_buffer(self, out_shape, slot=None):
        """Get or create buffer for pre-FFT averaging output."""
        if slot is None:
            if self.gpu_pre_avg_gpu_buffer is None or self.gpu_pre_avg_gpu_buffer.shape != out_shape:
                self.gpu_pre_avg_gpu_buffer = cupy.empty(out_shape, dtype=cupy.float32)
            return self.gpu_pre_avg_gpu_buffer
        return None  # Stream buffers not needed for simple reshape+mean

    def apply_pre_avg_filter(self, y_gpu, avg_factor, slot=None):
        """
        Apply pre-FFT averaging on GPU to reduce frame count.
        This reduces GPU load for slower cards when processing many frames.

        Returns: tuple (averaged_gpu, effective_frame_count)
        """
        if avg_factor <= 1:
            return y_gpu, y_gpu.shape[0]

        chunk_shape = y_gpu.shape

        # Calculate number of complete groups
        total_frames = chunk_shape[0]
        complete_frames = (total_frames // avg_factor) * avg_factor

        if complete_frames < avg_factor:
            # Not enough frames to average
            return y_gpu, total_frames

        # Trim to complete groups
        y_trimmed = y_gpu[:complete_frames]

        # Reshape: (N*factor, X, Z) -> (N, factor, X, Z) -> mean over axis 1
        new_frame_count = complete_frames // avg_factor
        out_shape = (new_frame_count, chunk_shape[1], chunk_shape[2])

        # Use buffer or create new array
        out_gpu = self.gpu_pre_avg_buffer(out_shape, slot=slot)
        if out_gpu is None:
            out_gpu = cupy.empty(out_shape, dtype=cupy.float32)

        # Reshape and mean on GPU
        y_reshaped = y_trimmed.reshape(new_frame_count, avg_factor, chunk_shape[1], chunk_shape[2])
        cupy.mean(y_reshaped, axis=1, out=out_gpu)

        return out_gpu, new_frame_count

