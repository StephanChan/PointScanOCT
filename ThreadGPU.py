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
from CameraUi import effective_camera_sample_count, x_pixel_count

# GPU performance -------------------------------------------------------------
GPU_RELEASE_MEMORY_EACH_FFT = False
GPU_FFT_CHUNK_FRAMES = 32
GPU_OVERLAP_TRANSFER = True

# Notch (band-stop) filters ----------------------------------------------------
# X notch: sharp, DC-centred band-stop along the X GALVO axis - the fast/horizontal
# axis inside a B-line image (X = galvo scan direction). This is NOT the
# spectral/depth axis. It is applied per spectral channel while the data is still
# in the spectral domain: after interpolation and before the depth FFT. The
# suppressed half-width is a fraction of the X Nyquist frequency and is adjustable
# from the UI (XNotchFilter / XNotchBandwidth). Disabled by default.
NOTCH_X_FILTER_DEFAULT = False
NOTCH_X_CENTER_BAND_HALF_WIDTH_FRACTION = 0.25
NOTCH_X_PRINT_FILTER_INFO = True

# Xfrequency diagnostic display -------------------------------------------------
# What the Xfrequency acquisition mode renders after its X-direction transform:
#   "ifft"     - the B-line picture reconstructed after blanking the notch band in
#                the X-frequency domain (normal B-scan look, so the notch effect is
#                visible directly). With the notch off it is a plain FFT/IFFT round
#                trip, i.e. the untouched (BlineAVG-averaged) B-line image.
#   "spectrum" - the raw log-scale X-frequency spectrum (fx on screen).
X_FREQUENCY_DISPLAY = "ifft"

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
        # X notch state (see NOTCH_X_* above).
        self.notch_x_enabled = NOTCH_X_FILTER_DEFAULT
        self.notch_x_half_width_fraction = NOTCH_X_CENTER_BAND_HALF_WIDTH_FRACTION
        self._notch_x_weights_cache = {}
        self._notch_x_weights_gpu_cache = {}
        self._notch_x_info_printed = False
        self.gpu_overlap_transfer = GPU_OVERLAP_TRANSFER
        self.kx_display_scale = CONTINUOUS_KX_DISPLAY_SCALE
        self.kx_normalize_per_frame = CONTINUOUS_KX_NORMALIZE_PER_FRAME
        # Mode of the package currently being processed (set per item).
        self.current_acq_mode = None
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
        return False


    @staticmethod
    def fft_depth_limit(samples):
        return max(1, int(samples))


    def depth_crop_bounds(self, total_samples, pixel_start, pixel_range):
        limit = self.fft_depth_limit(total_samples)
        start = min(max(0, int(pixel_start)), limit - 1)
        end = min(start + max(1, int(pixel_range)), limit)
        end = max(start + 1, end)
        return start, end


    def current_fft_output_range(self, samples, pixel_start, pixel_range):
        if self.current_kx_display_enabled():
            return int(samples)
        start, end = self.depth_crop_bounds(samples, pixel_start, pixel_range)
        return end - start




    def kx_display_target_max(self):
        target = getattr(self, "kx_display_scale", "linear")
        if target == "linear":
            return 65025
        return 255

    # -- Xfrequency diagnostic display ------------------------------------- #

    def current_x_frequency_mode(self):
        return getattr(self, "current_acq_mode", None) == AcqTypes.X_FREQUENCY

    def x_frequency_display(self, data_cpu):
        """Xfrequency: transform the B-line along X (the galvo axis) and show the result.

        The processed B-line image (A-lines x depth) is first averaged down to one row
        per X position, then transformed along X. The buffer layout is preserved -
        (frames, 1, depth) - so the normal B-line display path renders it.

        X_FREQUENCY_DISPLAY selects what is shown:
          "ifft"     - the picture reconstructed after blanking the X notch band in the
                       fx domain (default). Notch off -> plain FFT/IFFT round trip, i.e.
                       the untouched B-line; XNotchBandwidth = 0 -> only the fx = 0
                       column (the DC/streak component) is removed.
          "spectrum" - the log-scale fx spectrum itself (fx axis on screen, 0 = centre),
                       with the notch band blanked black.
        Content that is constant along X (horizontal stripes) lives at fx = 0.
        """
        arr = np.asarray(data_cpu)
        if arr.ndim != 3 or arr.shape[1] != 1:
            self.emit_status("Xfrequency: unexpected data shape %s" % (arr.shape,))
            return data_cpu
        x_pixels = max(1, int(x_pixel_count(self.ui)))
        series = arr[:, 0, :].astype(np.float32, copy=False)
        rows = (int(series.shape[0]) // x_pixels) * x_pixels
        if rows < x_pixels:
            self.emit_status(
                "Xfrequency: %d A-line(s) is fewer than %d X pixel(s); reduce X pixels "
                "or BlineAVG." % (int(series.shape[0]), x_pixels)
            )
            return data_cpu
        depth = int(series.shape[1])
        image = series[:rows].reshape(rows // x_pixels, x_pixels, depth).mean(axis=0)
        spectrum = np.fft.fftshift(np.fft.fft(image, axis=0), axes=0)
        magnitude = np.abs(spectrum)
        raw_peak = float(magnitude.max()) if magnitude.size else 0.0
        log_peak = float(np.log10(raw_peak + np.float32(1.0))) if raw_peak > 0.0 else 0.0
        notch_band = self.x_frequency_notch_band(x_pixels)
        if notch_band is not None:
            # Blank exactly the fx band the X notch suppresses in the acquisition
            # pipeline (XNotchBandwidth = 0 -> the single fx = 0 column only).
            spectrum[notch_band[0]:notch_band[1] + 1, :] = 0.0
            magnitude[notch_band[0]:notch_band[1] + 1, :] = 0.0
        self.announce_x_frequency_notch_band(x_pixels, notch_band)
        if X_FREQUENCY_DISPLAY == "spectrum":
            # fx spectrum, normalised by the unfiltered peak so switching the notch
            # on/off only blacks out that band and leaves every other pixel unchanged.
            frame = np.log10(magnitude + np.float32(1.0))
            if log_peak > 0.0:
                frame *= np.float32(255.0 / log_peak)
        else:
            # Inverse transform back onto the X (galvo) axis: the B-line picture left
            # behind by the same notch band. Linear magnitude in the regular B-scan
            # units, so the XZmin/XZmax contrast applies as usual (with the notch off
            # this is a plain FFT/IFFT round trip of the B-line image).
            frame = np.abs(np.fft.ifft(np.fft.ifftshift(spectrum, axes=0), axis=0))
            frame = frame.astype(np.float32, copy=False)
        result = frame.reshape(frame.shape[0], 1, frame.shape[1])
        depth_out = min(int(result.shape[2]), int(arr.shape[2]))
        out = np.zeros_like(arr)
        # Process_bline averages each X-sized block back into one B-line, so the
        # transformed frame is tiled over every complete block: without this the
        # average would come out divided by the repeat (BlineAVG) count.
        blocks = int(out.shape[0]) // max(1, int(result.shape[0]))
        for block in range(max(1, blocks)):
            start = block * int(result.shape[0])
            stop = start + int(result.shape[0])
            if stop > int(out.shape[0]):
                break
            out[start:stop, 0, :depth_out] = result[:, 0, :depth_out]
        return out.astype(arr.dtype, copy=False)

    def x_frequency_notch_band(self, x_pixels):
        """Display-order bin range (inclusive) that the X notch suppresses.

        Uses the very same notch_x_weights() mask as the acquisition pipeline, so the
        blank band drawn on the Xfrequency picture matches the suppressed band 1:1.
        Returns None when the notch is off or removes nothing.
        """
        x_pixels = int(x_pixels)
        if x_pixels < 2 or not self.current_x_notch_enabled():
            return None
        weights = np.asarray(self.notch_x_weights(x_pixels), dtype=np.float32).reshape(-1)
        if int(weights.size) != x_pixels:
            return None
        removed = np.nonzero(np.fft.fftshift(weights) <= np.float32(0.5))[0]
        if removed.size == 0:
            return None
        return int(removed[0]), int(removed[-1])

    def announce_x_frequency_notch_band(self, x_pixels, band):
        """Report the notch band once per parameter change (keeps the status bar sane)."""
        if band is None:
            return
        x_pixels = max(1, int(x_pixels))
        low, high = int(band[0]), int(band[1])
        fraction = float(self.current_x_notch_half_width_fraction())
        key = (x_pixels, low, high, fraction)
        if key == getattr(self, "_x_frequency_band_key", None):
            return
        self._x_frequency_band_key = key
        half = x_pixels / 2.0
        message = (
            "Xfrequency: X notch blanks fx %.3f..%.3f x Nyquist (display bins %d..%d of %d, "
            "%.1f%% of the X band), XNotchBandwidth=%.3f."
            % (
                (low - half) / half,
                (high - half) / half,
                low,
                high,
                x_pixels,
                100.0 * (high - low + 1) / x_pixels,
                fraction,
            )
        )
        print(message)
        self.emit_status(message)

    # -- X notch (band-stop) filter ---------------------------------------- #

    def current_x_notch_enabled(self):
        widget = getattr(self.ui, "XNotchFilter", None)
        if widget is not None and hasattr(widget, "isChecked"):
            try:
                return bool(widget.isChecked())
            except Exception:
                pass
        return bool(self.notch_x_enabled)

    def current_x_notch_half_width_fraction(self):
        widget = getattr(self.ui, "XNotchBandwidth", None)
        if widget is not None and hasattr(widget, "value"):
            try:
                value = float(widget.value())
                if np.isfinite(value):
                    return max(0.0, min(value, 1.0))
            except Exception:
                pass
        return float(self.notch_x_half_width_fraction)

    def notch_x_weights(self, x_pixels):
        """Sharp DC-centred band-stop along the X galvo axis: 0 inside |fx| <= cutoff,
        1 outside.

        The transform axis is the X galvo direction inside a B-line (x_pixels =
        A-lines per B-line), not the spectral/depth axis. np.fft.fftfreq is
        cycles/sample in [-0.5, 0.5), so Nyquist is 0.5 and the cutoff is expressed
        directly as a fraction of it (no pixel size needed).
        Native FFT order: no fftshift/ifftshift around the filter.
        """
        x_pixels = int(x_pixels)
        fraction = float(self.current_x_notch_half_width_fraction())
        cache_key = (x_pixels, fraction)
        cached = self._notch_x_weights_cache.get(cache_key)
        if cached is not None:
            return cached
        freq_axis = np.fft.fftfreq(x_pixels).astype(np.float32)
        cutoff = np.float32(fraction * 0.5)
        weights = (np.abs(freq_axis) > cutoff).astype(np.float32)
        self._notch_x_weights_cache = {cache_key: weights}
        if NOTCH_X_PRINT_FILTER_INFO and not self._notch_x_info_printed:
            self._notch_x_info_printed = True
            print(
                "X notch (band-stop along the X galvo axis): enabled=%s, "
                "half-width=%.3f x X Nyquist, x_pixels=%d"
                % (self.current_x_notch_enabled(), fraction, x_pixels)
            )
        return weights

    def notch_x_weights_gpu(self, x_pixels):
        x_pixels = int(x_pixels)
        fraction = float(self.current_x_notch_half_width_fraction())
        cached = self._notch_x_weights_gpu_cache.get(x_pixels)
        if cached is not None and cached[0] == fraction:
            return cached[1]
        weights_gpu = cupy.asarray(self.notch_x_weights(x_pixels))
        self._notch_x_weights_gpu_cache = {x_pixels: (fraction, weights_gpu)}
        return weights_gpu

    def notch_x_pixels(self, alines, frames=None):
        """Transform length of the X notch FFT: the UI X pixel count.

        `alines` is the flattened A-line count along the acquisition axis. X (the galvo
        axis) is laid out innermost, so that count must be a whole multiple of the X
        pixel count for every group of X A-lines to be one valid B-line.

        `frames` is accepted for call-site compatibility and ignored: deriving the
        length by dividing the A-line count by a frame count returned 1 (and the
        filter was then silently skipped) whenever the A-line count itself was passed
        in as the frame count.
        """
        alines = int(alines)
        x_pixels = max(1, int(x_pixel_count(self.ui)))
        if alines <= 0 or x_pixels < 2:
            return None
        if alines % x_pixels:
            self.warn_x_notch_unaligned(alines, x_pixels)
            return None
        return x_pixels

    def notify_x_notch_applied(self, alines, x_pixels):
        """Log the first X notch application for a given geometry (never silent)."""
        x_pixels = max(1, int(x_pixels))
        groups = max(1, int(alines) // x_pixels)
        fraction = float(self.current_x_notch_half_width_fraction())
        key = (int(alines), x_pixels, fraction)
        if key == getattr(self, "_x_notch_applied_key", None):
            return
        self._x_notch_applied_key = key
        message = (
            "X notch: filtering %d B-line group(s) of %d A-lines along X "
            "(half width %.3f x X Nyquist)." % (groups, x_pixels, fraction)
        )
        print(message)
        self.emit_status(message)

    def warn_x_notch_unaligned(self, alines, x_pixels):
        """Report an applied X notch that cannot be formed (never silent)."""
        x_pixels = max(1, int(x_pixels))
        key = (int(alines) % x_pixels, x_pixels)
        if key == getattr(self, "_x_notch_unaligned_key", None):
            return
        self._x_notch_unaligned_key = key
        message = (
            "X notch skipped: %d A-line(s) is not a multiple of %d X pixel(s)."
            % (int(alines), x_pixels)
        )
        print(message)
        self.emit_status(message)

    def apply_x_notch_gpu(self, k_domain_gpu, frames, x_pixels, samples):
        """Filter one chunk along X, in the spectral domain (native FFT order)."""
        self.notify_x_notch_applied(int(frames) * int(x_pixels), x_pixels)
        field_gpu = k_domain_gpu.reshape(int(frames), int(x_pixels), int(samples))
        weights_gpu = self.notch_x_weights_gpu(x_pixels)
        spectrum_gpu = cupy.fft.fft(field_gpu, axis=1)
        spectrum_gpu *= weights_gpu[cupy.newaxis, :, cupy.newaxis]
        filtered_gpu = cupy.fft.ifft(spectrum_gpu, axis=1)
        return filtered_gpu.reshape(int(frames) * int(x_pixels), int(samples))

    def apply_x_notch_cpu(self, k_domain_cpu, frames, x_pixels, samples):
        """CPU counterpart of apply_x_notch_gpu (used by the CPU FFT path)."""
        self.notify_x_notch_applied(int(frames) * int(x_pixels), x_pixels)
        field_cpu = k_domain_cpu.reshape(int(frames), int(x_pixels), int(samples))
        weights = self.notch_x_weights(x_pixels)
        spectrum_cpu = np.fft.fft(field_cpu, axis=1)
        spectrum_cpu *= weights[np.newaxis, :, np.newaxis]
        filtered_cpu = np.fft.ifft(spectrum_cpu, axis=1)
        return filtered_cpu.reshape(int(frames) * int(x_pixels), int(samples))

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

    def select_fft_depth_result(self, fft_data, pixel_start, pixel_range, xp):
        samples = int(fft_data.shape[-1])
        start, end = self.depth_crop_bounds(samples, pixel_start, pixel_range)
        return fft_data[..., start:end]

    def current_gpu_avg_count(self, frame_count=None):
        return 1

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
        self.current_acq_mode = acq_mode
        timing = {"_wall_start": time.perf_counter()}
        package_profile = dict(getattr(self.item, "profile", {}) or {})
        package_profile["gpu_start_s"] = timing["_wall_start"]
        if package_profile.get("gpu_queue_put_s") is not None:
            package_profile["gpu_queue_wait_s"] = timing["_wall_start"] - package_profile["gpu_queue_put_s"]
        request_start = self.gpu_timing_start()
        samples = self.current_nsamples()
        Pixel_start = self.current_depth_start()
        Pixel_range = self.current_depth_range()
        output_range = self.current_fft_output_range(samples, Pixel_start, Pixel_range)
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
        slot_x_count = int(slot_shape[1]) if len(slot_shape) >= 3 else 1
        self.data_CPU, output_buffer_token, allocation_seconds = self.acquire_pinned_output_buffer(
            (output_frames, slot_x_count, output_range), np.float32
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
        if self.current_x_frequency_mode():
            self.data_CPU = self.x_frequency_display(self.data_CPU)
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
        self.current_acq_mode = acq_mode
        cpu_start_s = time.perf_counter()
        package_profile = dict(getattr(self.item, "profile", {}) or {})
        samples = self.current_nsamples()
        pixel_start = self.current_depth_start()
        pixel_range = self.current_depth_range()
        output_range = self.current_fft_output_range(samples, pixel_start, pixel_range)
        valid_input_frames = self.valid_input_frame_count(memory_slot)
        gpu_avg_count = self.current_gpu_avg_count(valid_input_frames)
        effective_frames = self.averaged_frame_count(valid_input_frames, gpu_avg_count) * gpu_avg_count
        self.data_CPU = self.Memory[memory_slot][:effective_frames].astype(np.float32, copy=True)
        if self.data_CPU.ndim == 2:
            self.data_CPU = self.data_CPU[:, np.newaxis, :]
        processed_shape = self.data_CPU.shape
        background_reference_cpu = self.determine_background_cpu(memory_slot)
        self.apply_saved_background_subtraction_cpu(self.data_CPU, background_reference_cpu)
        alines = processed_shape[0] * processed_shape[1]
        self.data_CPU = self.data_CPU.reshape([alines, samples])
        if self.interp:
            self.data_CPU = self.interpolate_cpu(self.data_CPU)
        if (
            self.current_x_notch_enabled()
            and not self.current_kx_display_enabled()
            and not self.current_x_frequency_mode()
        ):
            alines = int(self.data_CPU.shape[0])
            x_pixels = self.notch_x_pixels(alines)
            if x_pixels:
                self.data_CPU = self.apply_x_notch_cpu(
                    self.data_CPU, alines // x_pixels, x_pixels, samples
                )
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
            self.data_CPU = self.data_CPU.reshape(processed_shape[0], processed_shape[1], output_range)
        if not self.current_kx_display_enabled():
            self.data_CPU *= np.float32(self.AMPLIFICATION)
            self.apply_background_x_normalization_cpu(self.data_CPU)
            if gpu_avg_count > 1:
                self.data_CPU = self.data_CPU.reshape(
                    effective_frames // gpu_avg_count,
                    gpu_avg_count,
                    processed_shape[1],
                    output_range,
                ).mean(axis=1, dtype=np.float32)
        if self.current_x_frequency_mode():
            self.data_CPU = self.x_frequency_display(self.data_CPU)
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
        if (
            self.current_x_notch_enabled()
            and not self.current_kx_display_enabled()
            and not self.current_x_frequency_mode()
        ):
            # The X notch transforms every group of X pixels inside a chunk, so a chunk
            # has to carry whole B-lines: round the chunk up to a multiple of the X
            # pixel count (and of the averaging count, which is applied per chunk too).
            x_pixels = max(1, int(x_pixel_count(self.ui)))
            if x_pixels >= 2:
                step = x_pixels * avg_count
                requested = max(requested, ((requested + step - 1) // step) * step)
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
            chunk = self.Memory[memory_slot][input_start:input_end]
            if chunk.ndim == 2:
                chunk = chunk[:, np.newaxis, :]
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

        if (
            self.current_x_notch_enabled()
            and not self.current_kx_display_enabled()
            and not self.current_x_frequency_mode()
        ):
            alines = int(yp_gpu.shape[0])
            x_pixels = self.notch_x_pixels(alines)
            if x_pixels:
                step_start = self.gpu_timing_start(stream)
                yp_gpu = self.apply_x_notch_gpu(yp_gpu, alines // x_pixels, x_pixels, samples)
                self.gpu_timing_end(timing, "x_notch", step_start, stream)
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
        output_range = self.current_fft_output_range(samples, Pixel_start, Pixel_range)
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
        expected_shape = shape[1:] if len(shape) >= 3 else (shape[-1],)
        background = self.background
        if self.bg_sub and background.shape == expected_shape:
            pass
        elif self.bg_sub and len(shape) == 2 and background.shape == (1, shape[-1]):
            pass
        else:
            return None
        if self.bg_sub:
            if self.background_gpu is None or self.background_gpu.shape != self.background.shape:
                self.background_gpu = cupy.asarray(self.background, dtype=cupy.float32)
            return self.background_gpu
        return None






    def determine_background_cpu(self, memory_slot):
        shape = self.Memory[memory_slot].shape
        expected_shape = shape[1:] if len(shape) >= 3 else (shape[-1],)
        if self.bg_sub and self.background.shape == expected_shape:
            return np.asarray(self.background, dtype=np.float32)
        if self.bg_sub and len(shape) == 2 and self.background.shape == (1, shape[-1]):
            return np.asarray(self.background, dtype=np.float32)
        return None

    def ensure_dispersion_gpu_cache(self):
        if self.dispersion_gpu is None:
            self.update_Dispersion()
        intpX_gpu = getattr(self, "intpX_gpu", None)
        if intpX_gpu is None:
            self.update_Dispersion()

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
        samples = self.current_nsamples()
        interp_path = os.path.normpath(str(self.ui.InD_DIR.text()).strip())
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

        if not interp_path or not os.path.isdir(interp_path):
            message = f"Interpolation/dispersion folder not found: {interp_path or '<empty>'}"
            print(message)
            self.emit_status(message)
            self.interp = False
            return

        calibration_files = (
            ("intpX", "intpX.bin", np.float32),
            ("intpXp", "intpXp.bin", np.float32),
            ("indice", "intpIndice.bin", np.uint16),
            ("dispersion", "dspPhase.bin", np.float32),
        )
        missing = []
        loaded_formats = {}
        file_sizes = {}
        try:
            for attr, filename, dtype in calibration_files:
                filepath = os.path.join(interp_path, filename)
                if not os.path.isfile(filepath):
                    missing.append(filename)
                    continue
                file_sizes[attr] = os.path.getsize(filepath)
                if attr == "indice":
                    value = self.load_interpolation_indices(filepath, samples)
                    loaded_format = "uint16 index table"
                elif attr == "dispersion":
                    phase = np.fromfile(filepath, dtype=np.float32)
                    value = np.exp(-1j * phase.reshape([1, phase.size])).astype(np.complex64)
                    loaded_format = "float32 phase"
                else:
                    value = np.fromfile(filepath, dtype=dtype)
                    loaded_format = np.dtype(dtype).name
                setattr(self, attr, value)
                loaded_formats[attr] = loaded_format
        except Exception as error:
            message = f"Interpolation/dispersion calibration could not be loaded from {interp_path}: {error}"
            print(message)
            self.emit_status(message)
            self.interp = False
            return

        if missing:
            message = (
                "Interpolation/dispersion calibration is incomplete. Missing file(s): "
                f"{', '.join(missing)} in {interp_path}"
            )
            print(message)
            self.emit_status(message)
            self.interp = False
            return

        self.indice1 = self.indice[0, :]
        self.indice2 = self.indice[1, :]
        arrays = (self.intpX, self.intpXp, self.indice1, self.indice2, self.dispersion)
        self.interp = (
            self.intpX is not None
            and self.intpX.size == samples
            and self.intpXp is not None
            and self.intpXp.size == samples
            and self.indice is not None
            and self.indice.shape == (2, samples)
            and self.dispersion is not None
            and self.dispersion.size == samples
        )
        if not self.interp:
            sizes = ", ".join(
                (
                    f"{name}={getattr(self, name).shape if getattr(self, name) is not None else 'missing'}"
                    f" ({loaded_formats.get(name, 'not loaded')}, {file_sizes.get(name, 0)} bytes)"
                )
                for name, _, _ in calibration_files
            )
            message = (
                "Interpolation/dispersion calibration is incomplete or does not match "
                f"the current spectral sample count ({samples}). Sizes: {sizes}. "
                "Processing without calibration."
            )
            print(message)
            self.emit_status(message)
            return

        self.intpX_gpu = cupy.asarray(self.intpX, dtype=cupy.float32)
        self.intpXp_gpu = cupy.asarray(self.intpXp, dtype=cupy.float32)
        self.indice1_gpu = cupy.asarray(self.indice1, dtype=cupy.uint16)
        self.indice2_gpu = cupy.asarray(self.indice2, dtype=cupy.uint16)
        self.dispersion_gpu = cupy.asarray(self.dispersion, dtype=cupy.complex64)
        message = f"Interpolation/dispersion calibration loaded from {interp_path}."
        print(message)
        self.emit_status(message)

    def update_background(self):
        samples = self.current_nsamples()
        Xpixels = x_pixel_count(self.ui)
        background_shape = (1, int(samples))
        background_path = self.ui.BG_DIR.text()
        if os.path.isfile(background_path):
            try:
                raw_background = np.fromfile(background_path, dtype=np.float32)
                if raw_background.size == int(samples):
                    background = raw_background.reshape(background_shape)
                    message_suffix = "line-scan spectral background"
                elif raw_background.size == int(Xpixels) * int(samples):
                    background = raw_background.reshape([int(Xpixels), int(samples)]).mean(
                        axis=0,
                        keepdims=True,
                        dtype=np.float32,
                    )
                    message_suffix = (
                        "legacy X-by-spectral background averaged to line-scan background"
                    )
                else:
                    raise ValueError(
                        f"values={raw_background.size}, expected={int(samples)}"
                    )
                self.background = np.asarray(background, dtype=np.float32)
                self.update_background_x_normalization()
                self.background_gpu = cupy.asarray(self.background, dtype=cupy.float32)
                if self.background_x_normalization is not None:
                    self.background_x_normalization_gpu = cupy.asarray(
                        self.background_x_normalization,
                        dtype=cupy.float32,
                    )
                message = f"Background file loaded as {message_suffix}."
                self.emit_status(message)
                print(message)
                self.bg_sub = True
            except (OSError, ValueError) as error:
                self.bg_sub = False
                self.background = np.zeros(background_shape, dtype=np.float32)
                self.background_gpu = None
                self.background_x_normalization = None
                self.background_x_normalization_gpu = None
                message = f"Background mismatch; using zero background ({error})."
                print(message)
            except Exception as error:
                self.bg_sub = False
                self.background = np.zeros(background_shape, dtype=np.float32)
                self.background_gpu = None
                self.background_x_normalization = None
                self.background_x_normalization_gpu = None
                message = f"Background load failed unexpectedly. Using zero background. {error}"
                self.emit_status(message)
                print(message)
        else:
            self.bg_sub = False
            self.background = np.zeros(background_shape, dtype=np.float32)
            self.background_gpu = None
            self.background_x_normalization = None
            self.background_x_normalization_gpu = None
            print("No background selected; using zero background.")

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













