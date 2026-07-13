# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 11:10:17 2024

@author: admin
"""

#################################################################
from PyQt5.QtCore import  QThread
import time
import numpy as np
from Generaic_functions import *
from ActionFields import DnSActionField, AODOActionField, GPUActionField, DActionField
from ActionTypes import AcqTypes, DnSActions, EXIT_ACTION, GPUActions, WeaverActions
import traceback
import os
import matplotlib.pyplot as plt
from queue import Empty
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
# from matplotlib.path import Path
# from scipy.signal import hilbert
import datetime
import cv2
from UsbCamera import open_usb_camera, orient_usb_frame
from CameraUi import effective_camera_sample_count, camera_pixel_format
from HardwareSpecs import get_objective_spec

BLINE_MODES = (
    AcqTypes.FINITE_BLINE,
    AcqTypes.CONTINUOUS_BLINE,
    AcqTypes.CONTINUOUS_KX,
)

ALINE_MODES = ()
CSCAN_MODES = (
    AcqTypes.FINITE_CSCAN,
    AcqTypes.CONTINUOUS_CSCAN,
)

CONTINUOUS_MODES = (
    AcqTypes.CONTINUOUS_BLINE,
    AcqTypes.CONTINUOUS_KX,
    AcqTypes.CONTINUOUS_CSCAN,
)

AUTO_BACKGROUND_PER_FOV_ENABLED = False
class WeaverThread(QThread):
    def __init__(self):
        super().__init__()
        self.exit_message = 'Acquisition thread exited.'
        
    def run(self):
        if getattr(self, "file_naming", None) is None:
            raise RuntimeError("WeaverThread.file_naming must be assigned before starting the thread.")
        self.QueueOut()

    def emit_status(self, message):
        if message is None:
            return
        self.ui_bridge.status_message.emit(str(message))

    def finish_with_message(self, message):
        if message is None:
            return
        print(str(message))
        self.emit_status(message)
        
    def QueueOut(self):
        self.item = self.queue.get()
        while self.item.action != EXIT_ACTION:
            try:
                if self.item.action in (
                    AcqTypes.CONTINUOUS_BLINE,
                    AcqTypes.CONTINUOUS_KX,
                    AcqTypes.CONTINUOUS_CSCAN,
                ):
                    if not self.wait_for_processing_barrier(label=f"starting {self.item.action}"):
                        message = f"{self.item.action} stopped by user."
                        self.finish_with_message(message)
                        raise StopIteration
                    self.InitMemory()
                    message = self.RptScan(DnS_action=self.item.action, acq_mode=self.item.action)
                    self.finish_with_message(message)
                    
                elif self.item.action in (
                    AcqTypes.FINITE_BLINE,
                    AcqTypes.FINITE_CSCAN,
                ):
                    if not self.wait_for_processing_barrier(label=f"starting {self.item.action}"):
                        message = f"{self.item.action} stopped by user."
                        self.finish_with_message(message)
                        raise StopIteration
                    self.InitMemory()
                    message = self.SingleScan(DnS_action=self.item.action, acq_mode=self.item.action)
                    self.finish_with_message(message)
                elif self.item.action == AcqTypes.LOCATION_CAMERA_LIVE:
                    self.live()

                elif self.item.action == WeaverActions.ZSTAGE_REPEATIBILITY:
                    message = self.ZstageRepeatibility()
                    self.finish_with_message(message)

                elif self.item.action == WeaverActions.GOTO_ZERO:
                    message = self.Gotozero()
                    self.finish_with_message(message)

                elif self.item.action == WeaverActions.GET_BACKGROUND:
                    message = self.get_background()
                    self.finish_with_message(message)

                elif self.item.action == WeaverActions.GET_SURFACE:
                    message = self.get_surfCurve()
                    self.finish_with_message(message)
                    
                else:
                    message = f"Unknown acquisition command: {self.item.action}"
                    self.finish_with_message(message)

            except StopIteration:
                pass
            except Exception as error:
                message = f"Acquisition command failed: {self.item.action}"
                self.finish_with_message(message)
                print(traceback.format_exc())
            # reset RUN button
            self.wait_for_processing_barrier(
                label=f"finishing {self.item.action}",
                stop_if_run_unchecked=False,
            )
            self.ui.RunButton.setChecked(False)
            self.ui.RunButton.setText('Go')
            self.ui.PauseButton.setChecked(False)
            self.ui.PauseButton.setText('Pause')
            self.ui.RunButton.setEnabled(True)
            self.ui.PauseButton.setEnabled(True)
            self.GPUQueue.put(GPUActionField(GPUActions.CLEAR))
            if self.ui_bridge is not None:
                self.ui_bridge.acquisition_controls_locked.emit(False)
            # wait for next command
            self.item = self.queue.get()
        # exit weaver thread
        self.emit_status(self.exit_message)
            
    def drain_queue(self, queue, name, keep=None):
        """Remove queued items, optionally keeping items selected by keep(item)."""
        drained = 0
        kept = []
        while True:
            try:
                item = queue.get_nowait()
            except Empty:
                break
            if keep is not None and keep(item):
                kept.append(item)
            else:
                drained += 1
        for item in kept:
            queue.put(item)
        if drained:
            message = f"Cleared {drained} stale item(s) from {name}."
            print(message)
        return drained

    def drain_continuous_backlog(self, reason=""):
        def keep_gpu_item(item):
            is_fft_action = getattr(item, 'action', None) in {GPUActions.GPU, GPUActions.CPU}
            is_continuous_mode = getattr(item, 'DnS_action', None) in CONTINUOUS_MODES
            return not (is_fft_action and is_continuous_mode)

        drained_gpu = self.drain_queue(self.GPUQueue, "GPUQueue", keep=keep_gpu_item)
        drained_camera = self.drain_queue(self.DatabackQueue, "DatabackQueue")
        if drained_gpu or drained_camera:
            suffix = f" ({reason})" if reason else ""
            print(
                "Continuous backlog drain complete"
                f"{suffix}: GPUQueue={drained_gpu}, DatabackQueue={drained_camera}"
            )

    def processing_backlog(self):
        gpu_pending = self.GPUQueue.qsize()
        gpu_active = getattr(getattr(self, "gpu_thread", None), "active_tasks", 0)
        dns_pending = self.DnSQueue.qsize()
        dns_active = getattr(getattr(self, "dns_thread", None), "active_tasks", 0)
        return gpu_pending, gpu_active, dns_pending, dns_active

    def current_acq_mode(self):
        return self.ui.ACQMode.currentText()

    def current_fft_device(self):
        return self.ui.FFTDevice.currentText()

    def current_save_enabled(self):
        return self.ui.Save.isChecked()

    def current_dynamic_enabled(self):
        return False

    def current_geometry(self):
        return None

    def current_bline_avg(self):
        return max(1, int(self.ui.BlineAVG.value()))

    def current_y_pixels(self):
        return max(1, int(self.ui.Ypixels.value()))

    def current_max_y_fov_mm(self):
        objective = get_objective_spec(self.ui.Objective.currentText())
        if objective is None:
            raise RuntimeError(f"Unknown objective: {self.ui.Objective.currentText()}")
        return float(objective.max_y_fov_mm)

    def current_alines_per_bline(self):
        return max(1, int(self.ui.AlinesPerBline.value()))

    def current_nsamples(self):
        return effective_camera_sample_count(self.ui)

    def current_depth_range(self):
        return int(self.ui.DepthRange.value())

    def current_realtime_dynamic_enabled(self):
        return False

    def current_pre_avg_factor(self):
        if self.current_fft_device() not in ['GPU', 'CPU']:
            return 1
        return self.current_bline_avg()

    def current_processed_repeat_count(self, raw_frame_count):
        pre_avg_factor = self.current_pre_avg_factor()
        if pre_avg_factor <= 1:
            return int(raw_frame_count)
        complete_frames = (int(raw_frame_count) // pre_avg_factor) * pre_avg_factor
        if complete_frames < pre_avg_factor:
            return int(raw_frame_count)
        return max(1, complete_frames // pre_avg_factor)

    def finalize_partial_dynamic_naming(self, acq_mode):
        if (
            not self.current_save_enabled()
            or not self.current_dynamic_enabled()
            or self.current_realtime_dynamic_enabled()
        ):
            return
        if self.file_naming.dynamic_bline_idx == 1:
            return
        if acq_mode in CSCAN_MODES:
            self.file_naming.increment_cscan()
            self.file_naming.reset_dynamic_bline_idx()

    def build_filename_bundle(
        self,
        DnS_action,
        acq_mode,
        memory_slot,
        raw=False,
        package_start=0,
        package_frame_count=None,
        package_total_frames=None,
        is_final=True,
    ):
        if not self.current_save_enabled():
            return {}
        if self.file_naming is None:
            raise RuntimeError("WeaverThread.file_naming is required when saving is enabled.")

        raw_shape = self.Memory[memory_slot].shape
        raw_frames = int(raw_shape[0] if package_frame_count is None else package_frame_count)
        x_pixels = int(raw_shape[1])
        z_pixels = self.current_nsamples() if raw else self.current_depth_range()
        repeat_count = raw_frames if raw else self.current_processed_repeat_count(raw_frames)
        y_pixels = self.current_y_pixels()
        dynamic_bline_idx = int(self.file_naming.dynamic_bline_idx)
        bundle = {}

        if acq_mode in BLINE_MODES:
            filename = self.file_naming.get_filename("bline", acq_mode, [repeat_count, x_pixels, z_pixels])
            bundle = {"filename": filename, "log_filename": filename}
            if self.current_realtime_dynamic_enabled():
                bundle["dynamic_filename"] = self.file_naming.get_filename(
                    "bline_dyn",
                    acq_mode,
                    [repeat_count, x_pixels, z_pixels],
                )
                bundle["dynamic_rgb_filename"] = self.file_naming.get_filename(
                    "bline_dyn_rgb",
                    acq_mode,
                    [repeat_count, x_pixels, z_pixels],
                )
            self.file_naming.increment_bline()
            return bundle

        if acq_mode in CSCAN_MODES:
            if self.current_dynamic_enabled():
                if self.current_realtime_dynamic_enabled():
                    dynamic_filename = self.file_naming.get_filename(
                        "cscan_dyn",
                        acq_mode,
                        [y_pixels, x_pixels, z_pixels],
                        ypixels=y_pixels,
                    )
                    dynamic_rgb_filename = self.file_naming.get_filename(
                        "cscan_dyn_rgb",
                        acq_mode,
                        [y_pixels, x_pixels, z_pixels],
                    )
                    mean_filename = self.file_naming.get_filename(
                        "cscan_mean",
                        acq_mode,
                        [y_pixels, x_pixels, z_pixels],
                    )
                    bundle = {
                        "dynamic_filename": dynamic_filename,
                        "dynamic_rgb_filename": dynamic_rgb_filename,
                        "mean_filename": mean_filename,
                        "log_filename": dynamic_filename,
                    }
                    if dynamic_bline_idx == y_pixels:
                        self.file_naming.increment_cscan()
                        self.file_naming.reset_dynamic_bline_idx()
                    else:
                        self.file_naming.increment_dynY()
                    return bundle

                bline_filename = self.file_naming.get_filename(
                    "cscan_bline",
                    acq_mode,
                    [repeat_count, x_pixels, z_pixels],
                    ypixels=y_pixels,
                )
                bundle = {
                    "filename": bline_filename,
                    "log_filename": bline_filename,
                }
                self.file_naming.advance_cscan_dynamic_bline(y_pixels)
                return bundle

            if package_total_frames is not None:
                total_y_pixels = max(1, int(package_total_frames) // max(1, self.current_bline_avg()))
                active = getattr(self, "_active_cscan_filename_bundle", None)
                if active is None:
                    filename = self.file_naming.get_filename(
                        "cscan",
                        acq_mode,
                        [total_y_pixels, x_pixels, z_pixels],
                    )
                    active = {"filename": filename, "log_filename": filename}
                    self._active_cscan_filename_bundle = active
                if is_final:
                    self.file_naming.increment_cscan()
                    self._active_cscan_filename_bundle = None
                return dict(active)

            filename = self.file_naming.get_filename("cscan", acq_mode, [y_pixels, x_pixels, z_pixels])
            self.file_naming.increment_cscan()
            return {"filename": filename, "log_filename": filename}

        return {}

    def wait_for_processing_barrier(self, label="", poll_interval=0.2, stop_if_run_unchecked=True):
        label = label or "the next sample"
        while self.ui.RunButton.isChecked() or not stop_if_run_unchecked:
            gpu_pending, gpu_active, dns_pending, dns_active = self.processing_backlog()
            if gpu_pending <= 0 and gpu_active <= 0 and dns_pending <= 0 and dns_active <= 0:
                return True
            self.emit_status(
                "Waiting for processing of the previous sample before "
                f"{label}. GPU queue={gpu_pending}, GPU active={gpu_active}, "
                f"DnS queue={dns_pending}, DnS active={dns_active}."
            )
            time.sleep(poll_interval)
            if stop_if_run_unchecked and not self.ui.RunButton.isChecked():
                return False
        return False
        
    
    def reset_memory_free_queue(self):
        if not hasattr(self, "MemoryFreeQueue"):
            return
        while True:
            try:
                self.MemoryFreeQueue.get_nowait()
            except Empty:
                break
        for slot in range(self.memoryCount):
            self.MemoryFreeQueue.put(slot)

    def release_memory_slot(self, memory_slot):
        if memory_slot is None or not hasattr(self, "MemoryFreeQueue"):
            return
        self.MemoryFreeQueue.put(memory_slot)

    def cscan_package_frame_count(self):
        package_y_lines = int(getattr(self, 'cscan_package_y_lines', 64))
        return max(1, package_y_lines) * max(1, self.current_bline_avg())

    def InitMemory(self):
        #################################################################
        # get number samplers per Aline
        samples = self.current_nsamples()
        alines_per_bline = self.current_alines_per_bline()
        configured_bline_avg = self.current_bline_avg()
        configured_y_pixels = self.current_y_pixels()
        configured_dynamic = self.current_dynamic_enabled()
        configured_acq_mode = self.current_acq_mode()
        # print(self.ui.PixelFormat_display.text())
        if camera_pixel_format(self.ui) in ['Mono8']:
            data_type =  np.uint8
        else:
            data_type =  np.uint16
            
        self.reset_memory_free_queue()
        total_cscan_frames = configured_y_pixels * configured_bline_avg
        cscan_package_frames = min(total_cscan_frames, self.cscan_package_frame_count())

        for ii in range(self.memoryCount):
             if configured_acq_mode in (
                 AcqTypes.FINITE_BLINE,
                 AcqTypes.CONTINUOUS_BLINE,
                 AcqTypes.CONTINUOUS_KX,
             ):
                 self.Memory[ii]=np.zeros([configured_bline_avg, alines_per_bline, samples], dtype = data_type)
                 self.NAcq = 1
             elif configured_acq_mode == AcqTypes.CONTINUOUS_CSCAN:
                 self.Memory[ii]=np.zeros([configured_y_pixels*configured_bline_avg, alines_per_bline, samples], dtype = data_type)
                 self.NAcq = 1
             elif configured_acq_mode in (
                 AcqTypes.FINITE_CSCAN,
             ):
                 self.Memory[ii]=np.zeros([cscan_package_frames, alines_per_bline, samples], dtype = data_type)
                 self.NAcq = int(np.ceil(total_cscan_frames / max(1, cscan_package_frames)))
                 self._active_cscan_filename_bundle = None

        ###########################################################################################
        
    def SingleScan(self, DnS_action, acq_mode, context=None, skip_save=False):
        # an_action = DnSActionField(DnSActions.CLEAR)
        # self.DnSQueue.put(an_action)
        if not self.wait_for_processing_barrier(label=f"starting {DnS_action}"):
            return f"{DnS_action} stopped by user."
        self.drain_continuous_backlog(reason=f"before {DnS_action}")
        fft_device = self.current_fft_device()
        t0=time.time()
        # print(self.DbackQueue.qsize())
        an_action = DActionField('ConfigureBoard')
        self.DQueue.put(an_action)
        # self.DbackQueue.get()
        t1=time.time()
        ###########################################################################################
        # start AODO 
        an_action = AODOActionField('ConfigTask')
        self.AODOQueue.put(an_action)
        self.StagebackQueue.get()
        t2=time.time()
        # start camera

        self.drain_queue(self.DbackQueue, "DbackQueue")
        an_action = DActionField('Acquire')
        self.DQueue.put(an_action)
        self.DbackQueue.get()
        t3=time.time()

        # print('current dbackqueue size:', self.DbackQueue.qsize())
        an_action = AODOActionField('StartTask')
        self.AODOQueue.put(an_action)
        self.StagebackQueue.get()
        t4=time.time()
        # print('\n')
        # print('Camera config took: ',round(t1-t0,3),'sec')
        # print('Camera start took: ',round(t3-t2,3),'sec')
        # print('Galvo board start took: ',round(t4-t3,3),'sec')
        # print('current dbackqueue size:', self.DbackQueue.qsize())
        # print('\n')
        message = f"{DnS_action} stopped by user."
        acquisition_failed = False
        for iAcq in range(self.NAcq):
            start = time.time()
            dynamic_bline_idx = None
            ######################################### collect data
            # collect data from digitizer, data format: [Y pixels, Xpixels, Z pixels]
            # print('waiting for camera data...')
            while self.ui.RunButton.isChecked():
                try:
                    an_action = self.DatabackQueue.get(timeout = 2.5)
                    if getattr(an_action, "error", None):
                        message = str(an_action.error)
                        print(message)
                        acquisition_failed = True
                        break
                    # print('camera queue size:', self.DatabackQueue.qsize())
                    # print('time to fetch data: '+str(round(time.time()-start,3))+'sec')
                    memory_slot = an_action.memory_slot
                    package_start = getattr(an_action, "package_start", 0)
                    package_frame_count = getattr(an_action, "package_frame_count", None)
                    package_total_frames = getattr(an_action, "package_total_frames", None)
                    is_final = getattr(an_action, "is_final", True)
                    package_profile = dict(getattr(an_action, "profile", {}) or {})
                    package_profile["weaver_received_s"] = time.perf_counter()
                    filename_bundle = self.build_filename_bundle(
                        DnS_action,
                        acq_mode,
                        memory_slot,
                        raw=(fft_device in ['None']),
                        package_start=package_start,
                        package_frame_count=package_frame_count,
                        package_total_frames=package_total_frames,
                        is_final=is_final,
                    )
                    # print(memory_slot)
                    ############################################### display and save data
                    if fft_device in ['None']:
                        # put raw spectrum data into memory for dipersion compensation and background subtraction usage
                        valid_frames = self.Memory[memory_slot].shape[0] if package_frame_count is None else int(package_frame_count)
                        self.data = self.Memory[memory_slot][:valid_frames].copy()
                        self.release_memory_slot(memory_slot)
                        # In None mode, directly do display and save
                        if np.sum(self.data)<10:
                            message = "No usable spectral data received."
                            print(message)
                        else:
                            an_action = DnSActionField(
                                DnS_action,
                                acq_mode=acq_mode,
                                data=self.data,
                                raw=True,
                                context=context,
                                dynamic_bline_idx=dynamic_bline_idx,
                                filename_bundle=filename_bundle,
                                skip_save=skip_save,
                                package_start=package_start,
                                package_frame_count=package_frame_count,
                                package_total_frames=package_total_frames,
                                is_final=is_final,
                                profile=package_profile,
                            )
                            an_action.profile["dns_queue_put_s"] = time.perf_counter()
                            self.DnSQueue.put(an_action)
                            message = f"{DnS_action} completed."
                    else:
                        # In other modes, do FFT first
                        an_action = GPUActionField(
                            action=fft_device,
                            DnS_action=DnS_action,
                            acq_mode=acq_mode,
                            memory_slot=memory_slot,
                            context=context,
                            dynamic_bline_idx=dynamic_bline_idx,
                            filename_bundle=filename_bundle,
                            skip_save=skip_save,
                            package_start=package_start,
                            package_frame_count=package_frame_count,
                            package_total_frames=package_total_frames,
                            is_final=is_final,
                            profile=package_profile,
                        )
                        an_action.profile["gpu_queue_put_s"] = time.perf_counter()
                        self.GPUQueue.put(an_action)
                        message = f"{DnS_action} completed."
                    break
                except:
                    print(f"{DnS_action}: waiting for camera data...")
            if acquisition_failed:
                break
                    
        an_action = AODOActionField('tryStopTask')
        self.AODOQueue.put(an_action)
        an_action = AODOActionField('CloseTask')
        self.AODOQueue.put(an_action)
        self.StagebackQueue.get() # wait for AODO CloseTask
        self.finalize_partial_dynamic_naming(acq_mode)
        self.wait_for_processing_barrier(label=f"finishing {DnS_action}", stop_if_run_unchecked=False)
        print(message)
        return message
    
            
    def RptScan(self, DnS_action, acq_mode):
        if not self.wait_for_processing_barrier(label=f"starting {DnS_action}"):
            return f"{DnS_action} stopped by user."
        fft_device = self.current_fft_device()
        # an_action = DnSActionField(DnSActions.CLEAR)
        # self.DnSQueue.put(an_action)
        frame_rate = self.ui.FrameRate_DH.value()
        if acq_mode in tuple(mode for mode in BLINE_MODES if mode in CONTINUOUS_MODES):
            self.ui.FrameRate_DH.setValue(50)
        an_action = DActionField('ConfigureBoard')
        self.DQueue.put(an_action)
        # self.DbackQueue.get()
        # config AODO
        an_action = AODOActionField('ConfigTask')
        self.AODOQueue.put(an_action)
        self.StagebackQueue.get()
        data_backs = 0 # count number of data backs
        skipped_fft_actions = 0
        camera_error_message = None

        # start digitizer for one acuquqisition
        self.drain_queue(self.DbackQueue, "DbackQueue")
        an_action = DActionField('Acquire')
        self.DQueue.put(an_action)
        self.DbackQueue.get()

        # start AODO 
        an_action = AODOActionField('StartTask')
        self.AODOQueue.put(an_action)
        self.StagebackQueue.get()

        ######################################################### repeat acquisition until Stop button is clicked
        while self.ui.RunButton.isChecked():
            ######################################### collect data
            try: # use try-except in cases where Stop button clicked and camera stopped prior to while loop
                start = time.time()
                an_action = self.DatabackQueue.get(timeout=2) # never time out
                if getattr(an_action, "error", None):
                    message = str(an_action.error)
                    print(message)
                    camera_error_message = message
                    break
                # print('time to fetch data: '+str(round(time.time()-start,3)))
                memory_slot = an_action.memory_slot
                # print(memory_slot)
                data_backs += 1
                
                ######################################### display data
                if fft_device in ['None']:
                    filename_bundle = self.build_filename_bundle(DnS_action, acq_mode, memory_slot, raw=True)
                    # put raw spectrum data into memory for dipersion compensation and background subtraction usage
                    self.data = self.Memory[memory_slot].copy()
                    self.release_memory_slot(memory_slot)
                    # In None mode, directly do display and save
                    if np.sum(self.data)<10:
                        message = "No usable spectral data received."
                        print(message)
                    else:
                        an_action = DnSActionField(
                            DnS_action,
                            acq_mode=acq_mode,
                            data=self.data,
                            raw=True,
                            filename_bundle=filename_bundle,
                            skip_save=False,
                        )
                        self.DnSQueue.put(an_action)
                        message = f"{DnS_action} completed."
                else:
                    # In other modes, do FFT first
                    if self.GPUQueue.qsize() == 0:
                        filename_bundle = self.build_filename_bundle(DnS_action, acq_mode, memory_slot, raw=False)
                        an_action = GPUActionField(
                            fft_device,
                            DnS_action=DnS_action,
                            acq_mode=acq_mode,
                            memory_slot=memory_slot,
                            filename_bundle=filename_bundle,
                            skip_save=False,
                        )
                        self.GPUQueue.put(an_action)
                    else:
                        skipped_fft_actions += 1
                    message = f"{DnS_action} completed."
                ######################################## check if Pause or Stop button is clicked
            except:
                pass
                # print('camera stopped')
            # handle pause action
            if self.ui.PauseButton.isChecked():
                # camera will wait for trigger, no need to stop
                # stop AODO task, can be restarted
                an_action = AODOActionField('StopTask')
                self.AODOQueue.put(an_action)
                # wait until stop button or pause button is clicked
                while self.ui.PauseButton.isChecked() and self.ui.RunButton.isChecked():
                    time.sleep(0.5)
                # if resume, restart AODO task
                if not self.ui.PauseButton.isChecked():
                    # start AODO 
                    an_action = AODOActionField('StartTask')
                    self.AODOQueue.put(an_action)
                    self.StagebackQueue.get()
        # Camera will stop once Stop Button is clicked
        # AODO thread will need StopTask command
        an_action = AODOActionField('tryStopTask')
        self.AODOQueue.put(an_action)
        # close AODO
        an_action = AODOActionField('CloseTask')
        self.AODOQueue.put(an_action)
        self.StagebackQueue.get() # wait for AODO CloseTask
        # digitizer will close automatically
        self.finalize_partial_dynamic_naming(acq_mode)
        if camera_error_message is None:
            message = f"{DnS_action} stopped. Received {data_backs} camera buffer(s)."
            if skipped_fft_actions:
                message += f" Skipped {skipped_fft_actions} stale continuous FFT request(s)."
        else:
            message = camera_error_message
        print(message)
        self.drain_continuous_backlog(reason=f"after {DnS_action}")
        an_action = GPUActionField(GPUActions.DISPLAY_FFT_ACTIONS)
        self.GPUQueue.put(an_action)
        an_action = GPUActionField(GPUActions.DISPLAY_COUNTS, context=DnS_action)
        self.GPUQueue.put(an_action)
        self.wait_for_processing_barrier(label=f"finishing {DnS_action}", stop_if_run_unchecked=False)
        self.ui.FrameRate_DH.setValue(frame_rate)
        return message
  

    def move_stage_axis(self, axis, target, tolerance=0.005):
        position_widget = getattr(self.ui, f"{axis}Position")
        current_widget = getattr(self.ui, f"{axis}current")
        position_widget.setValue(target)

        current = current_widget.value()
        distance = target - current
        if abs(distance) <= tolerance:
            return

        an_action = AODOActionField(f"{axis}move2")
        self.AODOQueue.put(an_action)
        timeout = self.stage_move_timeout(axis, distance)
        try:
            self.StagebackQueue.get(timeout=timeout)
        except Empty:
            message = (
                f"Stage move timeout during {axis} move: "
                f"target={target:.4f}, current={current:.4f}, "
                f"distance={distance:.4f}, timeout={timeout:.1f}s."
            )
            print(message)
            self.emit_status(message + " Assuming motion completed and continuing.")
            return False
        return True

    def stage_move_timeout(self, axis, distance):
        return 300.0

    def get_background(self):
        # an_action = AODOActionField('rotate_servo_out')
        # self.AODOQueue.put(an_action)
        # self.StagebackQueue.get()
        # print('start getting background...')
        acq_mode = self.ui.ACQMode.currentText()
        fft_device = self.ui.FFTDevice.currentText()
        BAvg = self.ui.BlineAVG.value()
        self.ui.ACQMode.setCurrentText(AcqTypes.FINITE_BLINE)
        self.ui.FFTDevice.setCurrentText('None')
        self.ui.BlineAVG.setValue(20)
        ############################# measure an Aline
        # print('acquiring Bline')
        self.ui.RunButton.setChecked(True)
        self.InitMemory()
        self.SingleScan(
            DnS_action=AcqTypes.FINITE_BLINE,
            acq_mode=AcqTypes.FINITE_BLINE,
            skip_save=True,
        )
        # print('got Bline')
        # print(self.data.shape)
        #######################################################################
        Xpixels = self.ui.AlinesPerBline.value()
        Yrpt = self.ui.BlineAVG.value()
        BLINE = self.data.reshape([Yrpt, Xpixels, self.current_nsamples()])
        
        background = np.float32(np.mean(BLINE,0))
        # plt.figure()
        # plt.imshow(background)
        # plt.show()
        # background = np.smooth()
        # print(background.shape)
        filePath = self.ui.DIR.toPlainText()
        current_time = datetime.datetime.now()
        filePath = filePath + "/" + 'background_'+\
            str(current_time.year)+'-'+\
            str(current_time.month)+'-'+\
            str(current_time.day)+'-'+\
            str(current_time.hour)+'-'+\
            str(current_time.minute)+'-'+\
            str(current_time.second)+\
            '.bin'
        fp = open(filePath, 'wb')
        background.tofile(fp)
        fp.close()
        
        self.ui.BG_DIR.setText(filePath)
        self.ui.ACQMode.setCurrentText(acq_mode)
        self.ui.FFTDevice.setCurrentText(fft_device)
        self.ui.BlineAVG.setValue(BAvg)
        
        # an_action = AODOActionField('rotate_servo_back')
        # self.AODOQueue.put(an_action)
        # self.StagebackQueue.get()
        return "Background measurement completed."

    def get_background_cscan(self):
        acq_mode = self.ui.ACQMode.currentText()
        fft_device = self.ui.FFTDevice.currentText()
        bline_avg = self.ui.BlineAVG.value()

        self.ui.ACQMode.setCurrentText(AcqTypes.FINITE_CSCAN)
        self.ui.FFTDevice.setCurrentText('None')
        self.ui.BlineAVG.setValue(1)

        self.ui.RunButton.setChecked(True)
        self.InitMemory()
        self.SingleScan(
            DnS_action=AcqTypes.FINITE_CSCAN,
            acq_mode=AcqTypes.FINITE_CSCAN,
            skip_save=True,
        )

        background = np.float32(np.mean(self.data, axis=0))

        filePath = self.ui.DIR.toPlainText()
        current_time = datetime.datetime.now()
        filePath = filePath + "/" + 'background_'+\
            str(current_time.year)+'-'+\
            str(current_time.month)+'-'+\
            str(current_time.day)+'-'+\
            str(current_time.hour)+'-'+\
            str(current_time.minute)+'-'+\
            str(current_time.second)+\
            '.bin'
        fp = open(filePath, 'wb')
        background.tofile(fp)
        fp.close()

        self.ui.BG_DIR.setText(filePath)
        self.ui.ACQMode.setCurrentText(acq_mode)
        self.ui.FFTDevice.setCurrentText(fft_device)
        self.ui.BlineAVG.setValue(bline_avg)
        return "Background measurement completed."
    
    def get_surfCurve(self):
        
        print('start getting background...')
        acq_mode = self.ui.ACQMode.currentText()
        fft_device = self.ui.FFTDevice.currentText()
        self.ui.ACQMode.setCurrentText(AcqTypes.FINITE_BLINE)
        self.ui.FFTDevice.setCurrentText('GPU')
        if getattr(self, "gpu_thread", None) is not None:
            self.gpu_thread.return_processed_to_weaver = True
        ############################# measure an Cscan
        print('acquiring Bline')
        self.ui.RunButton.setChecked(True)
        self.InitMemory()
        self.SingleScan(DnS_action='SingleCscan', acq_mode='SingleCscan')
        # while self.GPU2weaverQueue.qsize()<1:
        #     time.sleep(1)
        cscan =self.GPU2weaverQueue.get()
        
        Zpixels = self.ui.DepthRange.value()
        # get number of X pixels
        Xpixels = self.ui.AlinesPerBline.value()
        # get number of Y pixels
        Ypixels = self.ui.Ypixels.value()* self.ui.BlineAVG.value()
        # reshape into Ypixels x Xpixels x Zpixels
        cscan = cscan.reshape([Ypixels,Xpixels,Zpixels])
        
        Bline = np.float32(np.mean(cscan,0))
        surfCurve = np.zeros([Xpixels])

        plt.figure()
        plt.imshow(Bline)
        plt.title('Bline for finding surface')
        for xx in range(Xpixels):
            surfCurve[xx] = findchangept(Bline[xx,:],1)
        
        surfCurve = surfCurve - min(surfCurve)
        plt.figure()
        plt.plot(surfCurve)
        plt.title('surface')
        plt.figure()

        filePath = self.ui.DIR.toPlainText()
        filePath = filePath + "/" + 'surfCurve.bin'
        fp = open(filePath, 'wb')
        np.uint16(surfCurve).tofile(fp)
        fp.close()
        
        self.ui.Surf_DIR.setText(filePath)
        self.ui.ACQMode.setCurrentText(acq_mode)
        self.ui.FFTDevice.setCurrentText(fft_device)
        if getattr(self, "gpu_thread", None) is not None:
            self.gpu_thread.return_processed_to_weaver = False
        return "Surface curve measurement completed."
    
    def live(self):
        """Continuously captures and displays video until RunButton is unchecked."""
        cap = open_usb_camera(configure_exposure=True)
    
        while self.ui.RunButton.isChecked():
            ret, frame = cap.read()
            if not ret:
                break
    
            frame = orient_usb_frame(frame)
            
            # 3. Convert BGR to RGB
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            
            # 4. Convert to QImage then QPixmap
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
    
            # 5. Display on Label (scaled to fit the widget)
            # Note: If this is a separate thread, UI updates should ideally 
            # use Signals, but for a simple script, this often works:
            self.ui.XZplane.setPixmap(pixmap.scaled(
                self.ui.XZplane.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
    
        # 6. Release resources when button is unchecked
        
        
    


