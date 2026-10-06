
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 20:14:40 2023

@author: Shuaibin Chang
"""

# Queue functions:
# maxsize - Number of items allowed in the queue.
# empty() - Return True if the queue is empty, False otherwise.
# full() - Return True if the queue has maxsize items; maxsize=0 means it never reports full.
# get() - Remove and return an item, waiting if the queue is empty.
# get_nowait() - Return an item immediately, or raise QueueEmpty.
# put(item) - Add an item, waiting if the queue is full.
# put_nowait(item) - Add an item immediately, or raise QueueFull.
# qsize() - Return the number of items in the queue.

# TODO: revisit pause handling so it does not depend on thread interruption.
import sys
from queue import Queue
from PyQt5.QtWidgets import QApplication
from PyQt5 import QtWidgets as QW
from PyQt5.QtCore import QObject, pyqtSignal
from mainWindow import MainWindow
from ActionFields import AODOActionField, DActionField, EXITField, GPUActionField, WeaverActionField
from ActionTypes import AcqTypes, GPUActions, WeaverActions
from FileNaming import FileNaming
from Generaic_functions import LOG
import time
from Display_rendering import (
    render_aodo_waveform_ready,
    render_bline_ready,
    render_cscan_ready,
)

CONTINUOUS_ACQ_MODES = (
    AcqTypes.CONTINUOUS_BLINE,
    "ContinueBline",
    AcqTypes.CONTINUOUS_CSCAN,
)

FINITE_ACQ_MODES = (
    AcqTypes.FINITE_BSCAN,
    AcqTypes.X_FREQUENCY,
    AcqTypes.FINITE_CSCAN,
)

LIVE_ONLY_MODES = (
    AcqTypes.LOCATION_CAMERA_LIVE,
)
# Shared raw-data ring buffer. More than two slots allows acquisition and processing to overlap safely.
global memoryCount
memoryCount = 4


# Number of processed Cscan Y-lines per streaming package.
CSCAN_PACKAGE_Y_LINES = 64
global Memory
Memory = list(range(memoryCount))


# Simulation switch for running without live hardware.
global SIM
SIM = False
# Optional mayavi-style 3D visualization path.
global use_maya
use_maya = False

# Queue topology for thread-to-thread coordination.
WeaverQueue = Queue()
# Queue for galvo / stage control actions.
AODOQueue = Queue()
# Status/ack queue from the galvo / stage thread back to Weaver.
StagebackQueue = Queue()
# Queue for display and save actions.
DnSQueue = Queue()
# Queue for GPU / CPU FFT processing actions.
GPUQueue = Queue()
# Status/ack queue from GPU thread back to Weaver.
GPU2weaverQueue = Queue()
# Queue for camera-control actions.
DQueue = Queue()
# Status/ack queue from the camera thread back to Weaver.
DbackQueue = Queue()
# Queue carrying completed raw-data memory slots.
DatabackQueue = Queue()
# Queue carrying memory slots that are safe for the camera to reuse.
MemoryFreeQueue = Queue()
# DnS returns GPU pinned-output buffers only after processing/saving is done.
PinnedOutputFreeQueue = Queue()

        
# Thread wrappers inject shared queues, memory, logging, and the UI bridge.

from ThreadCamera_AK import Camera
class Camera_2(Camera):
    def __init__(self, ui, log):
        super().__init__()
        global Memory
        self.memoryCount = memoryCount
        self.Memory = Memory
        self.ui = ui
        self.queue = DQueue
        self.DbackQueue = DbackQueue
        self.DatabackQueue = DatabackQueue
        self.MemoryFreeQueue = MemoryFreeQueue
        self.log = log
        self.SIM = SIM    
        self.ui_bridge = None
            

from ThreadWeaver import WeaverThread
class WeaverThread_2(WeaverThread):
    def __init__(self, ui, log):
        super().__init__()
        global Memory
        self.Memory = Memory
        self.memoryCount = memoryCount
        self.ui = ui
        self.queue = WeaverQueue
        self.DnSQueue = DnSQueue
        self.AODOQueue = AODOQueue
        self.StagebackQueue = StagebackQueue
        self.DbackQueue = DbackQueue
        self.DatabackQueue = DatabackQueue
        self.MemoryFreeQueue = MemoryFreeQueue
        self.cscan_package_y_lines = CSCAN_PACKAGE_Y_LINES
        self.GPUQueue = GPUQueue
        self.DQueue = DQueue
        self.GPU2weaverQueue = GPU2weaverQueue
        self.log = log
        self.ui_bridge = None

# GPU processing thread wrapper.
from ThreadGPU import GPUThread
class GPUThread_2(GPUThread):
    def __init__(self, ui, log):
            super().__init__()
            global Memory
            self.Memory = Memory
            self.ui = ui
            self.queue = GPUQueue
            self.DnSQueue = DnSQueue
            self.GPU2weaverQueue = GPU2weaverQueue
            self.MemoryFreeQueue = MemoryFreeQueue
            self.PinnedOutputFreeQueue = PinnedOutputFreeQueue
            self.log = log
            self.SIM = SIM
            self.AMPLIFICATION = 100#AMPLIFICATION
            self.ui_bridge = None
            
# Galvo / stage control thread wrapper.
from ThreadAODO_art import AODOThread
class AODOThread_2(AODOThread):
    def __init__(self, ui, log):
        super().__init__()
        self.ui = ui
        self.queue = AODOQueue
        self.StagebackQueue = StagebackQueue
        self.log = log
        self.SIM = SIM
        self.ui_bridge = None

# Display and save thread wrapper.
from ThreadDnS import DnSThread
class DnSThread_2(DnSThread):
    def __init__(self, ui, log):
        super().__init__()
        self.ui = ui
        self.queue = DnSQueue
        self.log = log
        self.use_maya = use_maya
        self.PinnedOutputFreeQueue = PinnedOutputFreeQueue
        self.ui_bridge = None
        

# GUI-thread bridge for cross-thread status and display payload delivery.
class UiBridge(QObject):
    status_message = pyqtSignal(str)
    acquisition_controls_locked = pyqtSignal(bool)
    cu_slice_value = pyqtSignal(int)
    time_reader_value = pyqtSignal(int)
    aline_ready = pyqtSignal(object)   # dict payload
    bline_ready = pyqtSignal(object)   # dict payload
    cscan_ready = pyqtSignal(object)   # dict payload
    aodo_waveform_ready = pyqtSignal(object)  # dict payload


# Main GUI object with thread wiring and queue orchestration.
class GUI(MainWindow):
    def __init__(self):
        super().__init__()
        # if use_maya:
        #     self.addMaya()
        self.log = LOG(self.ui)
        self.log.install_stream_redirects()
        
        
        self.ui.RunButton.clicked.connect(self.run_task)
        self.ui.PauseButton.clicked.connect(self.Pause_task)
        self.ui.CenterGalvo.clicked.connect(self.CenterGalvo)
        if hasattr(self.ui, "FFTresults"):
            self.ui.FFTresults.clear()
            self.ui.FFTresults.addItem("AMP")
        # set window length for FFT
        # self.ui.PostSamples.valueChanged.connect(self.update_Dispersion)
        # self.ui.PreSamples.valueChanged.connect(self.update_Dispersion)
        # self.ui.PostSamples_2.valueChanged.connect(self.update_Dispersion)
        # self.ui.DelaySamples.valueChanged.connect(self.update_Dispersion)
        # self.ui.TrimSamples.valueChanged.connect(self.update_Dispersion)
        # set stage boundary
        self.ui.XZmax.valueChanged.connect(self.Update_contrast)
        # self.ui.DepthStart.valueChanged.connect(self.Update_contrast_Bline)
        # self.ui.DepthRange.valueChanged.connect(self.Update_contrast_Bline)
        self.ui.XZmin.valueChanged.connect(self.Update_contrast)
        if hasattr(self.ui, "ZDepthBar"):
            self.ui.ZDepthBar.valueChanged.connect(self.Update_contrast)
        if hasattr(self.ui, "BlineBar"):
            self.ui.BlineBar.valueChanged.connect(self.Update_contrast)
        self.ui.redoBG.clicked.connect(self.redo_background)
        self.ui.redoSurf.clicked.connect(self.redo_surface)
        self.ui.BG_DIR.textChanged.connect(self.update_background)
        self.ui.Xpixels.valueChanged.connect(self.update_background)
        self.ui.XGalvoBias.valueChanged.connect(self.update_background)
        self.ui.NSamples_PF.valueChanged.connect(self.update_background)
        self.ui.NSamples_DH.valueChanged.connect(self.update_background)
        if hasattr(self.ui, "NSamples_AK"):
            self.ui.NSamples_AK.valueChanged.connect(self.update_background)
        self.ui.SpectralDS_DH.valueChanged.connect(self.update_background)
        self.ui.SpectralDS_DH.valueChanged.connect(self.update_Dispersion)
        self.ui.SpectralDS_PF.valueChanged.connect(self.update_background)
        self.ui.SpectralDS_PF.valueChanged.connect(self.update_Dispersion)
        if hasattr(self.ui, "SpectralDS_AK"):
            self.ui.SpectralDS_AK.valueChanged.connect(self.update_background)
            self.ui.SpectralDS_AK.valueChanged.connect(self.update_Dispersion)
        self.ui.offsetW_PF.valueChanged.connect(self.update_background)
        self.ui.offsetW_DH.valueChanged.connect(self.update_background)
        if hasattr(self.ui, "offsetW_AK"):
            self.ui.offsetW_AK.valueChanged.connect(self.update_background)
        self.ui.Camera.currentTextChanged.connect(self.update_background)
        self.ui.Camera.currentTextChanged.connect(self.update_Dispersion)
        self.ui.InD_DIR.textChanged.connect(self.update_Dispersion)
        self.ui.Xmove2.clicked.connect(self.Xmove2)
        self.ui.Ymove2.clicked.connect(self.Ymove2)
        self.ui.Zmove2.clicked.connect(self.Zmove2)
        self.ui.XUP.clicked.connect(self.XUP)
        self.ui.YUP.clicked.connect(self.YUP)
        self.ui.ZUP.clicked.connect(self.ZUP)
        self.ui.XHome.clicked.connect(self.XHome)
        self.ui.YHome.clicked.connect(self.YHome)
        self.ui.ZHome.clicked.connect(self.ZHome)
        self.ui.XDOWN.clicked.connect(self.XDOWN)
        self.ui.YDOWN.clicked.connect(self.YDOWN)
        self.ui.ZDOWN.clicked.connect(self.ZDOWN)
        
        self.ui.XSpeed.valueChanged.connect(self.SetXSpeed)
        self.ui.YSpeed.valueChanged.connect(self.SetYSpeed)
        self.ui.ZSpeed.valueChanged.connect(self.SetZSpeed)
        
        self.ui.XAccelerate.valueChanged.connect(self.SetXAcc)
        self.ui.YAccelerate.valueChanged.connect(self.SetYAcc)
        self.ui.ZAccelerate.valueChanged.connect(self.SetZAcc)
        
        self.ui.InitStageButton.clicked.connect(self.InitStages)
        self.ui.StageUninit.clicked.connect(self.Uninit)
        if hasattr(self.ui, "SliceN"):
            self.ui.SliceN.valueChanged.connect(self._on_slice_n_changed)
        
        # testing buttons
        self.ui.TestButten1.clicked.connect(self.TestButton1Func)
        self.ui.TestButten2.clicked.connect(self.TestButton2Func)
        self.ui.TestButten3.clicked.connect(self.TestButton3Func)

        # UI bridge (must live on GUI thread)
        self._ui_bridge = UiBridge()
        self._ui_bridge.status_message.connect(self._on_status_message)
        self._ui_bridge.acquisition_controls_locked.connect(self._on_acquisition_controls_locked)
        self._ui_bridge.cu_slice_value.connect(self._on_cu_slice_value)
        self._ui_bridge.time_reader_value.connect(self._on_time_reader_value)
        self._ui_bridge.bline_ready.connect(self._on_bline_ready)
        self._ui_bridge.cscan_ready.connect(self._on_cscan_ready)
        self._ui_bridge.aodo_waveform_ready.connect(self._on_aodo_waveform_ready)
        # AiKe 页选了新的 .vlcf：空闲时立即让相机线程重新设置，采集时等下次扫描。
        if hasattr(self, "ak_vlcf_changed"):
            self.ak_vlcf_changed.connect(self._on_ak_vlcf_changed)
        # 曝光/行频/增益改动：立即让相机线程重设这三个相机参数。
        if hasattr(self, "ak_runtime_changed"):
            self.ak_runtime_changed.connect(self._on_ak_runtime_changed)
        # 标尺轴向尺度（水中/空气）切换：用缓存的最后一次 payload 重画显示。
        if hasattr(self, "ruler_scale_changed"):
            self.ruler_scale_changed.connect(self._on_ruler_scale_changed)
        if hasattr(self.ui, "SliceN"):
            self._on_slice_n_changed(self.ui.SliceN.value())
        self._last_display_payloads = {
            "bline": None,
            "cscan": None,
        }
        self._acquisition_lock_depth = 0
        self._locked_widget_states = {}
        
        # Init all threads
        self.Init_allThreads()
        # Simple FPS limiter for rendering-heavy slots
        self._render_fps_limit = 30.0
        self._last_render_t = {"bline": 0.0, "cscan": 0.0}

    def Init_allThreads(self):
        self.Weaver_thread = WeaverThread_2(self.ui, self.log)
        self.AODO_thread = AODOThread_2(self.ui, self.log)
        self.DnS_thread = DnSThread_2(self.ui, self.log)
        self.GPU_thread = GPUThread_2(self.ui, self.log)
        self.D_thread = Camera_2(self.ui, self.log)
        self.file_naming = FileNaming(self.ui)

        # Inject the GUI-thread bridge into worker threads.
        self.Weaver_thread.ui_bridge = self._ui_bridge
        self.AODO_thread.ui_bridge = self._ui_bridge
        self.DnS_thread.ui_bridge = self._ui_bridge
        self.GPU_thread.ui_bridge = self._ui_bridge
        self.D_thread.ui_bridge = self._ui_bridge
        self.Weaver_thread.file_naming = self.file_naming
        self.Weaver_thread.gpu_thread = self.GPU_thread
        self.Weaver_thread.dns_thread = self.DnS_thread
        
        self.D_thread.start()
        self.GPU_thread.start()
        self.Weaver_thread.start()
        self.AODO_thread.start()
        self.DnS_thread.start()

    def _fps_ok(self, key: str) -> bool:
        now = time.monotonic()
        min_dt = 1.0 / max(self._render_fps_limit, 1.0)
        if now - self._last_render_t.get(key, 0.0) < min_dt:
            return False
        self._last_render_t[key] = now
        return True

    def _on_status_message(self, msg: str):
        self.ui.statusbar.showMessage(msg)

    def _on_ruler_scale_changed(self):
        """标尺轴向尺度切换：用缓存的最后一次 payload 重画，立刻看到新刻度。"""
        self.Update_contrast()

    def _on_ak_runtime_changed(self):
        """用户改了曝光/行频/增益：让相机线程立即重设对应相机参数（采集时延后生效）。"""
        try:
            DQueue.put(DActionField("ApplyRuntimeParams"))
        except Exception as error:
            self._ui_bridge.status_message.emit("下发相机参数失败: %s" % error)

    def _on_ak_vlcf_changed(self, vlcf_path: str):
        """用户换了 .vlcf：空闲时立刻让相机线程重新配置，采集进行中则等下次扫描。"""
        try:
            if self.ui.RunButton.isChecked():
                self._ui_bridge.status_message.emit(
                    "已选择 VLCF: %s；当前正在采集，将在下次扫描开始时生效。" % vlcf_path
                )
                return
            DQueue.put(DActionField("ConfigureBoard"))
            self._ui_bridge.status_message.emit("已请求相机按 %s 重新设置。" % vlcf_path)
        except Exception as error:
            self._ui_bridge.status_message.emit("请求相机重设失败: %s" % error)

    def _on_slice_n_changed(self, value: int):
        if hasattr(self.ui, "CuSlice"):
            self.ui.CuSlice.setValue(int(value))

    def _on_cu_slice_value(self, value: int):
        if hasattr(self.ui, "CuSlice"):
            self.ui.CuSlice.setValue(int(value))

    def _on_time_reader_value(self, value: int):
        if hasattr(self.ui, "timeReader"):
            self.ui.timeReader.setValue(int(value))

    def _wait_stageback(self, label, timeout=300.0):
        timeout = float(timeout)
        try:
            StagebackQueue.get(timeout=timeout)
        except Exception:
            message = (
                f"Stage timeout while waiting for {label} acknowledgement "
                f"after {timeout:.1f}s. "
                "Assuming motion completed and continuing."
            )
            print(message)
            self.ui.statusbar.showMessage(message)
            return False
        return True

    def stage_move_timeout(self, axis):
        return 300.0

    def _managed_acquisition_widgets(self):
        widget_types = (
            QW.QAbstractButton,
            QW.QAbstractSpinBox,
            QW.QComboBox,
            QW.QLineEdit,
            QW.QTextEdit,
            QW.QPlainTextEdit,
            QW.QAbstractSlider,
        )
        return [widget for widget in self.findChildren(QW.QWidget) if isinstance(widget, widget_types)]

    def _live_acquisition_widgets(self):
        live_widgets = set()
        live_names = (
            "RunButton",
            "PauseButton",
            "Xmove2",
            "Ymove2",
            "Zmove2",
            "XUP",
            "YUP",
            "ZUP",
            "XDOWN",
            "YDOWN",
            "ZDOWN",
            "XHome",
            "YHome",
            "ZHome",
            "XPosition",
            "YPosition",
            "ZPosition",
            "XStepSize",
            "Xstagestepsize",
            "Ystagestepsize",
            "Zstagestepsize",
            "XSpeed",
            "YSpeed",
            "ZSpeed",
            "XAccelerate",
            "YAccelerate",
            "ZAccelerate",
        )
        for name in live_names:
            widget = getattr(self.ui, name, None)
            if isinstance(widget, QW.QWidget):
                live_widgets.add(widget)
        for widget in self._managed_acquisition_widgets():
            if isinstance(widget, QW.QAbstractSlider):
                live_widgets.add(widget)
        return live_widgets

    @staticmethod
    def _is_descendant_of(widget, ancestors):
        parent = widget.parentWidget()
        while parent is not None:
            if parent in ancestors:
                return True
            parent = parent.parentWidget()
        return False

    @staticmethod
    def _widget_and_descendants(widget):
        yield widget
        for child in widget.findChildren(QW.QWidget):
            yield child

    def _restore_editor_children(self, widget, prior_states):
        if isinstance(widget, QW.QAbstractSpinBox):
            editor = widget.findChild(QW.QLineEdit)
            if editor is not None:
                editor.setEnabled(prior_states.get(editor, True))
        elif isinstance(widget, QW.QComboBox):
            editor = widget.lineEdit()
            if editor is not None:
                editor.setEnabled(prior_states.get(editor, True))

    def set_acquisition_controls_locked(self, locked: bool):
        if locked:
            self._acquisition_lock_depth += 1
            if self._acquisition_lock_depth != 1:
                return
            live_widgets = self._live_acquisition_widgets()
            self._locked_widget_states = {
                widget: widget.isEnabled() for widget in self.findChildren(QW.QWidget)
            }
            for widget in self._managed_acquisition_widgets():
                if widget not in live_widgets and not self._is_descendant_of(widget, live_widgets):
                    for controlled_widget in self._widget_and_descendants(widget):
                        controlled_widget.setEnabled(False)
            return

        if self._acquisition_lock_depth == 0:
            return
        self._acquisition_lock_depth -= 1
        if self._acquisition_lock_depth != 0:
            return
        prior_states = self._locked_widget_states
        self._locked_widget_states = {}
        for widget, was_enabled in prior_states.items():
            try:
                widget.setEnabled(was_enabled)
            except RuntimeError:
                continue
        for widget, was_enabled in prior_states.items():
            if not was_enabled:
                continue
            try:
                self._restore_editor_children(widget, prior_states)
            except RuntimeError:
                continue

    def _on_acquisition_controls_locked(self, locked: bool):
        self.set_acquisition_controls_locked(locked)

    def enqueue_weaver_action(self, action):
        self.set_acquisition_controls_locked(True)
        WeaverQueue.put(action)

    def _on_bline_ready(self, payload: dict):
        self._last_display_payloads["bline"] = payload
        if not self._fps_ok("bline"):
            return
        render_bline_ready(self.ui, payload)

    def _on_cscan_ready(self, payload: dict):
        self._last_display_payloads["cscan"] = payload
        if not self._fps_ok("cscan"):
            return
        render_cscan_ready(self.ui, payload)

    def _on_aodo_waveform_ready(self, payload: dict):
        render_aodo_waveform_ready(self.ui, payload)
            
    def Stop_allThreads(self):
        self.ui.RunButton.setChecked(False)
        self.ui.RunButton.setText('Go')
        self.ui.PauseButton.setChecked(False)
        self.ui.PauseButton.setText('Pause')

        # Ask hardware tasks to stop before their worker thread receives exit.
        AODOQueue.put(AODOActionField('tryStopTask'))
        AODOQueue.put(AODOActionField('CloseTask'))

        exit_element = EXITField()
        WeaverQueue.put(exit_element)
        AODOQueue.put(exit_element)
        DnSQueue.put(exit_element)
        GPUQueue.put(exit_element)
        DQueue.put(exit_element)

    def _wait_for_threads_to_finish(self, timeout_ms=5000):
        deadline = time.time() + timeout_ms / 1000.0
        threads = [
            ("Weaver", self.Weaver_thread),
            ("AODO", self.AODO_thread),
            ("DnS", self.DnS_thread),
            ("GPU", self.GPU_thread),
            ("Camera", self.D_thread),
        ]
        unfinished = []
        for name, thread in threads:
            remaining_ms = max(0, int((deadline - time.time()) * 1000))
            if not thread.wait(remaining_ms):
                unfinished.append(name)
        return unfinished
        
    def run_task(self):
        acq_mode = self.ui.ACQMode.currentText()
        if acq_mode == "ContinueBline":
            acq_mode = AcqTypes.CONTINUOUS_BLINE
        if acq_mode in CONTINUOUS_ACQ_MODES + LIVE_ONLY_MODES:
            if self.ui.RunButton.isChecked():
                self.ui.RunButton.setText('Stop')
                an_action = WeaverActionField(acq_mode, acq_mode=acq_mode)
                self.enqueue_weaver_action(an_action)
            else:
                self.Stop_task()
        elif acq_mode in FINITE_ACQ_MODES:
            if self.ui.RunButton.isChecked():
                self.ui.RunButton.setText('Stop')
                # self.ui.RunButton.setEnabled(False)
                # self.ui.PauseButton.setEnabled(False)
                an_action = WeaverActionField(acq_mode, acq_mode=acq_mode)
                self.enqueue_weaver_action(an_action)
        
    def InitStages(self):
        an_action = AODOActionField('Init')
        AODOQueue.put(an_action)
        self._wait_stageback("stage init")
        
        
    def Uninit(self):
        an_action = AODOActionField('Uninit')
        AODOQueue.put(an_action)
        self._wait_stageback("stage uninit")
        
    def Xmove2(self):
        an_action = AODOActionField('Xmove2')
        AODOQueue.put(an_action)
        self._wait_stageback("X move", timeout=self.stage_move_timeout('X'))
        
    def Ymove2(self):
        an_action = AODOActionField('Ymove2')
        AODOQueue.put(an_action)
        self._wait_stageback("Y move", timeout=self.stage_move_timeout('Y'))
        
    def Zmove2(self):
        an_action = AODOActionField('Zmove2')
        AODOQueue.put(an_action)
        self._wait_stageback("Z move", timeout=self.stage_move_timeout('Z'))
        
    def XUP(self):
        an_action = AODOActionField('XUP')
        AODOQueue.put(an_action)
        self._wait_stageback("X step up")
    def YUP(self):
        an_action = AODOActionField('YUP')
        AODOQueue.put(an_action)
        self._wait_stageback("Y step up")
    def ZUP(self):
        an_action = AODOActionField('ZUP')
        AODOQueue.put(an_action)
        self._wait_stageback("Z step up")
        
    def XDOWN(self):
        an_action = AODOActionField('XDOWN')
        AODOQueue.put(an_action)
        self._wait_stageback("X step down")
    def YDOWN(self):
        an_action = AODOActionField('YDOWN')
        AODOQueue.put(an_action)
        self._wait_stageback("Y step down")
    def ZDOWN(self):
        an_action = AODOActionField('ZDOWN')
        AODOQueue.put(an_action)
        self._wait_stageback("Z step down")
        
    def XHome(self):
        an_action = AODOActionField('XHome')
        AODOQueue.put(an_action)
        self._wait_stageback("X home")
    def YHome(self):
        an_action = AODOActionField('YHome')
        AODOQueue.put(an_action)
        self._wait_stageback("Y home")
    def ZHome(self):
        an_action = AODOActionField('ZHome')
        AODOQueue.put(an_action)
        self._wait_stageback("Z home")
        
    def SetXSpeed(self):
        an_action = AODOActionField('XSpeed')
        AODOQueue.put(an_action)
        
    def SetYSpeed(self):
        an_action = AODOActionField('YSpeed')
        AODOQueue.put(an_action)
        
    def SetZSpeed(self):
        an_action = AODOActionField('ZSpeed')
        AODOQueue.put(an_action)
        
    def SetXAcc(self):
        an_action = AODOActionField('XAcc')
        AODOQueue.put(an_action)
        
    def SetYAcc(self):
        an_action = AODOActionField('YAcc')
        AODOQueue.put(an_action)
        
    def SetZAcc(self):
        an_action = AODOActionField('ZAcc')
        AODOQueue.put(an_action)
        
    def CenterGalvo(self):
        an_action = AODOActionField('centergalvo')
        AODOQueue.put(an_action)
        
    def Pause_task(self):
        if self.ui.PauseButton.isChecked():
            self.ui.PauseButton.setText('Resume')
            self.ui.statusbar.showMessage('acquisition paused...')
            print('acquisition paused...')
        else:
            self.ui.PauseButton.setText('Pause')
            self.ui.statusbar.showMessage('acquisition resumed...')
            print('acquisition resumed...')
      
    def Stop_task(self):
        # RunButton state already communicates the stop request; keep routine
        # stop handling quiet so it does not overwrite meaningful errors.
        pass
        
    def update_Dispersion(self):
        an_action = GPUActionField(GPUActions.UPDATE_DISPERSION)
        GPUQueue.put(an_action)
        # self.update_background()
        
    def update_background(self):
        an_action = GPUActionField(GPUActions.UPDATE_BACKGROUND)
        GPUQueue.put(an_action)
        
    def Update_contrast(self):
        acq_mode = self.ui.ACQMode.currentText()

        if acq_mode in ["FiniteBscan", "ContinuousBline", "Xfrequency"]:
            payload = self._last_display_payloads.get("bline")
            if payload is not None:
                render_bline_ready(self.ui, payload)
        elif acq_mode in ["FiniteCscan", "ContinuousCscan"]:
            payload = self._last_display_payloads.get("cscan")
            if payload is not None:
                render_cscan_ready(self.ui, payload, follow_latest_bline=False)

    # def redo_dispersion_compensation(self):
    #     an_action = WeaverActionField('dispersion_compensation')
    #     WeaverQueue.put(an_action)
        
    def redo_background(self):
        an_action = WeaverActionField(
            WeaverActions.GET_BACKGROUND,
            acq_mode=AcqTypes.FINITE_BSCAN,
        )
        self.enqueue_weaver_action(an_action)
        
    def redo_surface(self):
        an_action = WeaverActionField(
            WeaverActions.GET_SURFACE,
            acq_mode=AcqTypes.FINITE_BSCAN,
        )
        self.enqueue_weaver_action(an_action)
        
    # def update_intDk(self):
    #     self.ui.intDk.setValue(self.ui.intDkSlider.value()/100)
    #     an_action = GPUActionField('update_intDk')
    #     GPUQueue.put(an_action)
        
    # def UninitBoard(self):
    #     an_action = DActionField('UninitBoard')
    #     DQueue.put(an_action)
    
    def TestButton1Func(self):
        self.ui.statusbar.showMessage("Test button is not used in ASOCT.")
        
    def TestButton2Func(self):
        self.ui.statusbar.showMessage("Test button is not used in ASOCT.")
    
    def TestButton3Func(self):
        self.ui.statusbar.showMessage("Test button is not used in ASOCT.")
        
    def closeEvent(self, event):
        print('Exiting all threads')
        self.ui.statusbar.showMessage('Closing: stopping acquisition and worker threads...')
        print('Closing: stopping acquisition and worker threads...')
        self.SaveSettings()
        self.Stop_allThreads()
        unfinished = self._wait_for_threads_to_finish(timeout_ms=5000)
        if unfinished:
            message = "Close delayed: waiting for " + ", ".join(unfinished) + " thread(s)."
            print(message)
            self.ui.statusbar.showMessage(message)
            event.ignore()
            return
        event.accept()

                

if __name__ == '__main__':
    app = QApplication(sys.argv)
    example = GUI()
    example.show()
    sys.exit(app.exec_())

    








