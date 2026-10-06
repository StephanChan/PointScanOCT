# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 16:35:04 2023

@author: admin
"""

from GUI import Ui_MainWindow
import os
from PyQt5 import QtWidgets as QW
from PyQt5.QtWidgets import  QMainWindow, QFileDialog, QWidget, QVBoxLayout
from Dialogs import  StageDialog
import PyQt5.QtCore as qc
import numpy as np
from ActionFields import *
from Generaic_functions import *
from HardwareSpecs import (
    AXIAL_PIXEL_SIZE_UM,
    AXIAL_PIXEL_SIZE_UM_IN_WATER,
    camera_step_size_um,
    get_camera_spec,
    get_laser_spec,
    get_objective_spec,
)
from ImageViewer import PanZoomImageView
from CameraUi import (
    CAMERA_AK_ALIASES,
    effective_camera_sample_count,
    set_widget_value,
    x_pixel_count,
    x_position_count,
)
import traceback
# try:
#     from traits.api import HasTraits, Instance, on_trait_change
#     from traitsui.api import View, Item
#     from mayavi.core.ui.api import MayaviScene, MlabSceneModel, \
#             SceneEditor
#     from mayavi import mlab
#     print('using maya for 3D visulizaiton')
#     maya_installed = True
# except:
#     print('maya import failed, no 3D visulization')
#     maya_installed = False


# if maya_installed:
#     class Visualization(HasTraits):
#         scene = Instance(MlabSceneModel, ())
#         def __init__(self, data):
#             HasTraits.__init__(self)
#             self.data = data

#         def update_contrast(self, low, high):
#             # pass
#             # calculate data according to low and high
#             # self.plot.mlab_source.scalars = data-low+1000-high
#             M=np.max(self.plot.mlab_source.scalars)
#             self.plot.current_range=(low,M*high/1000)
#             # # print(low, high)
#             # print(self.plot.current_range)

#         def update_data(self, data):
#             self.plot.mlab_source.scalars = data
#             # print(data.shape)
#             # print(self.plot.current_range)
#             # print(np.max(self.plot.mlab_source.scalars))
#             # print(data[1,1,:])
#             M=np.max(data)
#             self.plot.current_range=(0, M*0.2)
#             # print(self.plot.current_range)

#         @on_trait_change('scene.activated')
#         def inital_plot(self):
#             self.plot = mlab.pipeline.volume(mlab.pipeline.scalar_field(self.data))
#             self.scene.background=(0,0,0)
#             # print(self.plot.current_range)
#             # self.plot.lut_manager.lut_mode = 'grey'
#             # a=self.plot.lut_manager
#             # print(self.plot.mlab_source.all_trait_names())
#             # print(self.plot.all_trait_names())

#         # the layout of the dialog screated
#         view = View(Item('scene', editor=SceneEditor(scene_class=MayaviScene),
#                           AlinesPerBline=250, width=300, show_label=False),
#                     resizable=True # We need this to resize with the parent widget
#                     )

#     # The QWidget containing the visualization, this is pure PyQt4 code.
#     class MayaviQWidget(QWidget):
#         def __init__(self, data = None):
#             super().__init__()

#             data = np.random.random((200,1000,300))
#             self.visualization = Visualization(data)
#             layout = QVBoxLayout(self)
#             # layout.setContentsMargins(0,0,0,0)
#             # layout.setSpacing(0)
#             # The edit_traits call will generate the widget to embed.
#             self.ui = self.visualization.edit_traits(parent=self,
#                                                       kind='subpanel').control
#             layout.addWidget(self.ui)
#             # self.setLayout(layout)
#             # self.ui.setParent(self)

class MainWindow(QMainWindow):
    # 相机 .vlcf 选择：UI 线程只发信号/置标志，真正的设备设置由相机线程完成。
    ak_vlcf_changed = qc.pyqtSignal(str)
    # 曝光 / 行频 / 增益：用户改动后立即下发相机（.vlcf/.ccf 只作为初始值）。
    ak_runtime_changed = qc.pyqtSignal()
    # 标尺轴向尺度（水中/空气）切换：由 OCT_MT 用缓存的最后一次 payload 重画显示。
    ruler_scale_changed = qc.pyqtSignal()
    AK_VLCF_DEFAULT = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "AKCameraConfig", "12bit4tap.vlcf"
    )
    # 与 .vlcf 重复、需要锁定的输入控件（名字不存在时自动跳过）
    AK_VLCF_LOCKED_WIDGETS = (
        "PixelFormat_AK",
        "Tap",
        "Taps",
        "offsetW_AK",
        "NSamples_AK",
        "TriggerON_AK",
        "TriggerSelector_AK",
        "TriggerSource_AK",
    )
    # 保持可编辑、且改动后立即下发相机的参数（曝光 / 行频 / 增益）
    AK_RUNTIME_WIDGETS = (
        "Exposure_AK",
        "LineRate_AK",
        "AnalogGain_AK",
        "PGAGain_AK",
        "PgaGain_AK",
        "AGain_AK",
        "DGain_AK",
    )
    # 陷波滤波控件（tab_2）：X 在 GPU 侧逐 B-line，Y 在 DnS 侧整幅 C-scan
    NOTCH_WIDGETS = ("XNotchFilter", "XNotchBandwidth", "YNotchFilter", "YNotchBandwidth")

    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.Objective.clear()
        self.ui.Objective.addItems([
            "0.66X",
            "4X",
            "5X",
            "10X",
            "20X",
        ])
        self.ui.ACQMode.clear()
        self.ui.ACQMode.addItems([
            "FiniteBscan",
            "Xfrequency",

            "ContinuousBline",
            "FiniteCscan",
            "ContinuousCscan",
        ])
        if hasattr(self.ui, "FFTresults"):
            self.ui.FFTresults.clear()
            self.ui.FFTresults.addItem("AMP")
        self.LoadSettings("config.ini")
        if hasattr(self.ui, "Camera"):
            for camera_name in ("AiKe", "AK", "IKap", "PL4KCL-200KT"):
                if self.ui.Camera.findText(camera_name) >= 0:
                    self.ui.Camera.setCurrentText(camera_name)
                    break
        if self.ui.ACQMode.currentText() not in [
            "Xfrequency",

            "FiniteBscan",
            "ContinuousBline",
            "FiniteCscan",
            "ContinuousCscan",
        ]:
            self.ui.ACQMode.setCurrentText("ContinuousBline")
        if hasattr(self.ui, "FFTresults"):
            self.ui.FFTresults.setCurrentText("AMP")
        self.setStageMinMax()
        self.Calculate_CameraWidth_settings()
        self.Calculate_Galvo_settings()
        # self.Update_laser()
        self._on_galvo_waveform_input_changed()
        self.update_depth_bar_limits()
        # self.ui.DepthEndBar.setValue(0)
        self.Adjust_Bline_Height()
        self.connectActions()
        self.setup_ak_vlcf_row()
        self.setup_ak_runtime_params()
        self.setup_notch_ui()
        self.setup_ruler_views()
        self.setup_ruler_ui()

    # -- 物理标尺（移植自 LineScanOCT 控制程序的 Rulers.py / ImageViewer.py）---- #
    RULER_DISPLAY_NAMES = ("XZplane", "XYplane")

    def setup_ruler_views(self):
        """把 XZplane / XYplane 的 QLabel 换成带物理标尺的交互查看器。

        与本项目 notch 控件同样的做法：GUI.ui / GUI.py 不改，运行时替换并把同名
        句柄挂回 self.ui，因此现有渲染代码、DnS/GPU 线程都不受影响；渲染侧用
        hasattr(label, "set_data") 判断是否支持标尺，不支持时自动退回 setPixmap。
        """
        self.ui.axial_pixel_size_um = self.ruler_axial_pixel_size_um()
        for display_name in self.RULER_DISPLAY_NAMES:
            label = getattr(self.ui, display_name, None)
            if label is None or isinstance(label, PanZoomImageView):
                continue
            parent = label.parentWidget()
            layout = parent.layout() if parent is not None else None
            if parent is None or layout is None:
                print("ruler viewer: %s has no parent layout; kept as QLabel" % display_name)
                continue
            minimum = label.minimumSize()
            viewer = PanZoomImageView(parent)
            if minimum.width() > 0 or minimum.height() > 0:
                viewer.setMinimumSize(minimum)
            layout.replaceWidget(label, viewer)
            label.hide()
            label.deleteLater()
            setattr(self.ui, display_name, viewer)

    def ruler_axial_pixel_size_um(self):
        """当前轴向标尺尺度（µm/px）：勾选=水中（默认），不勾=空气等效。"""
        checkbox = getattr(self.ui, "RulerWaterCheck", None)
        if checkbox is not None and hasattr(checkbox, "isChecked"):
            try:
                return (
                    AXIAL_PIXEL_SIZE_UM_IN_WATER
                    if bool(checkbox.isChecked())
                    else AXIAL_PIXEL_SIZE_UM
                )
            except Exception:
                pass
        return AXIAL_PIXEL_SIZE_UM_IN_WATER

    def setup_ruler_ui(self):
        """在 tab_2（图像优化）里补一个"标尺轴向尺度"分组：水中/空气 + 当前 µm/px。"""
        water_check = getattr(self.ui, "RulerWaterCheck", None)
        if water_check is None:
            tab = getattr(self.ui, "tab_2", None)
            if tab is None:
                print("tab_2 (图像优化) not found; ruler controls skipped.")
                return
            box = QW.QGroupBox("标尺轴向尺度 (Ruler)", tab)
            grid = QW.QGridLayout(box)
            water_check = QW.QCheckBox("水中 (n=1.33)", box)
            water_check.setChecked(True)
            water_check.setObjectName("RulerWaterCheck")
            grid.addWidget(water_check, 0, 0)
            pitch_label = QW.QLabel("", box)
            pitch_label.setObjectName("RulerAxialPitchLabel")
            grid.addWidget(pitch_label, 0, 1)
            layout = tab.layout()
            if layout is None:
                layout = QW.QGridLayout(tab)
            layout.addWidget(box, 1, 4, 1, 1)
            self.ui.RulerBox = box
            self.ui.RulerWaterCheck = water_check
            self.ui.RulerAxialPitchLabel = pitch_label
        # GUI.ui 未重新生成时也能从 config.ini 恢复（默认水中）
        try:
            settings = qc.QSettings("config.ini", qc.QSettings.IniFormat)
            stored = settings.value("RulerWaterCheck")
            if stored is not None:
                water_check.setChecked(str(stored).lower() != "false")
        except Exception:
            pass
        water_check.toggled.connect(self.on_ruler_axial_changed)
        self.ui.axial_pixel_size_um = self.ruler_axial_pixel_size_um()
        self.update_ruler_axial_pitch_label()

    def on_ruler_axial_changed(self, _checked=False):
        """切换水中/空气：更新轴向尺度、刷新标尺显示并存 config.ini。"""
        self.ui.axial_pixel_size_um = self.ruler_axial_pixel_size_um()
        self.update_ruler_axial_pitch_label()
        try:
            settings = qc.QSettings("config.ini", qc.QSettings.IniFormat)
            settings.setValue("RulerWaterCheck", bool(self.ruler_axial_pixel_size_um() == AXIAL_PIXEL_SIZE_UM_IN_WATER))
        except Exception:
            pass
        self.ruler_scale_changed.emit()

    def update_ruler_axial_pitch_label(self):
        label = getattr(self.ui, "RulerAxialPitchLabel", None)
        if label is None:
            return
        label.setText("%.2f um/px" % self.ruler_axial_pixel_size_um())

    def setStageMinMax(self):
        self.ui.XPosition.setMinimum(self.ui.Xmin.value())
        self.ui.XPosition.setMaximum(self.ui.Xmax.value())

        self.ui.YPosition.setMinimum(self.ui.Ymin.value())
        self.ui.YPosition.setMaximum(self.ui.Ymax.value())

        self.ui.ZPosition.setMinimum(self.ui.Zmin.value())
        self.ui.ZPosition.setMaximum(self.ui.Zmax.value())

        self.ui.Xcurrent.setMinimum(self.ui.Xmin.value())
        self.ui.Xcurrent.setMaximum(self.ui.Xmax.value())

        self.ui.Ycurrent.setMinimum(self.ui.Ymin.value())
        self.ui.Ycurrent.setMaximum(self.ui.Ymax.value())

        self.ui.Zcurrent.setMinimum(self.ui.Zmin.value())
        self.ui.Zcurrent.setMaximum(self.ui.Zmax.value())



    # def addMaya(self):
    #     if (maya_installed and self.use_maya):
    #         self.ui.mayavi_widget = MayaviQWidget()
    #         self.ui.mayavi_widget.setMinimumSize(qc.QSize(100, 100))
    #         self.ui.mayavi_widget.setMaximumSize(qc.QSize(1000, 1000))
    #         self.ui.mayavi_widget.setObjectName("XYZView")

    #         # self.ui.verticalLayout_2.removeWidget(self.ui.tmp_label)
    #         # self.ui.verticalLayout_2.addWidget(self.ui.mayavi_widget)

    def LoadSettings(self, config_filepath):
        settings = qc.QSettings(config_filepath, qc.QSettings.IniFormat)
        for ii in dir(self.ui):
            widget = getattr(self.ui, ii)
            # print(ii, type(self.ui.__getattribute__(ii)) )
            if ii == 'ACQMode':
                pass
            elif isinstance(widget, QW.QComboBox):
                try:
                    widget.setCurrentText(settings.value(ii))
                except:
                    print(ii, ' setting missing, using default...')
            elif isinstance(widget, QW.QDoubleSpinBox):
                try:
                    widget.setValue(np.float32(settings.value(ii)))
                except:
                    print(ii, ' setting missing, using default...')
            elif isinstance(widget, QW.QSpinBox):
                try:
                    widget.setValue(np.int16(settings.value(ii)))
                except:
                    print(ii, ' setting missing, using default...')
            elif isinstance(widget, QW.QTextEdit):
                try:
                    widget.setText(settings.value(ii))
                except:
                    print(ii, ' setting missing, using default...')
            elif isinstance(widget, QW.QLineEdit):
                try:
                    widget.setText(settings.value(ii))
                except:
                    print(ii, ' setting missing, using default...')
            elif isinstance(widget, QW.QSlider):
                # print(ii, int(settings.value(ii)))
                try:
                    widget.setValue(int(settings.value(ii)))
                except:
                    print(ii, ' setting missing, using default...')
            elif isinstance(widget, QW.QScrollBar):
                try:
                    widget.setValue(int(settings.value(ii)))
                except:
                    print(ii, ' setting missing, using default...')
            elif isinstance(widget, (QW.QPushButton, QW.QCheckBox)):
                try:
                    value = settings.value(ii)
                    widget.setChecked(str(value).lower() == 'true')
                except:
                    print(ii, ' setting missing, using default...')


    def Calculate_CameraWidth_settings(self):
        # In spot-scan mode, X sampling is set by the user-defined galvo
        # step size instead of the camera pixel pitch.
        camera = get_camera_spec(self.ui.Camera.currentText())
        if camera is None:
            status = 'camera not calibrated, abort FOV calculation'
            self.ui.statusbar.showMessage(status)
            return None, status
        if hasattr(self.ui, "NSamples_AK") and self.ui.Camera.currentText() in CAMERA_AK_ALIASES:
            self.ui.NSamples_AK.setMaximum(int(camera.spectral_samples_px))
            if self.ui.NSamples_AK.value() > int(camera.spectral_samples_px):
                self.ui.NSamples_AK.setValue(int(camera.spectral_samples_px))
        x_step_size = max(float(self.ui.XStepSize.value()), 1e-9)
        aline_avg = max(1, int(self.ui.AlineAVG.value()))

        x_positions = int(round(self.ui.XLength.value() * 1000.0 / x_step_size))
        x_pixels = max(1, x_positions * aline_avg)
        x_widget = getattr(self.ui, "XPixels", getattr(self.ui, "Xpixels", None))
        if x_widget is not None and hasattr(x_widget, "setMaximum") and x_pixels > x_widget.maximum():
            x_widget.setMaximum(x_pixels)
        set_widget_value(self.ui, ("XPixels", "Xpixels"), x_pixels)
        objective = get_objective_spec(self.ui.Objective.currentText())
        if objective is not None and hasattr(self.ui, "XGalvoBias"):
            # 偏置电压 = 偏移[mm]/ratio；K=1 deg/V 后 ratio 变小约 5 倍，电压范围也要放大，
            # 否则偏移量稍大就会被 ±2 V 的控件范围截断（Y 已在 Calculate_Galvo_settings 里放宽）。
            self.ui.XGalvoBias.setMinimum(-10)
            self.ui.XGalvoBias.setMaximum(10)
            self.ui.XGalvoBias.setValue(self.ui.Xoffsetlength.value()/objective.angle_to_mm_ratio)

    def Calculate_Galvo_settings(self):
        # select objective magnification
        objective = get_objective_spec(self.ui.Objective.currentText())
        if objective is None:
            status = 'objective not calibrated, abort generating Galvo waveform'
            self.ui.statusbar.showMessage(status)
            return None, status
        angle2mmratio = objective.angle_to_mm_ratio
        self.ui.YLength.setMaximum(float(objective.max_y_fov_mm))

        # Calculate AlinesPerBline pixel numbers based on user set FOV size
        Ypixels = np.uint16(np.round(self.ui.YLength.value()*1000/self.ui.YStepSize.value()))
        self.ui.Ypixels.setValue(Ypixels)
        self.ui.YGalvoBias.setMinimum(-10)
        self.ui.YGalvoBias.setMaximum(10)
        self.ui.YGalvoBias.setValue(self.ui.Yoffsetlength.value()/angle2mmratio)

    def Adjust_Bline_Height(self):
        try:
            effective_camera_sample_count(self.ui)
        except ValueError as error:
            self.ui.statusbar.showMessage(str(error))
            return
        depth_start = min(self.ui.DepthStartBar.value(), self.ui.DepthEndBar.value())
        depth_end = max(self.ui.DepthStartBar.value(), self.ui.DepthEndBar.value())
        self.ui.DepthStart.setValue(depth_start)
        self.ui.DepthRange.setValue(max(depth_end - depth_start, 1))
        self.update_z_depth_bar_range()

    def update_z_depth_bar_range(self):
        depth_range = max(1, int(self.ui.DepthRange.value()))
        self.ui.ZDepthBar.setMaximum(depth_range - 1)
        if self.ui.ZDepthBar.value() >= depth_range:
            self.ui.ZDepthBar.setValue(depth_range - 1)

    def update_depth_bar_limits(self):
        try:
            samples = effective_camera_sample_count(self.ui)
        except ValueError as error:
            self.ui.statusbar.showMessage(str(error))
            return
        depth_limit = max(1, int(samples))
        self.ui.DepthStartBar.setMinimum(0)
        self.ui.DepthEndBar.setMinimum(0)
        self.ui.DepthStartBar.setMaximum(max(0, depth_limit - 1))
        self.ui.DepthEndBar.setMaximum(depth_limit)
        if hasattr(self.ui, "DepthStart"):
            self.ui.DepthStart.setMaximum(max(0, depth_limit - 1))
        if hasattr(self.ui, "DepthRange"):
            self.ui.DepthRange.setMaximum(depth_limit)
        self.ui.DepthStartBar.setInvertedAppearance(False)
        self.ui.DepthEndBar.setInvertedAppearance(False)
        self.ui.DepthStartBar.setInvertedControls(False)
        self.ui.DepthEndBar.setInvertedControls(False)
        if self.ui.DepthStartBar.value() > max(0, depth_limit - 1):
            self.ui.DepthStartBar.setValue(max(0, depth_limit - 1))
        if self.ui.DepthEndBar.value() > depth_limit:
            self.ui.DepthEndBar.setValue(depth_limit)
        self.Adjust_Bline_Height()
        self.update_z_depth_bar_range()

    def SaveSettings(self):
        settings = qc.QSettings("config.ini", qc.QSettings.IniFormat)
        for ii in dir(self.ui):
            widget = getattr(self.ui, ii)
            # print(ii, type(self.ui.__getattribute__(ii)) )
            if isinstance(widget, QW.QComboBox):
                settings.setValue(ii, widget.currentText())
            elif isinstance(widget, QW.QDoubleSpinBox):
                settings.setValue(ii, widget.value())
            elif isinstance(widget, QW.QSpinBox):
                settings.setValue(ii, widget.value())
            elif isinstance(widget, QW.QTextEdit):
                settings.setValue(ii, widget.toPlainText())
            elif isinstance(widget, QW.QLineEdit):
                settings.setValue(ii, widget.text())
            elif isinstance(widget, (QW.QPushButton, QW.QCheckBox)):
                settings.setValue(ii, widget.isChecked())
            elif isinstance(widget, QW.QSlider):
                settings.setValue(ii, widget.value())
            elif isinstance(widget, QW.QScrollBar):
                settings.setValue(ii, widget.value())
                # print(ii,self.ui.__getattribute__(ii).value())

    def connectActions(self):
        self.ui.Objective.currentTextChanged.connect(self.Calculate_CameraWidth_settings)
        self.ui.Objective.currentTextChanged.connect(self.Calculate_Galvo_settings)
        self.ui.Camera.currentTextChanged.connect(self.Calculate_CameraWidth_settings)
        self.ui.Camera.currentTextChanged.connect(self.update_depth_bar_limits)

        self.ui.XLength.valueChanged.connect(self.Calculate_CameraWidth_settings)
        self.ui.XStepSize.valueChanged.connect(self.Calculate_CameraWidth_settings)
        self.ui.AlineAVG.valueChanged.connect(self.Calculate_CameraWidth_settings)
        self.ui.Xoffsetlength.valueChanged.connect(self.Calculate_CameraWidth_settings)
        self.ui.YLength.valueChanged.connect(self.Calculate_Galvo_settings)
        self.ui.Yoffsetlength.valueChanged.connect(self.Calculate_Galvo_settings)
        self.ui.YStepSize.valueChanged.connect(self.Calculate_Galvo_settings)
        self.ui.BlineAVG.valueChanged.connect(self.Calculate_Galvo_settings)
        self.ui.DepthStartBar.valueChanged.connect(self.Adjust_Bline_Height)
        self.ui.DepthEndBar.valueChanged.connect(self.Adjust_Bline_Height)
        self.ui.DepthRange.valueChanged.connect(self.update_z_depth_bar_range)
        self.ui.NSamples_DH.valueChanged.connect(self.update_depth_bar_limits)
        self.ui.NSamples_PF.valueChanged.connect(self.update_depth_bar_limits)
        if hasattr(self.ui, "NSamples_AK"):
            self.ui.NSamples_AK.valueChanged.connect(self.update_depth_bar_limits)
        self.ui.SpectralDS_DH.valueChanged.connect(self.update_depth_bar_limits)
        self.ui.SpectralDS_PF.valueChanged.connect(self.update_depth_bar_limits)
        if hasattr(self.ui, "SpectralDS_AK"):
            self.ui.SpectralDS_AK.valueChanged.connect(self.update_depth_bar_limits)

        # self.ui.Objective.currentTextChanged.connect(self.update_galvoXwaveform)
        # self.ui.PreClock.valueChanged.connect(self.update_galvoXwaveform)
        # 波形预览：任一相关参数变化 → 去抖 250ms 后重画 X/Y/DO 波形
        for widget_name in (
            "ACQMode",
            "Objective",
            "FlyBack",
            "XLength",
            "XStepSize",
            "XGalvoBias",
            "Ypixels",
            "YStepSize",
            "YGalvoBias",
            "AlineAVG",
            "BlineAVG",
        ):
            widget = getattr(self.ui, widget_name, None)
            if widget is None:
                continue
            if hasattr(widget, "currentTextChanged"):
                widget.currentTextChanged.connect(self._on_galvo_waveform_input_changed)
            elif hasattr(widget, "valueChanged"):
                widget.valueChanged.connect(self._on_galvo_waveform_input_changed)
        # self.ui.Ysteps.valueChanged.connect(self.update_galvoYwaveform)
        # self.ui.YStepSize.valueChanged.connect(self.update_galvoYwaveform)
        # self.ui.BlineAVG.valueChanged.connect(self.update_galvoYwaveform)
        # self.ui.YBias.valueChanged.connect(self.update_galvoYwaveform)

        # self.ui.ImageZDepth.valueChanged.connect(self.Calculate_ImageDepth)
        # self.ui.ImageZnumber.valueChanged.connect(self.Calculate_ImageDepth)

        # self.ui.SliceZStart.valueChanged.connect(self.Calculate_SliceDepth)
        # self.ui.SliceZDepth.valueChanged.connect(self.Calculate_SliceDepth)
        # self.ui.SliceZnumber.valueChanged.connect(self.Calculate_SliceDepth)
        # change brigtness and contrast
        # self.ui.ACQMode.currentIndexChanged.connect(self.Adjust_contrast)
        # self.ui.FFTDevice.currentIndexChanged.connect(self.Update_scale)

        # update laser model
        # self.ui.Laser.currentIndexChanged.connect(self.Update_laser)
        self.ui.Save.clicked.connect(self.chooseDir)
        self.ui.LoadInD.clicked.connect(self.chooseInD)
        self.ui.LoadBG.clicked.connect(self.chooseBackground)
        self.ui.ConfigButton.clicked.connect(self.LoadConfig)

        self.ui.Xmax.valueChanged.connect(self.setStageMinMax)
        self.ui.Xmin.valueChanged.connect(self.setStageMinMax)
        self.ui.Ymin.valueChanged.connect(self.setStageMinMax)
        self.ui.Ymax.valueChanged.connect(self.setStageMinMax)
        self.ui.Zmin.valueChanged.connect(self.setStageMinMax)
        self.ui.Zmax.valueChanged.connect(self.setStageMinMax)

        self.ui.LoadSurface.clicked.connect(self.chooseSurfaceFile)
        # self.ui.LoadDarkField.clicked.connect(self.chooseDarkFieldFile)
        # self.ui.LoadFlatField.clicked.connect(self.chooseFlatFieldFile)

    def chooseSurfaceFile(self):
        fileName_choose, filetype = QFileDialog.getOpenFileName(self,
                                   "select surface file",
                                   self.ui.DIR.toPlainText(), # 
                                   "All Files (*);;Text Files (*.txt)")   # ?

        if fileName_choose == "":
           print("\n use default")
           return
        self.ui.Surf_DIR.setText(fileName_choose)

    def setup_ak_vlcf_row(self):
        """确保 AiKe 页有 .vlcf 选择行（GUI.ui 里已包含则直接复用），并完成信号连接与锁定。"""
        if getattr(self.ui, "VLCFPath_AK", None) is None:
            self._ak_create_vlcf_row()
        path_edit = getattr(self.ui, "VLCFPath_AK", None)
        if path_edit is None:
            print("AiKe .vlcf selector is unavailable; skipped.")
            return
        if getattr(self.ui, "_ak_vlcf_wired", False):
            return
        browse_button = getattr(self.ui, "BrowseVLCF_AK", None)
        reload_button = getattr(self.ui, "ReloadVLCF_AK", None)
        auto_box = getattr(self.ui, "AutoVLCF_AK", None)
        if not str(path_edit.text()).strip():
            path_edit.setText(self.AK_VLCF_DEFAULT)
        if browse_button is not None:
            browse_button.clicked.connect(self._ak_browse_vlcf)
        if reload_button is not None:
            reload_button.clicked.connect(self._ak_reload_vlcf)
        if auto_box is not None:
            auto_box.toggled.connect(self._ak_auto_vlcf_toggled)
        # 触发语义与 IKapCViewer 对齐：Selector 用相机的 SynchronizationMode 取值
        selector_combo = getattr(self.ui, "TriggerSelector_AK", None)
        if selector_combo is not None and hasattr(selector_combo, "itemText"):
            items = [selector_combo.itemText(index) for index in range(selector_combo.count())]
            if items != ["InternalFreeRun", "ExternalPulse"]:
                selector_combo.clear()
                selector_combo.addItems(["InternalFreeRun", "ExternalPulse"])
        setattr(self.ui, "_ak_vlcf_dirty", False)
        setattr(self.ui, "_ak_vlcf_wired", True)
        self.apply_ak_vlcf_ui_lock(path_edit.text())

    def setup_ak_runtime_params(self):
        """曝光/行频/增益保持可编辑，并在用户改动后（去抖 400ms）立即下发相机。"""
        if getattr(self.ui, "_ak_runtime_wired", False):
            return
        # 强制保持可编辑（防止界面上被误设为 disabled）
        for name in self.AK_RUNTIME_WIDGETS:
            widget = getattr(self.ui, name, None)
            if widget is not None and hasattr(widget, "setEnabled"):
                widget.setEnabled(True)
        self._ak_runtime_timer = qc.QTimer(self)
        self._ak_runtime_timer.setSingleShot(True)
        self._ak_runtime_timer.setInterval(400)
        self._ak_runtime_timer.timeout.connect(self.ak_runtime_changed.emit)
        for name in self.AK_RUNTIME_WIDGETS:
            widget = getattr(self.ui, name, None)
            if widget is None:
                continue
            if hasattr(widget, "valueChanged"):
                widget.valueChanged.connect(self._ak_runtime_param_changed)
            if hasattr(widget, "currentTextChanged"):
                widget.currentTextChanged.connect(self._ak_runtime_param_changed)
        setattr(self.ui, "_ak_runtime_wired", True)

    def _ak_runtime_param_changed(self, *_args):
        """曝光/行频/增益被改动：去抖后通知相机线程重新设置对应参数。"""
        setattr(self.ui, "_ak_runtime_dirty", True)
        timer = getattr(self, "_ak_runtime_timer", None)
        if timer is not None:
            timer.start()

    def setup_notch_ui(self):
        """确保 tab_2 有陷波滤波控件（GUI.ui 已包含则直接复用）；缺失时运行时补一组。"""
        if getattr(self.ui, "XNotchFilter", None) is not None:
            return
        tab = getattr(self.ui, "tab_2", None)
        if tab is None:
            print("tab_2 (图像优化) not found; notch controls skipped.")
            return
        box = QW.QGroupBox("陷波滤波 (Notch)", tab)
        grid = QW.QGridLayout(box)
        x_check = QW.QCheckBox("X陷波滤波 (X 振镜/横向)", box)
        y_check = QW.QCheckBox("Y陷波滤波 (C-scan 扫描向)", box)
        x_spin = QW.QDoubleSpinBox(box)
        y_spin = QW.QDoubleSpinBox(box)
        for spin in (x_spin, y_spin):
            spin.setDecimals(3)
            spin.setMinimum(0.0)
            spin.setMaximum(1.0)
            spin.setSingleStep(0.05)
            spin.setValue(0.25)
        grid.addWidget(x_check, 0, 0)
        grid.addWidget(QW.QLabel("X陷波带宽 (x Nyquist)", box), 0, 1)
        grid.addWidget(x_spin, 0, 2)
        grid.addWidget(y_check, 1, 0)
        grid.addWidget(QW.QLabel("Y陷波带宽 (x Nyquist)", box), 1, 1)
        grid.addWidget(y_spin, 1, 2)
        layout = tab.layout()
        if layout is None:
            layout = QW.QVBoxLayout(tab)
        layout.addWidget(box)
        self.ui.XNotchFilter = x_check
        self.ui.XNotchBandwidth = x_spin
        self.ui.YNotchFilter = y_check
        self.ui.YNotchBandwidth = y_spin
        # GUI.ui 未重新生成时，控件创建晚于 LoadSettings，这里补读一次 config.ini
        settings = qc.QSettings("config.ini", qc.QSettings.IniFormat)
        for name, widget in (
            ("XNotchFilter", x_check),
            ("YNotchFilter", y_check),
        ):
            try:
                widget.setChecked(str(settings.value(name)).lower() == "true")
            except Exception:
                pass
        for name, widget in (
            ("XNotchBandwidth", x_spin),
            ("YNotchBandwidth", y_spin),
        ):
            try:
                widget.setValue(np.float32(settings.value(name)))
            except Exception:
                pass

    def _ak_create_vlcf_row(self):
        """GUI.ui 尚未包含 VLCF 行时，运行时补一行（兼容旧界面）。"""
        tab = getattr(self.ui, "tab_5", None)
        if tab is None:
            print("AiKe tab not found; .vlcf selector skipped.")
            return
        row = QW.QWidget(tab)
        row_layout = QW.QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        caption = QW.QLabel("配置文件 VLCF", tab)
        path_edit = QW.QLineEdit(tab)
        path_edit.setReadOnly(True)
        path_edit.setMaximumWidth(260)
        path_edit.setText(self.AK_VLCF_DEFAULT)
        browse_button = QW.QPushButton("浏览", tab)
        reload_button = QW.QPushButton("重新加载", tab)
        auto_box = QW.QCheckBox("启动时自动加载默认 VLCF", tab)
        auto_box.setChecked(True)
        info_label = QW.QLabel("未加载", tab)
        info_label.setMaximumWidth(80)
        for widget in (caption, path_edit, browse_button, reload_button, auto_box, info_label):
            row_layout.addWidget(widget)
        row_layout.addStretch(1)
        layout = tab.layout()
        if layout is None:
            layout = QW.QVBoxLayout(tab)
        layout.addWidget(row)
        self.ui.VLCFPath_AK = path_edit
        self.ui.BrowseVLCF_AK = browse_button
        self.ui.ReloadVLCF_AK = reload_button
        self.ui.AutoVLCF_AK = auto_box
        self.ui.VLCFInfo_AK = info_label

    def _ak_browse_vlcf(self):
        """选择 .vlcf 文件并请求相机线程按它重新设置。"""
        start_dir = os.path.dirname(self.ui.VLCFPath_AK.text().strip() or self.AK_VLCF_DEFAULT)
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择 IKap VLCF 配置文件",
            start_dir,
            "IKap config (*.vlcf);;All files (*)",
        )
        if not path:
            return
        self.ui.VLCFPath_AK.setText(path)
        if getattr(self.ui, "AutoVLCF_AK", None) is not None:
            self.ui.AutoVLCF_AK.setChecked(False)
        self.apply_ak_vlcf_ui_lock(path)
        setattr(self.ui, "_ak_vlcf_dirty", True)
        self.ak_vlcf_changed.emit(path)

    def _ak_reload_vlcf(self):
        """强制按当前路径的 .vlcf 重新设置相机（相机线程会跳过加载缓存）。"""
        path = self.ui.VLCFPath_AK.text().strip()
        if not path:
            return
        setattr(self.ui, "_ak_vlcf_dirty", True)
        self.ak_vlcf_changed.emit(path)

    def _ak_auto_vlcf_toggled(self, checked):
        if checked:
            self.ui.VLCFPath_AK.setText(self.AK_VLCF_DEFAULT)
            self.apply_ak_vlcf_ui_lock(self.AK_VLCF_DEFAULT)
            setattr(self.ui, "_ak_vlcf_dirty", True)
            self.ak_vlcf_changed.emit(self.AK_VLCF_DEFAULT)

    def apply_ak_vlcf_ui_lock(self, vlcf_path=None):
        """锁定与 .vlcf 重复的 UI 输入，并把生效状态显示到只读标签（详情放 tooltip）。"""
        for name in self.AK_VLCF_LOCKED_WIDGETS:
            widget = getattr(self.ui, name, None)
            if widget is not None and hasattr(widget, "setEnabled"):
                widget.setEnabled(False)
        path = (vlcf_path or self.ui.VLCFPath_AK.text()).strip()
        try:
            from IKapCamera_test import read_vlcf_settings, vlcf_int
        except Exception as error:
            print("VLCF helper import failed:", error)
            return
        settings = read_vlcf_settings(path) if path else {}
        if not settings:
            self._ak_set_vlcf_status("未加载", "%s 不存在或没有有效字段" % (os.path.basename(path) or "no file"))
            return
        sync_mode = self._ak_read_ccf_value(
            os.path.splitext(path)[0] + ".ccf", "SynchronizationMode"
        )
        detail = "文件: %s\n位深: %s bit\ntap: %s\n宽度: %s\n偏移X: %s\n板卡行触发源: %s\n相机触发(.ccf): %s" % (
            os.path.basename(path),
            vlcf_int(settings, "PIXEL_DEPTH"),
            vlcf_int(settings, "TAP_NUMBER"),
            vlcf_int(settings, "IMAGE_WIDTH"),
            vlcf_int(settings, "IMAGE_OFFSET_X"),
            vlcf_int(settings, "INTEGRATION_TRIGGER_SOURCE"),
            sync_mode or "(无 .ccf)",
        )
        self._ak_set_vlcf_status("已加载", detail)
        if sync_mode:
            try:
                self.ui.TriggerSelector_AK.setCurrentText(sync_mode)
                self.ui.TriggerON_AK.setCurrentText(
                    "Off" if "internal" in sync_mode.lower() else "On"
                )
            except Exception:
                pass

    def _ak_set_vlcf_status(self, short_text, detail=""):
        """只读标签只显示短状态，完整信息放 tooltip（避免把界面拉长）。"""
        label = getattr(self.ui, "VLCFInfo_AK", None)
        if label is None:
            return
        try:
            label.setText(short_text)
            label.setToolTip(detail or short_text)
        except Exception:
            pass

    @staticmethod
    def _ak_read_ccf_value(ccf_path, key):
        """从相机 .ccf 里读一个字段值（只读文件，不碰设备）。"""
        try:
            with open(ccf_path, "r", encoding="latin-1") as handle:
                for line in handle:
                    if "=" not in line:
                        continue
                    name, _, value = line.partition("=")
                    if name.strip() == key:
                        return value.strip()
        except Exception:
            return None
        return None

    # def chooseDarkFieldFile(self):
    #     fileName_choose, filetype = QFileDialog.getOpenFileName(self,
    #                                "select dark field file",
    #                                os.getcwd(), # 
    #                                "All Files (*);;Text Files (*.txt)")   # ?

    #     if fileName_choose == "":
    #        print("\n use default")
    #        return
    #     self.ui.DarkField_DIR.setText(fileName_choose)

    # def chooseFlatFieldFile(self):
    #     fileName_choose, filetype = QFileDialog.getOpenFileName(self,
    #                                "select flat field file",
    #                                os.getcwd(), # 
    #                                "All Files (*);;Text Files (*.txt)")   # ?

    #     if fileName_choose == "":
    #        print("\n use default")
    #        return
    #     self.ui.FlatField_DIR.setText(fileName_choose)

    def chooseDir(self):
        if self.ui.Save.isChecked():

             dir_choose = QFileDialog.getExistingDirectory(self,
                                         "select saving directory",
                                         self.ui.DIR.toPlainText()) # 

             if dir_choose == "":
                 print("\n use default")
                 return
             self.ui.DIR.setText(dir_choose)

    def LoadConfig(self):
        fileName_choose, filetype = QFileDialog.getOpenFileName(self,
                                   "select config file",
                                   os.getcwd(), # 
                                   "All Files (*);;Text Files (*.txt)")   # ?

        if fileName_choose == "":
           print("\n use default")
           return

        try:
            self.LoadSettings(fileName_choose)
        except Exception as error:
            print('settings reload failed, using default settings')
            print(traceback.format_exc())

    def chooseInD(self):
         dir_choose = QFileDialog.getExistingDirectory(self,
                                     "select saving directory",
                                     self.ui.InD_DIR.text()) # 

         if dir_choose == "":
             print("\n use default")
             return
         self.ui.InD_DIR.setText(dir_choose)

    def chooseBackground(self):
        fileName_choose, filetype = QFileDialog.getOpenFileName(self,
                                   "select background file",
                                   self.ui.DIR.toPlainText(), # 
                                   "All Files (*);;Text Files (*.txt)")   # ?

        if fileName_choose == "":
           print("\n use default")
           return

        self.ui.BG_DIR.setText(fileName_choose)
        # self.update_background()

    def _on_galvo_waveform_input_changed(self, *_args):
        """波形相关参数变化 → 去抖 250ms 后重画预览（避免编辑过程中反复重算）。"""
        timer = getattr(self, "_galvo_waveform_timer", None)
        if timer is None:
            timer = qc.QTimer(self)
            timer.setSingleShot(True)
            timer.setInterval(250)
            timer.timeout.connect(self._draw_galvo_waveform_preview)
            self._galvo_waveform_timer = timer
        timer.start()

    def _draw_galvo_waveform_preview(self):
        """重画预览；失败（例如物镜未标定）时在面板上提示而不抛异常。"""
        try:
            self.update_galvoXwaveform()
        except Exception as error:
            label = getattr(self.ui, "XwaveformLabel", None)
            if label is not None:
                label.clear()
                label.setText("waveform unavailable: %s" % error)

    def render_galvo_waveform_preview(self, ao_waveform, do_waveform):
        """把生成好的 X/Y/DO 波形画到 XGalvo waveform 面板（与 AODO 线程同参数）。"""
        label = getattr(self.ui, "XwaveformLabel", None)
        if label is None or ao_waveform is None:
            return
        ao_waveform = np.asarray(ao_waveform)
        if ao_waveform.size <= 0:
            return
        acq_mode = self.ui.ACQMode.currentText() if hasattr(self.ui, "ACQMode") else ""
        postclocks = int(self.ui.FlyBack.value()) if hasattr(self.ui, "FlyBack") else 0
        width = max(120, int(label.width() or 320))
        height = max(80, int(label.height() or 220))
        label.setPixmap(
            fastLinePlot(ao_waveform, do_waveform, width=width, height=height, m=None, M=None,
                         y_axis=True, y_unit="V")
        )
        x_wave = (
            np.asarray(ao_waveform[0], dtype=np.float64)
            if ao_waveform.ndim == 2
            else np.asarray(ao_waveform, dtype=np.float64)
        )
        lines = [
            "preview: B-line (ACQMode=%s)" % acq_mode,
            "X: %.4f .. %.4f V" % (float(x_wave.min()), float(x_wave.max())),
        ]
        if ao_waveform.ndim == 2 and ao_waveform.shape[0] >= 2:
            y_wave = np.asarray(ao_waveform[1], dtype=np.float64)
            lines.append("Y: %.4f .. %.4f V" % (float(y_wave.min()), float(y_wave.max())))
        if do_waveform is not None and np.size(do_waveform) > 0:
            lines.append("DO triggers: %d" % int(np.count_nonzero(np.asarray(do_waveform))))
        lines.append("samples: %d" % int(ao_waveform.shape[-1]))
        lines.append("FlyBack: %d clocks" % postclocks)
        label.setToolTip("\n".join(lines))


    def update_galvoXwaveform(self):
        # calculate the total X range
        Xrange=self.ui.XLength.value()
        # update X range in lable
        self.ui.XrangeLabel.setText('X(mm): '+str(Xrange))
        self.ui.YrangeLabel.setText('Y(mm): '+str(self.ui.Ypixels.value()*self.ui.YStepSize.value()/1000))
        # generate waveform
        # 预览只画一条 B-line 的波形（X 前扫 + X 回程 + DO）：整卷 C-scan 波形有几十万点，
        # 生成与绘制都很慢，所以无论 ACQMode 是 Cscan 还是 Bline 都用 B-line 模式。
        DOwaveform, AOwaveform, status = GenAODO(mode='ContinuousBline', \
                                                 XStepSize = self.ui.XStepSize.value(), \
                                                 XSteps = x_position_count(self.ui), \
                                                 obj = self.ui.Objective.currentText(),\
                                                 postclocks = max(1, int(self.ui.FlyBack.value())), \
                                                 YStepSize = self.ui.YStepSize.value(), \
                                                 YSteps =  max(1, int(self.ui.Ypixels.value())), \
                                                 BVG = self.ui.AlineAVG.value(),\
                                                 BlineRepeat = self.ui.BlineAVG.value(),\
                                                 Galvo_bias = self.ui.XGalvoBias.value(),\
                                                 GalvoY_bias = self.ui.YGalvoBias.value(),\
                                                 )
        # show generating waveform result
        #print(self.Xwaveform)
        # current_message = self.ui.statusbar.currentMessage()
        # self.ui.statusbar.showMessage(current_message+status)
        if hasattr(self.ui, "XwaveformLabel"):
            self.ui.XwaveformLabel.clear()
        self.render_galvo_waveform_preview(AOwaveform, DOwaveform)

    # def update_galvoYwaveform(self):
    #     # calculate the total X range
    #     Yrange=self.ui.Ysteps.value()*self.ui.YStepSize.value()/1000
    #     # update X range in lable
    #     self.ui.YrangeLabel.setText('Y range(mm): '+str(Yrange))
    #     # generate waveform
    #     self.Ywaveform, status = GenGalvoWave(self.ui.YStepSize.value(),\
    #                                     self.ui.Ysteps.value(),\
    #                                     self.ui.BlineAVG.value(),\
    #                                     self.ui.YBias.value(),\
    #                                     self.ui.Objective.currentText())
    #     # show generating waveform result
    #     self.ui.statusbar.showMessage(status)
    #     if self.Ywaveform != None:
    #         pixmap = LinePlot(self.Xwaveform)
    #         # clear content on the waveformLabel
    #         self.ui.YwaveformLabel.clear()
    #         # update iamge on the waveformLabel
    #         self.ui.YwaveformLabel.setPixmap(pixmap)

    def Calculate_ImageDepth(self):
        self.image_depths = GenAlinesPerBlines(self.ui.ImageZStart.value(),\
                                       self.ui.ImageZDepth.value(),\
                                       self.ui.ImageZnumber.value())
        #print(self.image_depths)
        #self.ui.statusbar.showMessage(self.image_depths)

    def Calculate_SliceDepth(self):
        self.slice_depths = GenAlinesPerBlines(self.ui.SliceZStart.value(),\
                                       self.ui.SliceZDepth.value(),\
                                       self.ui.SliceZnumber.value())
        #print(self.slice_depths)
        #self.ui.statusbar.showMessage(self.image_depths)

    def Update_laser(self):
        laser = get_laser_spec(self.ui.Laser.currentText())
        if laser is None:
            self.ui.statusbar.showMessage('Laser invalid!!!')
            return
        self.Aline_frq = laser.aline_frequency_hz




