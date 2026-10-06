# -*- coding: utf-8 -*-
"""Stage 2 panel: build a volume out of a segmented rotational C-scan.

Opened from the stage 1 window ("3D reconstruction..." button) with the stack that
is already loaded, or started on its own::

    python RotationalVolumePanel.py --file Cscan-2-Y3750-X4800-Z420.tif

The panel walks the four steps the offline module offers, in order:

    * Prepare      - align every turn onto one common phase and average them, so
                     the reconstruction works on one turn that carries the SNR of
                     the whole recording, then build the volume grid around the
                     axis (the rig puts that axis inside the B-scan plane, 90 deg).
                     The depth window stage 1 reports only decided which rows
                     *segment* the recording: every depth row of the file goes into
                     the average and into the volume
    * Calibrate the centre - scan the lateral position of the axis on the B-line and
                     keep the one where the two half-turns land on top of each other.
                     That position decides which radius every page pixel gets, it is
                     the largest single cause of doubled rings, and the page match
                     cannot see it, so it is measured instead of guessed
    * Reconstruct  - first the same centre scan in a small voxel-scaled window, then
                     the tilt inside the automatic window, then the pose fit, then
                     splat the TWO half-turns of the turn separately: a full turn
                     images every object point twice, so where the two halves agree
                     the structure is real and where they differ the residual is pose
                     error. Both halves and their combined volume are shown.
    * Refine the axis again (finer) - the pose fit at the reconstruction settings
                     instead of coarse ones, for when the halves still disagree;
                     press Reconstruct afterwards to paint it
    Save writes the combined volume TIFF and the slices PNG.

So the run is **Prepare, then Reconstruct, and nothing else**: the fits (centre, tilt,
pose) all happen inside Reconstruct, and the pose they settle on stays in the controls
until Prepare runs again (Prepare is what rebuilds the pose and the grid from the
controls). The separate Calibrate button is the same scan over a wide window, for a
fresh mount. ``Autofocus rounds`` = 0 turns every automatic fit off, which is what
makes a run reproducible from the numbers in the controls alone.

After Reconstruct the panel shows four panels: the first and the second half-turn,
each as an **XY** slice (across the axis, Z slider) and an **XZ** slice (along the
axis, Y slider; the middle Y is the B-scan plane itself). One slider step is one
voxel, and both halves are drawn at the same Z/Y so they can be compared directly.

After Reconstruct the panel shows four panels: the first and the second half-turn,
each as an **XY** slice (across the axis, Z slider) and an **XZ** slice (along the
axis, Y slider; the middle Y is the B-scan plane itself). One slider step is one
voxel, and both halves are drawn at the same Z/Y so they can be compared directly.

All heavy work runs in a :class:`VolumeWorker` thread, so the window stays alive,
and the module's own report lines are printed into the log. --self-test exercises
the whole panel on a tilted phantom without a display.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

try:  # pragma: no cover - the panel needs PyQt5
    from PyQt5 import QtCore, QtWidgets
    from PyQt5.QtCore import Qt, QThread, pyqtSignal

    QT_AVAILABLE = True
except Exception as _qt_error:  # pragma: no cover - keeps --self-test importable
    QT_AVAILABLE = False
    QtCore = QtWidgets = None
    _QT_ERROR = _qt_error

    class QThread:  # placeholder so the module stays importable without Qt
        def __init__(self, parent=None):
            pass

    def pyqtSignal(*args, **kwargs):  # noqa: N802 - mirrors the Qt name
        return None

    class Qt:
        Horizontal = 1

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import RotationalVolume as rv  # noqa: E402 - needs the folder on sys.path


class VolumeWorker(QThread):
    """One step of the stage 2 pipeline, off the GUI thread."""

    progressed = pyqtSignal(int, int, str)
    logged = pyqtSignal(str)                             # one line for the panel log
    prepared = pyqtSignal(object, object, object)        # pages, info, grid
    focused = pyqtSignal(object)                         # autofocus result
    calibrated = pyqtSignal(object)                      # centre scan result
    reconstructed = pyqtSignal(object)                   # volume, weight, axis, match
    failed = pyqtSignal(str)

    def __init__(self, task, settings, parent=None):
        super().__init__(parent)
        self.task = str(task)
        self.settings = dict(settings)

    def _progress(self, count, total, text):
        self.progressed.emit(int(count), int(total), str(text))

    def _note(self, text):
        """One line for the panel's log: what a stage cost and where it landed."""
        self.logged.emit(str(text))

    def _prepare(self):
        """Align the turns, average them, and size the grid for the tilt scan.

        The depth window stage 1 reports only decided which rows *segment* the
        recording. Every depth row of the file goes into the average and into the
        volume, so the reconstruction is not cut off at the window the segmentation
        used. ``notes`` carries one line per segment of the plan (its B-lines, its
        length, whether it was kept), which is what makes a plan that does not line
        up with the file visible instead of silently changing the stack.
        """
        settings = self.settings
        data, plan = settings["data"], settings["plan"]
        turn_pages = int(settings["turn_pages"])
        self._progress(0, 1, "locating the turns")
        start = time.perf_counter()
        notes = []
        spans = rv.segment_spans(plan, int(data.shape[0]), turn_pages, notes=notes)
        spans, matches = rv.refine_offsets(data, spans, turn_pages,
                                           max_shift=int(settings["max_shift"]),
                                           progress=self._progress)
        self._note(f"turns located: {len(spans)} turn(s) in "
                   f"{time.perf_counter() - start:.0f} s")
        if settings["average"]:
            self._progress(0, 1, "averaging the turns")
            start = time.perf_counter()
            pages, counts, match = rv.average_turns(data, spans, turn_pages)
            self._note(f"turns averaged in {time.perf_counter() - start:.0f} s")
        else:
            start, _usable = max(spans, key=lambda item: item[1])
            pages = np.asarray(data[start:start + turn_pages, :, :])
            counts = np.ones(int(pages.shape[0]), dtype=np.int32)
            match = 1.0
        if pages is None:
            raise RuntimeError("no complete turn in the recording to stack")
        centre = (float(settings["centre_s"]), float(settings["centre_t"]))
        # The rig's axis lies in the B-scan plane, so the pose starts at the tilt the
        # controls carry (90 deg) and Reconstruct re-checks it inside a small window.
        start_tilt = float(settings["tilt_deg"])
        axis = rv.AxisPose(centre, start_tilt, float(settings["azimuth_deg"]))
        # The grid holds the most tilted pose that will be tried, which is the start
        # value plus the automatic window: the pages of one pose only reach the cone
        # |z'| <= rho*tan(tilt) around the axis.
        envelope = rv.AxisPose(
            centre, abs(start_tilt) + max(0.0, float(settings["tilt_span"])),
            float(settings["azimuth_deg"]))
        voxel = tuple(settings["voxel_um"])
        grid = rv.default_grid(envelope, float(settings["dx"]),
                               float(settings["dz"]), int(pages.shape[1]),
                               int(data.shape[2]),
                               voxel_um=voxel,
                               # Room for the fits to move the axis: the grid is built
                               # around the pose, so a fitted pose always fits itself,
                               # but the *object* it places would walk out of the
                               # volume if the piercing point started far off.
                               margin_um=float(min(voxel)) + rv.POSE_MARGIN_UM)
        info = {"spans": [(int(start), int(usable)) for start, usable in spans],
                "matches": [float(value) for value in matches],
                "counts": counts, "match": float(match),
                "turn_pages": turn_pages, "notes": notes,
                "depth_rows": int(data.shape[2]),
                "depth_window": tuple(settings["depth_window"]),
                "axis": axis, "envelope": envelope}
        self.prepared.emit(pages, info, grid)

    def _focus(self):
        settings = self.settings
        result = rv.autofocus(settings["pages"], settings["angles"],
                              float(settings["dx"]), float(settings["dz"]),
                              settings["grid"], settings["axis"],
                              # The tilt stays where the automatic tilt scan put it;
                              # what is left is the piercing point, the in-plane angle
                              # and the offset along the plane normal. That offset is
                              # the term that flips sign between the two half-turns,
                              # so it is the one that can make them agree.
                              parameters=rv.POSE_PARAMETERS,
                              steps=rv.POSE_STEPS,
                              rounds=int(settings["rounds"]),
                              decimate=settings["decimate"],
                              page_step=int(settings["page_step"]),
                              row_offset=float(settings["row_offset"]),
                              limits={"offset_n_um": (-rv.DEFAULT_OFFSET_LIMIT_UM,
                                                      rv.DEFAULT_OFFSET_LIMIT_UM)},
                              tolerance=rv.FIT_TOLERANCE, progress=self._progress)
        self.focused.emit(result)

    def _coarse(self, settings):
        """Cheap settings for the fits: the same pose at a third of the detail."""
        decimate = settings["decimate"]
        return ((max(1, int(decimate[0]) * 3), max(1, int(decimate[1]) * 3)),
                max(1, int(settings["page_step"]) * 3))

    def _calibrate(self):
        """Find the lateral pierce point that makes the two half-turns agree."""
        settings = self.settings
        # page_step 1 here: the scan picks whichever of its two metrics separates the
        # candidates, and the one that works without a shell to fit (a solid sample) is
        # the noise-limited one, so it needs every page. This is the once-per-mount
        # scan; the small one Reconstruct runs can afford the coarser step.
        result = rv.calibrate_centre_s(
            settings["pages"], settings["angles"], float(settings["dx"]),
            float(settings["dz"]), settings["grid"], settings["axis"],
            decimate=settings["decimate"], page_step=1,
            row_offset=float(settings["row_offset"]), progress=self._progress)
        self.calibrated.emit(result)

    def _reconstruct(self):
        """Fit the axis (centre, tilt window, pose), then splat the two halves."""
        settings = self.settings
        axis = settings["axis"]
        coarse, coarse_step = self._coarse(settings)
        centre = None
        if int(settings["rounds"]) > 0:
            # The lateral pierce point comes first: it is the one term that pulls the
            # two half-turns apart, the page match cannot see it, and this scan is the
            # metric that can. Small window scaled to the voxel, one stage, so it costs
            # a handful of cheap half reconstructions.
            voxel = float(np.mean(np.asarray(settings["voxel_um"],
                                             dtype=np.float64)[:2]))
            self._progress(0, 1, "auto centre tune")
            start = time.perf_counter()
            centre = rv.calibrate_centre_s(
                settings["pages"], settings["angles"], float(settings["dx"]),
                float(settings["dz"]), settings["grid"], axis,
                span_um=rv.CENTRE_FINE_TUNE_VOXELS * voxel,
                step_um=rv.CENTRE_FINE_TUNE_STEP_VOXELS * voxel,
                # At the coarse settings the tilt and the pose fit use: the window is
                # +-2.5 voxels wide and the objective is a smooth curve of the assumed
                # centre, so sampling pages finer than this costs minutes and does not
                # move the vertex. 'Calibrate the centre' is the fine one.
                fine_span_um=0.0, decimate=coarse, page_step=coarse_step,
                row_offset=float(settings["row_offset"]), progress=self._progress)
            axis = centre["axis"]
            self._note(f"centre tune: {len(centre['curve'])} candidates in "
                       f"{time.perf_counter() - start:.0f} s -> B-line "
                       f"{axis.centre[0]:.1f} px (two-half ratio "
                       f"{float(centre['best_ratio']):.3f})")
        curve = []
        span = float(settings["tilt_span"])
        if span > 0.0:
            self._progress(0, 1, "auto tilt")
            start = time.perf_counter()
            axis, curve = rv.fine_tilt(
                settings["pages"], settings["angles"], float(settings["dx"]),
                float(settings["dz"]), settings["grid"], axis, span_deg=span,
                step_deg=0.25, decimate=coarse, page_step=coarse_step,
                row_offset=float(settings["row_offset"]), progress=self._progress)
            self._note(f"tilt scan: {len(curve)} trials in "
                       f"{time.perf_counter() - start:.0f} s -> "
                       f"{axis.tilt_deg:.2f} deg")
        fit = None
        # The pose fit runs after the tilt and before the render, so one button does
        # the whole job: with the tilt at exactly 90 deg the objective is mirror
        # symmetric in the offset along the normal and cannot see it at all, so the
        # tilt has to be settled first.
        if int(settings["rounds"]) > 0:
            self._progress(0, 1, "auto pose fit")
            start = time.perf_counter()
            fit = rv.autofocus(
                settings["pages"], settings["angles"], float(settings["dx"]),
                float(settings["dz"]), settings["grid"], axis,
                parameters=rv.POSE_PARAMETERS, steps=rv.POSE_STEPS,
                rounds=int(settings["rounds"]), decimate=coarse,
                page_step=coarse_step, row_offset=float(settings["row_offset"]),
                limits={"offset_n_um": (-rv.DEFAULT_OFFSET_LIMIT_UM,
                                        rv.DEFAULT_OFFSET_LIMIT_UM)},
                tolerance=rv.FIT_TOLERANCE, progress=self._progress)
            axis = fit["axis"]
            self._note(f"pose fit: {max(0, len(fit['history']) - 1)} round(s) in "
                       f"{time.perf_counter() - start:.0f} s -> match "
                       f"{float(fit['score']):.4f}")
        self._progress(0, 1, "splatting the two half-turns")
        start = time.perf_counter()
        halves, combined = rv.reconstruct_halves(
            settings["pages"], settings["angles"], axis, float(settings["dx"]),
            float(settings["dz"]), settings["grid"],
            decimate=settings["decimate"],
            page_step=int(settings["page_step"]),
            row_offset=float(settings["row_offset"]), progress=self._progress)
        self._note(f"render: the two half-turns at {tuple(settings['decimate'])} "
                   f"/ page step {int(settings['page_step'])} in "
                   f"{time.perf_counter() - start:.0f} s")
        self.reconstructed.emit(
            {"halves": halves, "combined": combined, "axis": axis,
             "tilt_curve": list(curve), "fit": fit, "centre_fit": centre,
             "match": float(np.mean([half["match"] for half in halves])),
             "agreement": float(rv.halves_agreement(halves))})

    def run(self):  # noqa: D102 - QThread entry point
        try:
            if self.task == "prepare":
                self._prepare()
            elif self.task == "calibrate":
                self._calibrate()
            elif self.task == "focus":
                self._focus()
            elif self.task == "reconstruct":
                self._reconstruct()
            else:
                raise ValueError(f"unknown task {self.task!r}")
        except Exception as error:  # noqa: BLE001 - surfaced in the panel
            self.failed.emit(f"{error}")


if QT_AVAILABLE:

    class VolumeWindow(QtWidgets.QMainWindow):
        """Stage 2: stack the turns, calibrate the axis, reconstruct the volume."""

        def __init__(self, data=None, meta=None, plan=None, parent=None):
            super().__init__(parent)
            self.setWindowTitle("Rotational C-scan 3D reconstruction (stage 2)  |  "
                                "tilted-axis stacking and splatting")
            self.resize(1300, 820)
            self.data = data
            self.meta = dict(meta or {})
            self.plan = dict(plan or {})
            self.pages = None
            self.angles = np.zeros(0)
            self.grid = None
            self.axis = None
            self.applied_pose = None
            self.info = {}
            self.match = 0.0
            self.agreement = 0.0
            self.centre_result = None
            self.centre_split = None
            self.halves = []
            self.combined = (None, None)
            self.tilt_curve = []
            self.display_min = 0.0
            self.display_max = 1.0
            self.volume = None
            self.weight = None
            self.worker = None
            self.busy = False
            self.blocking = False
            # Fixed by the recording or by stage 1, so they are reported, not edited:
            # the two pixel pitches of a B-line, the turn length stage 1 measured and
            # the depth rows the segmentation used. ``centre_t`` (the depth row the
            # axis pierces) is a gauge - moving the pierce point along the axis does
            # not move the line the object turns about - so it is kept but not shown.
            self.dx_um = float(rv.LATERAL_PIXEL_SIZE_UM)
            self.dz_um = float(rv.DEFAULT_AXIAL_STEP_UM)
            self.turn_pages = 0
            self.depth_window = (0, 0)
            self.centre_t = 0.0
            self._build_ui()
            if self.data is not None:
                self.adopt_scan(data, self.meta, self.plan)

        # -- small builders -------------------------------------------------- #

        def _spin(self, value, low, high, step=1.0, decimals=1, suffix=""):
            box = QtWidgets.QDoubleSpinBox(self)
            box.setRange(float(low), float(high))
            box.setSingleStep(float(step))
            box.setDecimals(int(decimals))
            if suffix:
                box.setSuffix(suffix)
            box.setValue(float(value))
            return box

        def _row(self, *widgets):
            row = QtWidgets.QHBoxLayout()
            for widget in widgets:
                if isinstance(widget, str):
                    row.addWidget(QtWidgets.QLabel(widget, self))
                else:
                    row.addWidget(widget)
            row.addStretch(1)
            return row

        def _log(self, text):
            self.log.appendPlainText(str(text))

        # -- layout ---------------------------------------------------------- #

        def _build_ui(self):
            central = QtWidgets.QWidget(self)
            self.setCentralWidget(central)
            root = QtWidgets.QVBoxLayout(central)
            top = QtWidgets.QHBoxLayout()

            controls = QtWidgets.QVBoxLayout()
            controls.addWidget(self._build_page_group())
            controls.addWidget(self._build_grid_group())
            controls.addWidget(self._build_axis_group())
            controls.addWidget(self._build_action_group())
            controls.addStretch(1)
            holder = QtWidgets.QWidget(self)
            holder.setLayout(controls)
            holder.setFixedWidth(440)
            top.addWidget(holder)

            right = QtWidgets.QVBoxLayout()
            right.addWidget(self._build_volume_view(), 1)
            self.log = QtWidgets.QPlainTextEdit(self)
            self.log.setReadOnly(True)
            self.log.setMaximumBlockCount(500)
            self.log.setFixedHeight(130)
            right.addWidget(self.log)
            top.addLayout(right, 1)
            root.addLayout(top, 1)

            self.progress = QtWidgets.QProgressBar(self)
            self.progress.setRange(0, 100)
            self.progress.setValue(0)
            root.addWidget(self.progress)
            self._refresh_sampling()

        def _build_page_group(self):
            group = QtWidgets.QGroupBox("Pages", self)
            layout = QtWidgets.QVBoxLayout(group)
            self.scan_label = QtWidgets.QLabel("No scan loaded.", self)
            self.scan_label.setWordWrap(True)
            layout.addWidget(self.scan_label)

            # Depth rows, both pixel pitches and the turn length come from the
            # recording and from stage 1, so they are printed above instead of being
            # offered as spin boxes: only the alignment tolerance is a choice.
            self.shift_spin = QtWidgets.QSpinBox(self)
            self.shift_spin.setRange(0, 20)
            self.shift_spin.setValue(2)
            self.shift_spin.setToolTip(
                "How far the automatic alignment may move a turn's first page; one "
                "page is 360 deg / (B-lines per turn).")
            layout.addLayout(self._row("Align the turns within +-", self.shift_spin,
                                       "page(s)"))
            self.average_check = QtWidgets.QCheckBox("average the turns", self)
            self.average_check.setChecked(True)
            self.average_check.setToolTip(
                "Page k of every turn shows the same direction, so averaging them "
                "multiplies the SNR; with the box clear only one turn is used.")
            layout.addWidget(self.average_check)
            self.turns_label = QtWidgets.QLabel("-", self)
            self.turns_label.setWordWrap(True)
            layout.addWidget(self.turns_label)
            return group

        def _build_grid_group(self):
            group = QtWidgets.QGroupBox("Volume grid", self)
            layout = QtWidgets.QVBoxLayout(group)
            self.voxel_lateral = self._spin(20.0, 2.0, 200.0, 1.0, 1, " um")
            self.voxel_lateral.setToolTip(
                "Voxel edge across the rotation axis (the XY plane of the volume). "
                "The reconstruction samples the volume at 16 bytes per voxel, so "
                "10 M voxels are 0.16 GB and 100 M are 1.6 GB - the grid label "
                "shows the count after Prepare. Press Prepare again after changing "
                "it.")
            self.voxel_axial = self._spin(rv.DEFAULT_AXIAL_STEP_UM, 1.0, 100.0, 0.1,
                                          2, " um")
            self.voxel_axial.setToolTip(
                "Voxel edge along the rotation axis. The depth axis is the sharp one "
                "- it is the OCT axial sampling of 6.6 um - and it is not tied to "
                "the lateral voxel: the two pixel pitches of the B-line are "
                "different, so one number for both would over-sample one of them.")
            layout.addLayout(self._row("Voxel", self.voxel_lateral, "lateral,",
                                       self.voxel_axial, "axial"))
            self.voxel_lateral.valueChanged.connect(self._refresh_sampling)
            self.voxel_axial.valueChanged.connect(self._refresh_sampling)

            # The page-pixel sampling is fixed to the empirical rule in the labels
            # below (one page pixel per voxel, never coarser than the pitch), so it
            # is reported rather than offered as two more spin boxes.
            self.sampling_label = QtWidgets.QLabel("-", self)
            self.sampling_label.setWordWrap(True)
            layout.addWidget(self.sampling_label)
            self.page_step_spin = QtWidgets.QSpinBox(self)
            self.page_step_spin.setRange(1, 64)
            self.page_step_spin.setValue(1)
            self.page_step_spin.setToolTip(
                "How many views are used per turn: the plane spacing at radius rho "
                "is rho * 360 deg / (B-lines per turn) * this. With 858 B-lines per "
                "turn one page is 0.42 deg, so at 6 mm the planes are 44 um apart "
                "with 1 and 88 um with 2 - this is the main speed lever and the "
                "main limit away from the axis.")
            layout.addLayout(self._row("Use every", self.page_step_spin,
                                       "th page of the turn"))
            self.grid_label = QtWidgets.QLabel("-", self)
            self.grid_label.setWordWrap(True)
            layout.addWidget(self.grid_label)
            return group

        def _refresh_sampling(self, _value=None):
            """Show the page-pixel sampling the voxels imply (no user setting)."""
            lateral, axial = self.decimate()
            self.sampling_label.setText(
                f"Page sampling: every {lateral} px lateral, {axial} px depth "
                f"(voxel / pixel pitch, {self.dx_um:.1f} um and {self.dz_um:.1f} um)")

        def _build_axis_group(self):
            group = QtWidgets.QGroupBox("Rotation axis", self)
            layout = QtWidgets.QVBoxLayout(group)
            self.centre_s = self._spin(0.0, -20000.0, 20000.0, 1.0, 1, " px")
            self.centre_s.setToolTip(
                "Lateral pixel of the B-line where the rotation axis pierces it. This "
                "decides which radius every page pixel gets: a wrong value splits the "
                "two half-turns apart (they move opposite ways by twice the error), and "
                "the page match does not see it at all. 'Calibrate the centre' measures "
                "it and the log reports the residual split. The depth row it pierces "
                "is not shown: sliding the pierce point along the axis leaves the line "
                "the object turns about where it was, so nothing depends on it.")
            layout.addLayout(self._row("Pierces the B-line at", self.centre_s))
            self.azimuth_spin = self._spin(90.0, -360.0, 360.0, 5.0, 1, " deg")
            self.azimuth_spin.setToolTip(
                "Into which in-plane direction the axis leans: 90 deg is along the "
                "imaging depth axis, 0 deg along the lateral sweep.")
            layout.addLayout(self._row("Axis in-plane angle", self.azimuth_spin))
            self.tilt_spin = self._spin(rv.DEFAULT_TILT_DEG, 60.0, 120.0, 0.5, 2,
                                        " deg")
            self.tilt_spin.setToolTip(
                "Angle between the axis and the plane normal. The rig has the axis "
                "in the B-scan plane, which is 90 deg here; Reconstruct re-checks it "
                "inside the window below.")
            layout.addLayout(self._row("Tilt", self.tilt_spin))
            self.tilt_span_spin = self._spin(2.0, 0.0, 10.0, 0.5, 2, " deg")
            self.tilt_span_spin.setToolTip(
                "Every Reconstruct scans the tilt inside +-this window (in 0.25 deg "
                "steps, at coarse settings) and keeps the value with the best page "
                "match. 0 switches the automatic search off.")
            layout.addLayout(self._row("Tilt auto +/-", self.tilt_span_spin))
            self.rounds_spin = QtWidgets.QSpinBox(self)
            self.rounds_spin.setRange(0, 8)
            self.rounds_spin.setValue(2)
            self.rounds_spin.setToolTip(
                "Rounds of coordinate descent the pose fit runs: the piercing point, "
                "the in-plane angle and the offset of the axis along the plane normal "
                "- the term that flips sign between the two half-turns. Reconstruct "
                "runs it at coarse settings, 'Refine the axis' at the settings below. "
                "0 turns the fit off, and Reconstruct then uses the pose as typed.")
            layout.addLayout(self._row("Autofocus rounds", self.rounds_spin))
            self.pose_label = QtWidgets.QLabel("-", self)
            self.pose_label.setWordWrap(True)
            layout.addWidget(self.pose_label)
            return group

        def _build_action_group(self):
            group = QtWidgets.QGroupBox("Run", self)
            layout = QtWidgets.QVBoxLayout(group)
            self.prepare_button = QtWidgets.QPushButton("1. Prepare the turns", self)
            self.prepare_button.clicked.connect(lambda: self.start_task("prepare"))
            layout.addWidget(self.prepare_button)
            self.calibrate_button = QtWidgets.QPushButton(
                "2. Calibrate the centre (once per mount)", self)
            self.calibrate_button.clicked.connect(
                lambda: self.start_task("calibrate"))
            self.calibrate_button.setToolTip(
                "Scans the lateral position of the rotation axis on the B-line, +-5 "
                "pixels around the value in the controls, and keeps the one where the "
                "two half-turns land on top of each other. That position is what "
                "decides which radius every page pixel gets, and the page match cannot "
                "see it - this metric can. It reads every page, so it takes a couple of "
                "minutes; Reconstruct runs a small version of the same scan by itself "
                "(+-2.5 voxels), which is the wider net of the two when the voxels are "
                "coarse. If the log says the best value sat on the edge of the window, "
                "move the centre closer by hand or widen the window in the module.")
            layout.addWidget(self.calibrate_button)
            self.reconstruct_button = QtWidgets.QPushButton("3. Reconstruct", self)
            self.reconstruct_button.clicked.connect(
                lambda: self.start_task("reconstruct"))
            self.reconstruct_button.setToolTip(
                "Fits the tilt inside the automatic window, fits the pose (piercing "
                "point, in-plane angle, offset along the normal) and then splats the "
                "two half-turns. Everything Reconstruct needs is done in this one "
                "step.")
            layout.addWidget(self.reconstruct_button)
            self.focus_button = QtWidgets.QPushButton(
                "Refine the axis again (finer)", self)
            self.focus_button.clicked.connect(lambda: self.start_task("focus"))
            self.focus_button.setToolTip(
                "The same pose fit as Reconstruct runs, but at the settings below "
                "instead of coarse ones, and without touching the tilt. Press "
                "Reconstruct afterwards to paint the result - it keeps this pose "
                "until you press Prepare again.")
            layout.addWidget(self.focus_button)
            row = QtWidgets.QHBoxLayout()
            self.save_button = QtWidgets.QPushButton("Save volume...", self)
            self.save_button.clicked.connect(self.save_volume)
            row.addWidget(self.save_button)
            self.slices_button = QtWidgets.QPushButton("Save slices...", self)
            self.slices_button.clicked.connect(self.save_slices)
            row.addWidget(self.slices_button)
            layout.addLayout(row)
            for button in (self.calibrate_button, self.focus_button,
                           self.reconstruct_button, self.save_button,
                           self.slices_button):
                button.setEnabled(False)
            return group

        def _slice_row(self, text):
            """A slider row: label, slider, spin box, ``i/N`` readout."""
            row = QtWidgets.QHBoxLayout()
            row.addWidget(QtWidgets.QLabel(text, self))
            slider = QtWidgets.QSlider(Qt.Horizontal, self)
            slider.setRange(0, 0)
            row.addWidget(slider, 1)
            spin = QtWidgets.QSpinBox(self)
            spin.setRange(0, 0)
            spin.valueChanged.connect(slider.setValue)
            slider.valueChanged.connect(spin.setValue)
            row.addWidget(spin)
            label = QtWidgets.QLabel("-", self)
            row.addWidget(label)
            return row, slider, spin, label

        def brightness(self):
            """Display gain on the slice window; 1.0 is the automatic range."""
            return max(0.01, float(self.brightness_spin.value()) / 100.0)

        def _display_levels(self):
            """``(vmin, vmax)`` the slices are painted with, brightness applied.

            The automatic levels keep the black end where the data put it and the
            gain scales the span above it, so 100 % reproduces the auto range
            exactly and the dark end of the picture does not move while the slider
            is dragged. ``(None, None)`` when there is no usable range, which lets
            the axes (or matplotlib, for the saved file) pick their own.
            """
            low = float(self.display_min)
            high = float(self.display_max)
            if not (np.isfinite(low) and np.isfinite(high)) or high <= low:
                return None, None
            return low, low + (high - low) / self.brightness()

        def _brightness_changed(self, _value=None):
            """Show the gain and repaint the four slices."""
            self.brightness_label.setText(f"x{self.brightness():.2f}")
            self.draw_slices()

        def _build_volume_view(self):
            group = QtWidgets.QGroupBox(
                "Volume: first half-turn (top) and second half-turn (bottom), "
                "XY across the axis and XZ along it", self)
            layout = QtWidgets.QVBoxLayout(group)
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
            from matplotlib.figure import Figure
            self.volume_figure = Figure(figsize=(9, 6), dpi=100)
            self.axes = {}
            for row, half in enumerate("AB"):
                self.axes[half + "_xy"] = self.volume_figure.add_subplot(
                    2, 2, row * 2 + 1)
                self.axes[half + "_xz"] = self.volume_figure.add_subplot(
                    2, 2, row * 2 + 2)
            self.volume_canvas = FigureCanvasQTAgg(self.volume_figure)
            layout.addWidget(self.volume_canvas, 1)

            # XY = a slice across the rotation axis; its Z slider walks up and down
            # the axis. XZ = a slice along the axis (Y = 0 is the B-scan plane
            # itself, because the axis lies in it); its Y slider walks out of that
            # plane through the sample.
            row, self.z_slider, self.z_spin, self.z_label = self._slice_row("Z slice")
            self.z_slider.setToolTip("Position along the rotation axis")
            layout.addLayout(row)
            row, self.y_slider, self.y_spin, self.y_label = self._slice_row(
                "Y slice (XZ view)")
            self.y_slider.setToolTip(
                "Position across the axis, out of the B-scan plane: the middle is "
                "the B-scan plane itself")
            layout.addLayout(row)
            # Brightness is a display gain on the window the slices are painted with:
            # the automatic levels stay what they are (100 % is exactly the auto
            # range), so this only changes the picture and never the volume.
            row = QtWidgets.QHBoxLayout()
            row.addWidget(QtWidgets.QLabel("Brightness", self))
            self.brightness_slider = QtWidgets.QSlider(Qt.Horizontal, self)
            self.brightness_slider.setRange(10, 400)
            self.brightness_slider.setValue(100)
            self.brightness_slider.setToolTip(
                "Display gain on the slice window. 100 % is the automatic range "
                "(the 2nd to 99.5th percentile of the voxels the two halves share); "
                "larger brightens the picture, smaller darkens it. It changes "
                "nothing in the volume itself, and the saved slices are written "
                "with the same window.")
            row.addWidget(self.brightness_slider, 1)
            self.brightness_spin = QtWidgets.QSpinBox(self)
            self.brightness_spin.setRange(10, 400)
            self.brightness_spin.setValue(100)
            self.brightness_spin.setSuffix(" %")
            row.addWidget(self.brightness_spin)
            self.brightness_label = QtWidgets.QLabel("x1.00", self)
            row.addWidget(self.brightness_label)
            layout.addLayout(row)
            self.brightness_slider.valueChanged.connect(self.brightness_spin.setValue)
            self.brightness_spin.valueChanged.connect(self.brightness_slider.setValue)
            self.brightness_slider.valueChanged.connect(self._brightness_changed)
            self.z_slider.valueChanged.connect(self.draw_slices)
            self.y_slider.valueChanged.connect(self.draw_slices)
            return group

        # -- state ----------------------------------------------------------- #

        def adopt_scan(self, data, meta=None, plan=None):
            """Take the stack stage 1 already loaded (a reference, no reload)."""
            self.data = data
            self.meta = dict(meta or self.meta or {})
            self.plan = dict(plan or self.plan or {})
            depth = int(data.shape[2])
            window = [int(value) for value in
                      (self.meta.get("depth_window") or [0, depth])]
            self.depth_window = (max(0, window[0]), min(depth, int(window[1])))
            self.dx_um = float(self.meta.get("x_step_um")
                               or rv.LATERAL_PIXEL_SIZE_UM)
            self.dz_um = float(self.meta.get("z_step_um")
                               or rv.DEFAULT_AXIAL_STEP_UM)
            self.turn_pages = self._turn_pages(data)
            self.centre_s.setValue(0.5 * (int(data.shape[1]) - 1))
            self.centre_t = 0.5 * (self.depth_window[0] + self.depth_window[1] - 1)
            self.scan_label.setText(self._scan_text())
            self._describe_plan()
            self._refresh_sampling()
            self._log(f"scan taken from stage 1: {tuple(data.shape)}, "
                      f"{self.turn_pages} B-lines per turn, pixel pitch "
                      f"{self.dx_um:.1f} x {self.dz_um:.1f} um")
            self._refresh_buttons()

        def set_depth_window(self, start, stop):
            """Rows the segmentation measured from, when stage 1 edits them.

            The volume always uses every row of the file; this only changes what the
            panel reports, so a window the user moved in stage 1 is not shown as the
            one the file was loaded with.
            """
            if self.data is None:
                return
            depth = int(self.data.shape[2])
            self.meta["depth_window"] = [int(start), int(stop)]
            self.depth_window = (max(0, int(start)), min(depth, int(stop)))
            self.scan_label.setText(self._scan_text())

        def _scan_text(self):
            """The fixed inputs, as one label: nothing here is a control."""
            if self.data is None:
                return "No scan loaded."
            depth = int(self.data.shape[2])
            return (
                f"{int(self.data.shape[0])} B-lines x {int(self.data.shape[1])} px "
                f"lateral x {depth} px depth (one page is one B-line).\n"
                f"pixel pitch {self.dx_um:.1f} um along the B-line, "
                f"{self.dz_um:.1f} um in depth.\n"
                f"stage 1 segmented the recording from depth rows "
                f"{self.depth_window[0]}..{self.depth_window[1] - 1}; the average "
                f"and the volume use all {depth} rows.\n"
                f"turn {int(self.turn_pages)} B-lines = "
                f"{int(self.turn_pages) / 2.0:.0f} per 180 deg.")

        def _turn_pages(self, data):
            """Turn length of the plan, with the nominal one as the fallback."""
            plan = self.plan or {}
            for key in ("turn_pages", "expected_turn_pages"):
                value = int(plan.get(key) or 0)
                if value > 1:
                    return value
            return max(2, int(data.shape[0] // 3))

        def _describe_plan(self):
            """Show every segment of the plan with its B-lines, kept or not.

            The label lists the segments Prepare will actually average, the log adds
            the ones that were dropped and why - a plan that does not line up with
            the B-lines in the file is the one thing that would quietly change the
            stack.
            """
            if self.data is None:
                return
            notes = []
            spans = rv.segment_spans(self.plan, int(self.data.shape[0]),
                                     int(self.turn_pages), notes=notes)
            planned = len((self.plan or {}).get("offsets") or [])
            suffix = ("" if planned == len(spans) else
                      f"  (stage 1 found {planned} segments: the log lists them)")
            self.turns_label.setText(
                f"{len(spans)} turn(s) of {int(self.turn_pages)} B-lines, at B-lines "
                + ", ".join(f"{start}..{start + usable - 1}"
                            for start, usable in spans)
                + suffix)
            self._log(f"segments: stage 1 found {planned} segment(s), "
                      f"{len(spans)} of them hold a whole turn")
            for line in notes:
                self._log(f"  {line}")
            starts = [start for start, _usable in spans]
            if len(starts) > 1:
                gaps = [int(value) for value in np.diff(starts)]
                if max(abs(value - int(self.turn_pages)) for value in gaps) > 2:
                    self._log(f"  note: the turns start every {gaps} B-lines but the "
                              f"turn is {int(self.turn_pages)} - either the turn or "
                              f"the segment starts are off, and the average would "
                              f"then smear; check the page match after Prepare")

        def decimate(self):
            """Page-pixel sampling: one page pixel per voxel (the empirical rule)."""
            lateral = max(1, int(round(float(self.voxel_lateral.value())
                                       / max(1e-6, float(self.dx_um)))))
            axial = max(1, int(round(float(self.voxel_axial.value())
                                     / max(1e-6, float(self.dz_um)))))
            return (lateral, axial)

        def settings(self):
            return {
                "data": self.data, "plan": self.plan, "pages": self.pages,
                "angles": self.angles, "grid": self.grid, "axis": self.axis,
                "depth_window": tuple(self.depth_window),
                "turn_pages": int(self.turn_pages),
                "max_shift": int(self.shift_spin.value()),
                "average": bool(self.average_check.isChecked()),
                "dx": float(self.dx_um),
                "dz": float(self.dz_um),
                "voxel_um": (float(self.voxel_lateral.value()),
                             float(self.voxel_axial.value())),
                "decimate": self.decimate(),
                "page_step": int(self.page_step_spin.value()),
                # Every depth row of the file is reconstructed, so the pages hold
                # rows 0.. and the row offset is zero.
                "row_offset": 0.0,
                "tilt_deg": float(self.tilt_spin.value()),
                "tilt_span": float(self.tilt_span_spin.value()),
                "centre_s": float(self.centre_s.value()),
                "centre_t": float(self.centre_t),
                "azimuth_deg": float(self.azimuth_spin.value()),
                "rounds": int(self.rounds_spin.value()),
            }

        def _pose_text(self):
            """One line for the pose: the depth row it pierces is a gauge, not shown."""
            if self.axis is None:
                return "-"
            return (f"axis at {self.axis.centre[0]:.1f} px on the B-line, tilt "
                    f"{self.axis.tilt_deg:.2f} deg, in-plane angle "
                    f"{self.axis.azimuth_deg:.1f} deg, offset out of the B-scan "
                    f"plane {self.axis.offset_n_um:.1f} um")

        def _adopt_edited_pose(self, task):
            """Take the centre/azimuth from the controls once the user edits them.

            Without this the pose the panel found (a tilt scan or an autofocus) is
            what every later step uses, and typing a new centre into the spin box
            would do nothing. A pose that is *found* is kept as long as the controls
            are not touched, so pressing Autofocus twice refines instead of
            restarting.
            """
            if self.axis is None:
                return
            current = (float(self.centre_s.value()), float(self.centre_t),
                       float(self.azimuth_spin.value()))
            if self.applied_pose == current:
                return
            self.axis = rv.AxisPose((current[0], current[1]), self.axis.tilt_deg,
                                    current[2], self.axis.offset_n_um)
            self.applied_pose = current
            self._log(f"pose taken from the controls: {self._pose_text()}")

        def start_task(self, task, blocking=False):
            """Run one pipeline step; ``blocking`` is for the offscreen self-test."""
            if task == "prepare" and self.data is None:
                self._log("load a scan first")
                return None
            if task != "prepare" and (self.pages is None or self.axis is None):
                self._log("run 'Prepare the turns' first")
                return None
            self._adopt_edited_pose(task)
            worker = VolumeWorker(task, self.settings(), self)
            worker.progressed.connect(self._on_progress)
            worker.logged.connect(self._log)
            worker.prepared.connect(self._on_prepared)
            worker.calibrated.connect(self._on_calibrated)
            worker.focused.connect(self._on_focused)
            worker.reconstructed.connect(self._on_reconstructed)
            worker.failed.connect(self._on_failed)
            self.worker = worker
            self.set_busy(True, f"{task}: running")
            if blocking:
                self.blocking = True
                worker.run()
                self.blocking = False
                self.worker = None
                self.set_busy(False, f"{task}: done")
            else:
                worker.finished.connect(lambda: self.set_busy(False, f"{task}: done"))
                worker.start()
            return worker

        # -- buttons --------------------------------------------------------- #

        def set_busy(self, busy, text=""):
            self.busy = bool(busy)
            if text:
                self.statusBar().showMessage(str(text))
            if busy:
                for button in self._all_buttons():
                    button.setEnabled(False)
            else:
                self._refresh_buttons()

        def _all_buttons(self):
            return (self.prepare_button, self.focus_button,
                    self.reconstruct_button, self.save_button, self.slices_button)

        def _refresh_buttons(self):
            ready = self.pages is not None and self.grid is not None
            posed = ready and self.axis is not None
            self.prepare_button.setEnabled(self.data is not None)
            self.calibrate_button.setEnabled(posed)
            self.focus_button.setEnabled(posed)
            self.reconstruct_button.setEnabled(posed)
            self.save_button.setEnabled(self.volume is not None)
            self.slices_button.setEnabled(self.volume is not None)

        # -- worker results -------------------------------------------------- #

        def _on_progress(self, count, total, message):
            self.progress.setRange(0, max(1, int(total)))
            self.progress.setValue(int(count))
            self.statusBar().showMessage(str(message))

        def _show_pose(self):
            """Push the pose actually used back into the controls and the label.

            Keeping the controls in step means the next run does not mistake the
            fitted numbers for a manual edit, and the panel always shows what the
            volume was built from.
            """
            if self.axis is None:
                return
            self.centre_s.setValue(self.axis.centre[0])
            self.centre_t = float(self.axis.centre[1])
            self.azimuth_spin.setValue(self.axis.azimuth_deg)
            self.tilt_spin.setValue(self.axis.tilt_deg)
            self.applied_pose = (float(self.centre_s.value()),
                                 float(self.centre_t),
                                 float(self.azimuth_spin.value()))
            self.pose_label.setText(self._pose_text())

        def _on_prepared(self, pages, info, grid):
            self.pages = np.asarray(pages)
            self.info = dict(info)
            self.grid = grid
            self.axis = info["axis"]
            self.angles = rv.page_angles(int(self.pages.shape[0]))
            counts = info.get("counts")
            reached = int(np.max(np.asarray(counts))) if counts is not None else 1
            grid_text = self._grid_text(grid)
            self._log(f"pages   : {tuple(self.pages.shape)} "
                      f"(B-lines x lateral x depth rows), {len(info['spans'])} "
                      f"turn(s) at B-lines "
                      + ", ".join(f"{start}..{start + usable - 1}"
                                  for start, usable in info["spans"])
                      + f", every phase fed by up to {reached} turn(s), page match "
                        f"{info['match']:.3f}")
            for line in info.get("notes") or []:
                self._log(f"  segment {line}")
            self._log(f"depth   : all {int(info.get('depth_rows', self.pages.shape[2]))} "
                      f"rows of the file are in the stack; stage 1 segmented from "
                      f"rows {info.get('depth_window', (0, 0))[0]}.."
                      f"{info.get('depth_window', (0, 0))[1] - 1}")
            self._log(f"grid    : {grid_text}")
            self._log(f"raw     : the pages are the recorded amplitudes of "
                      f"{len(info['spans'])} turn(s) averaged phase by phase - no "
                      f"offset, no scale, no mean subtracted (min "
                      f"{float(self.pages.min()):.4g}, mean "
                      f"{float(self.pages.mean()):.4g}, max "
                      f"{float(self.pages.max()):.4g})")
            self._log(f"pose    : {self._pose_text()}")
            self.turns_label.setText(
                f"{len(info['spans'])} turn(s) of {int(info['turn_pages'])} B-lines, "
                f"at B-lines "
                + ", ".join(f"{start}..{start + usable - 1}"
                            for start, usable in info["spans"])
                + f"  -  page match {info['match']:.3f}")
            self.grid_label.setText(grid_text)
            self._show_pose()
            self._refresh_buttons()

        @staticmethod
        def _grid_text(grid):
            """One line for the grid: shape, voxel edges, splat footprint."""
            spacing = np.asarray(grid.spacing, dtype=np.float64).reshape(-1)
            count = int(grid.size())
            total = count * 16        # two float64 volumes while the pages are splatted
            size = (f"{total / 1e6:.0f} MB" if total < 1e9
                    else f"{total / 1e9:.2f} GB")
            return (f"{tuple(grid.shape)} voxels of {spacing[0]:.1f} um lateral x "
                    f"{spacing[2]:.1f} um axial ({count / 1e6:.1f} M, about {size} "
                    f"while splatting)")

        def _on_calibrated(self, result):
            """Take the lateral pierce point the centre scan settled on."""
            self.centre_result = result
            self.axis = result["axis"]
            dx = float(self.dx_um)
            column = int(result["column"])
            before = float(result["start_objective"])
            after = float(result["objective"])
            moved = float(self.axis.centre[0]) - float(result["start_centre_s"])
            self._log(f"centre  : lateral pierce point {float(result['start_centre_s']):.1f} "
                      f"px -> {float(self.axis.centre[0]):.1f} px ({moved * dx:+.0f} um), "
                      f"{result['metric']} {before:.3g} -> {after:.3g}"
                      + ("" if result["accepted"] else
                         f"  [not accepted: nothing improved on the starting pose by more "
                         f"than {float(result['tolerance']) * 100:.0f}%, so it stands]")
                      + ("  [the best value sits on the edge of the window - widen it]"
                         if result.get("edge") else ""))
            self._log(f"          both metrics over the window: two-half mismatch "
                      f"{float(result['start_ratio']):.3f} -> {float(result['ratio']):.3f} "
                      f"(contrast {float(result['ratio_contrast']):.2f}), shell radius split "
                      f"{float(result['start_amplitude_um']):.0f} -> "
                      f"{float(result['amplitude_um']):.0f} um on the shell in "
                      f"{result['band_um'][0]:.0f}..{result['band_um'][1]:.0f} um "
                      f"(contrast {float(result['shell_contrast']):.2f}); the scan keeps "
                      f"whichever separates the candidates")
            curve = list(result["coarse"]["curve"])
            if result.get("fine"):
                curve = curve + list(result["fine"]["curve"])
            lowest = min(entry[column] for entry in curve if np.isfinite(entry[column]))
            stride = max(1, len(curve) // 10)
            for index in range(0, len(curve), stride):
                position, ratio, agreement, split = curve[index]
                mark = " <- best" if abs(curve[index][column] - lowest) < 1e-12 else ""
                self._log(f"  scan  : {position:7.1f} px  mismatch {ratio:6.3f}  "
                          f"split {split:7.0f} um  agreement {agreement:.3f}{mark}")
            self._show_pose()
            self._refresh_buttons()
            self._log("        press Reconstruct to paint the volume at this centre")

        def _on_focused(self, result):
            self.axis = result["axis"]
            self._log(f"refined : {self._pose_text()} "
                      f"(page match {result['score']:.3f})")
            for index, pose, score in result["history"]:
                self._log(f"  round {index}: B-line {pose.centre[0]:.1f} px, "
                          f"in-plane angle {pose.azimuth_deg:.1f} deg, tilt "
                          f"{pose.tilt_deg:.1f} deg, out of plane "
                          f"{pose.offset_n_um:.1f} um, match {score:.3f}")
            # Show what was refined, and keep the controls in step with it so the
            # next run does not mistake the new numbers for a manual edit.
            self._show_pose()
            self._refresh_buttons()

        def _on_reconstructed(self, result):
            self.axis = result["axis"]
            self.match = float(result["match"])
            self.agreement = float(result["agreement"])
            self.tilt_curve = list(result.get("tilt_curve") or [])
            # Keep float32 copies: two half volumes of a 20 um grid are big enough.
            self.halves = [
                {"start": int(half["start"]), "stop": int(half["stop"]),
                 "match": float(half["match"]),
                 "volume": np.asarray(half["volume"], dtype=np.float32),
                 "weight": np.asarray(half["weight"], dtype=np.float32)}
                for half in result["halves"]]
            combined = result["combined"]
            self.volume = (np.asarray(combined[0], dtype=np.float32)
                           if combined[0] is not None else None)
            self.weight = (np.asarray(combined[1], dtype=np.float32)
                           if combined[1] is not None else None)
            self._show_pose()
            fit = result.get("fit")
            dx = float(self.dx_um)
            centre_fit = result.get("centre_fit")
            if centre_fit is not None:
                moved = (float(centre_fit["centre_s"])
                         - float(centre_fit["start_centre_s"]))
                self._log(
                    f"auto centre: lateral pierce point {moved * dx:+.0f} um "
                    f"({float(centre_fit['start_centre_s']):.1f} -> "
                    f"{float(centre_fit['centre_s']):.1f} px), "
                    f"{centre_fit['metric']} "
                    f"{float(centre_fit['start_objective']):.3g} -> "
                    f"{float(centre_fit['objective']):.3g}"
                    + ("" if centre_fit["accepted"] else
                       "  [not accepted, so the pose stands]"))
            if fit:
                history = fit["history"]
                start_score = float(history[0][2])
                gain = float(fit["score"]) - start_score
                self._log(f"auto fit: {len(history) - 1} round(s) of pose fitting at "
                          f"coarse settings, score {start_score:.4f} -> "
                          f"{float(fit['score']):.4f}"
                          + ("" if gain > 0.0 and gain > rv.FIT_TOLERANCE else
                             " - no probe beat the starting pose by more than "
                             f"{rv.FIT_TOLERANCE:g}, so that pose stands"))
            self._log(f"pose    : {self._pose_text()}")
            if self.tilt_curve:
                best, _score = max(self.tilt_curve, key=lambda item: item[1])
                edge = (abs(best - self.tilt_curve[0][0]) < 1e-9
                        or abs(best - self.tilt_curve[-1][0]) < 1e-9)
                self._log(f"auto tilt: {self.axis.tilt_deg:.2f} deg chosen from "
                          f"{len(self.tilt_curve)} trials in +-"
                          f"{float(self.tilt_span_spin.value()):.2f} deg "
                          f"(page match {_score:.4f})"
                          + ("  [the best value sits on the edge of the window]"
                             if edge else ""))
            for index, half in enumerate(self.halves[:2]):
                self._log(f"half {index + 1}  : B-lines {half['start'] + 1}-"
                          f"{half['stop']}, page match {half['match']:.3f}")
            self._log(f"halves  : agreement {self.agreement:.3f} between the two "
                      f"half-turn volumes (what they share is structure, what they "
                      f"do not is residual pose error)")
            try:
                self.centre_split = rv.halves_radial_split(self.halves, self.grid)
                phase = float(self.centre_split["phase_deg"])
                self._log(
                    f"split   : two-half radial difference "
                    f"{float(self.centre_split['amplitude_um']):.0f} um peak (cosine "
                    f"amplitude on the shell in "
                    f"{self.centre_split['band_um'][0]:.0f}.."
                    f"{self.centre_split['band_um'][1]:.0f} um), phase {phase:.0f} deg "
                    f"- the residual that shows as doubled rings, smaller is better; "
                    f"a phase near 0/180 deg is a lateral pierce error, near 90/270 deg "
                    f"an offset out of the B-scan plane")
            except ValueError:
                self.centre_split = None
            if self.volume is not None:
                mask = rv.weight_mask(self.weight, 0.5)
                filled = int(np.count_nonzero(mask))
                if filled:
                    values = self.volume[mask]
                    samples = self.weight[mask]
                    self.display_min = float(np.percentile(values, 2.0))
                    self.display_max = float(np.percentile(values, 99.5))
                    self._log(f"combined: {tuple(self.volume.shape)}, {filled} of "
                              f"{self.volume.size} voxels filled, values "
                              f"{float(values.min()):.3g}..{float(values.max()):.3g}")
                    self._log(f"          median {float(np.median(samples)):.0f} "
                              f"sample(s) per voxel - under about 4 the speckle is "
                              f"not averaged away yet")
                else:
                    self._log("combined: nothing filled - the pose misses the data")
                self.z_slider.setRange(0, max(0, int(self.volume.shape[2]) - 1))
                self.z_spin.setRange(0, max(0, int(self.volume.shape[2]) - 1))
                self.y_slider.setRange(0, max(0, int(self.volume.shape[1]) - 1))
                self.y_spin.setRange(0, max(0, int(self.volume.shape[1]) - 1))
                lateral, axial = self.voxel_um()
                self.z_slider.setToolTip(f"{axial:.1f} um per slice, along the "
                                         f"rotation axis")
                self.y_slider.setToolTip(f"{lateral:.1f} um per slice, across the "
                                         f"axis, out of the B-scan plane")
                self.draw_volume()   # jumps to the middle slice
            self.draw_slices()
            self._refresh_buttons()

        def _on_failed(self, message):
            self._log(f"failed  : {message}")
            if not self.blocking:
                QtWidgets.QMessageBox.critical(self, "Stage 2 error", str(message))
            self._refresh_buttons()

        # -- drawing and saving ---------------------------------------------- #

        def draw_volume(self):
            """Jump both views to the middle slice; ``draw_slices`` paints them."""
            if self.volume is None:
                self.draw_slices()
                return
            self.z_slider.setValue(int(self.volume.shape[2]) // 2)
            self.y_slider.setValue(int(self.volume.shape[1]) // 2)

        def _clamp(self, value, count):
            return max(0, min(int(count) - 1, int(value)))

        def draw_slices(self, _value=None):
            """Repaint the two half-turn views from the sliders; four imshows, live."""
            if not self.halves or self.halves[0]["volume"] is None:
                for axes in self.axes.values():
                    axes.clear()
                    axes.set_title("no volume yet", fontsize=8)
                    axes.axis("off")
                self.z_label.setText("-")
                self.y_label.setText("-")
                self.volume_figure.tight_layout()
                self.volume_canvas.draw_idle()
                return
            labels = ("first half", "second half")
            lateral, axial = self.voxel_um()
            shape = self.halves[0]["volume"].shape
            ys, zs = int(shape[1]), int(shape[2])
            z = self._clamp(self.z_slider.value(), zs)
            y = self._clamp(self.y_slider.value(), ys)
            levels = self._display_levels()
            for index, half in enumerate(self.halves[:2]):
                volume = half["volume"]
                for suffix, plane, kind, ratio in (
                        ("_xy", volume[:, :, z], f"XY Z={z + 1}/{zs}", 1.0),
                        ("_xz", volume[:, y, :], f"XZ Y={y + 1}/{ys}",
                         lateral / max(1e-6, axial))):
                    axes = self.axes["AB"[index] + suffix]
                    axes.clear()
                    # The voxels are not cubes: the axial one is finer than the
                    # lateral ones, so the XZ view is drawn at the real scale between
                    # the two else the sample looks stretched along the axis.
                    axes.imshow(plane, cmap="gray", origin="lower", aspect=ratio,
                                vmin=levels[0], vmax=levels[1])
                    axes.set_title(
                        f"{labels[index]}: B-lines {half['start'] + 1}-{half['stop']}, "
                        f"match {half['match']:.2f}  |  {kind}, {lateral:.1f} lateral x "
                        f"{axial:.1f} axial um/voxel", fontsize=7)
                    axes.axis("off")
            self.z_label.setText(f"{z + 1}/{zs}")
            self.y_label.setText(f"{y + 1}/{ys}")
            self.volume_figure.tight_layout()
            self.volume_canvas.draw_idle()

        def voxel_um(self):
            """``(lateral, axial)`` voxel edge of the volume that is on screen."""
            if self.grid is None:
                return (float(self.voxel_lateral.value()),
                        float(self.voxel_axial.value()))
            edges = np.asarray(self.grid.spacing, dtype=np.float64).reshape(-1)
            return (float(edges[0]), float(edges[-1]))

        def save_volume(self):
            if self.volume is None:
                return None
            path, _filter = QtWidgets.QFileDialog.getSaveFileName(
                self, "Save volume", "volume.tif", "TIFF files (*.tif *.tiff)")
            if not path:
                return None
            from tifffile import imwrite
            imwrite(Path(path), self.volume)
            self._log(f"volume written: {path}")
            return path

        def render_halves(self, path):
            """Write the four panels (both half-turns, XY and XZ) the UI shows."""
            from matplotlib import pyplot as plt
            path = Path(path)
            if not self.halves or self.halves[0]["volume"] is None:
                return None
            lateral, axial = self.voxel_um()
            shape = self.halves[0]["volume"].shape
            zs, ys = int(shape[2]), int(shape[1])
            z = self._clamp(self.z_slider.value(), zs)
            y = self._clamp(self.y_slider.value(), ys)
            levels = self._display_levels()
            figure, axes = plt.subplots(2, 2, figsize=(11, 8), dpi=100)
            for index, half in enumerate(self.halves[:2]):
                volume = half["volume"]
                for column, plane, kind, ratio in (
                        (0, volume[:, :, z], f"XY Z={z + 1}/{zs}", 1.0),
                        (1, volume[:, y, :], f"XZ Y={y + 1}/{ys}",
                         lateral / max(1e-6, axial))):
                    axes[index, column].imshow(
                        plane, cmap="gray", origin="lower", aspect=ratio,
                        vmin=levels[0], vmax=levels[1])
                    axes[index, column].set_title(
                        f"{'first' if index == 0 else 'second'} half-turn: B-lines "
                        f"{half['start'] + 1}-{half['stop']}, page match "
                        f"{half['match']:.3f}  |  {kind}, {lateral:.1f} lateral x "
                        f"{axial:.1f} axial um/voxel", fontsize=8)
                    axes[index, column].axis("off")
            figure.suptitle(f"tilt {self.axis.tilt_deg:.2f} deg, agreement between the "
                            f"two halves {self.agreement:.3f}  |  "
                            f"pose {self._pose_text()}", fontsize=9)
            figure.tight_layout()
            figure.savefig(path)
            plt.close(figure)
            return path

        def save_slices(self):
            if self.volume is None:
                return None
            path, _filter = QtWidgets.QFileDialog.getSaveFileName(
                self, "Save slices", "slices.png", "PNG files (*.png)")
            if not path:
                return None
            rv.save_slice_png(self.volume, Path(path),
                              spacing=(None if self.grid is None
                                       else self.grid.spacing),
                              levels=self._display_levels())
            halves = self.render_halves(Path(path).with_name(
                Path(path).stem + "_halves.png"))
            self._log(f"slices written: {path} (combined) and {halves} "
                      f"(first and second half-turn, side by side), brightness "
                      f"x{self.brightness():.2f} on both")
            return path


# --------------------------------------------------------------------------- #
# Self-test (headless) and command line
# --------------------------------------------------------------------------- #


def run_self_test(verbose=True):
    """Drive the whole panel on a tilted phantom, without a display."""
    if not QT_AVAILABLE:
        if verbose:
            print(f"[FAIL] the panel needs PyQt5 ({_QT_ERROR})")
        return [("PyQt5 available", False, str(_QT_ERROR))]
    application = (QtWidgets.QApplication.instance()
                   or QtWidgets.QApplication(sys.argv[:1]))
    checks = []

    def report(name, ok, detail=""):
        checks.append((name, bool(ok), str(detail)))
        if verbose:
            print(f"[{'PASS' if ok else 'FAIL'}] {name}"
                  + (f"  {detail}" if detail else ""))
        return bool(ok)

    case, stack, truth, clean = rv._turns_case(turn_pages=180, offsets=(5, 11),
                                               tilt_deg=90.0)
    turn_pages = int(clean.shape[0])
    meta = {"x_step_um": float(case["dx"]),
            "depth_window": [0, int(stack.shape[2])],
            "z_pixels": int(stack.shape[2]), "pages": int(stack.shape[0])}
    plan = {"offsets": [int(start) + 1 for start in truth],
            "lengths": [turn_pages] * len(truth), "turn_pages": turn_pages}
    window = VolumeWindow(stack, meta, plan)
    # The voxels and the page-pixel sampling follow from each other now (one page
    # pixel per voxel), so four page pixels per voxel of each pitch reproduces the
    # sampling this check was written against.
    window.dx_um = float(case["dx"])
    window.dz_um = float(case["dz"])
    window.voxel_lateral.setValue(4.0 * float(case["dx"]))
    window.voxel_axial.setValue(4.0 * float(case["dz"]))
    window.page_step_spin.setValue(1)
    window.tilt_spin.setValue(float(case["axis"].tilt_deg))
    window.tilt_span_spin.setValue(2.0)
    window.azimuth_spin.setValue(float(case["axis"].azimuth_deg))
    window.rounds_spin.setValue(0)
    window.average_check.setChecked(True)

    report("the panel adopts the scan stage 1 loaded",
           int(window.turn_pages) == turn_pages
           and abs(float(window.dx_um) - float(case["dx"])) < 1e-9
           and window.prepare_button.isEnabled(),
           f"turn {int(window.turn_pages)} B-lines, pixel pitch "
           f"{float(window.dx_um):.1f} x {float(window.dz_um):.1f} um, page sampling "
           f"{window.decimate()} px, Prepare enabled: "
           f"{window.prepare_button.isEnabled()}")

    window.start_task("prepare", blocking=True)
    prepared = window.pages is not None and window.grid is not None
    window.start_task("reconstruct", blocking=True)
    produced = window.volume is not None and bool(np.any(window.volume > 0.0))
    halves_ok = (len(window.halves) == 2
                 and all(half["volume"] is not None
                         and bool(np.any(half["volume"] > 0.0))
                         for half in window.halves))
    report("the panel reconstructs both half-turns and combines them",
           prepared and produced and halves_ok and window.match > 0.5
           and window.agreement > 0.5
           and abs(float(window.axis.offset_n_um)) < 400.0,
           f"pages {None if window.pages is None else tuple(window.pages.shape)}, "
           f"halves "
           + ", ".join(f"{half['start'] + 1}-{half['stop']} "
                       f"({None if half['volume'] is None else tuple(half['volume'].shape)}) "
                       f"match {half['match']:.2f}" for half in window.halves)
           + f"; agreement {window.agreement:.3f}; auto tilt "
             f"{window.axis.tilt_deg:.2f} deg from a start of "
             f"{float(case['axis'].tilt_deg):.0f}; pose fit left the axis "
             f"{float(window.axis.offset_n_um):.0f} um out of the B-scan plane "
             f"(the phantom has it at 0); combined "
             f"{None if window.volume is None else tuple(window.volume.shape)}")

    # 2. The lateral pierce point is the term that splits the two half-turns apart, so
    #    the centre scan has to find it again from an offset start - and it has to be
    #    the metric, because the page match does not move for this error at all. The
    #    offset stays inside the default window (+-5 px).
    true_centre_s = float(case["axis"].centre[0])
    window.axis = case["axis"].adjusted("centre_s", 4.0)
    window.start_task("calibrate", blocking=True)
    found = float(window.axis.centre[0])
    before = float(window.centre_result["start_objective"])
    after = float(window.centre_result["objective"])
    report("the centre scan finds the lateral pierce point again",
           abs(found - true_centre_s) <= 2.0 and after < before,
           f"started 4 px off ({true_centre_s:.1f} -> {true_centre_s + 4.0:.1f} px), "
           f"settled on {found:.1f} px, {window.centre_result['metric']} "
           f"{before:.3f} -> {after:.3f}")

    # 3. Given the true pose, the volume the panel builds has to match the phantom
    #    it came from: that checks the whole chain, splatting and grid included,
    #    through the panel's own code path. Both fits off, so the pose really is the
    #    one handed in.
    window.rounds_spin.setValue(0)
    window.tilt_span_spin.setValue(0.0)
    window.axis = case["axis"]
    window.start_task("reconstruct", blocking=True)
    phantom_on_grid = rv.sample_trilinear(
        case["phantom"], case["grid"],
        rv.grid_positions(window.grid).reshape(-1, 3)).reshape(window.grid.shape)
    mask = rv.weight_mask(window.weight, 0.75)
    match = rv.correlation(window.volume, phantom_on_grid, mask)
    page_match = rv.consistency_score(window.pages, window.angles, case["axis"],
                                      float(case["dx"]), float(case["dz"]),
                                      window.grid, decimate=(4, 4))
    # A noisy three-turn stack at 2 degrees per page is what this test feeds the
    # panel, so the bar is the structure coming back (about 0.8) and not the 0.97
    # a noise-free single turn reaches in the module's own checks.
    report("the panel's volume matches the phantom at the true pose",
           match > 0.70,
           f"correlation {match:.3f} over {int(np.count_nonzero(mask))} voxels of "
           f"the {tuple(window.grid.shape)} grid, page match {page_match:.3f}")


    # 4. Both halves the UI paints can also be written out for the record.
    import tempfile
    with tempfile.TemporaryDirectory() as folder:
        written = window.render_halves(Path(folder) / "halves.png")
        size = written.stat().st_size if written is not None and written.exists() else 0
    report("the panel writes the two half-turn views as one PNG", size > 20000,
           f"{None if written is None else written.name}, {size} bytes")

    window.close()
    passed = sum(1 for _, ok, _ in checks if ok)
    if verbose:
        print(f"\n{passed}/{len(checks)} checks passed")
    return checks


def main(argv=None):
    """``--self-test`` for the headless check, ``--file`` for a recording."""
    parser = argparse.ArgumentParser(
        description="Stage 2 panel: turn a segmented rotational C-scan into a "
                    "volume (tilted-axis stacking and splatting).")
    parser.add_argument("--self-test", action="store_true",
                        help="drive the panel on a phantom without a display")
    parser.add_argument("--file", type=Path, action="append",
                        help="C-scan TIFF to load (repeatable); without it the "
                             "panel opens empty and takes a scan from stage 1")
    parser.add_argument("--turn-pages", type=int,
                        help="pages per turn (default: as segmented)")
    parser.add_argument("--voxel-um", type=float, default=20.0,
                        help="lateral voxel edge across the rotation axis")
    parser.add_argument("--voxel-axial-um", type=float,
                        default=rv.DEFAULT_AXIAL_STEP_UM,
                        help="voxel edge along the rotation axis (the OCT axial step)")
    args = parser.parse_args(argv)

    if args.self_test:
        checks = run_self_test()
        return 0 if all(ok for _, ok, _ in checks) else 1
    if not QT_AVAILABLE:
        print(f"PyQt5 is required for the panel ({_QT_ERROR})")
        return 1
    application = QtWidgets.QApplication(sys.argv[:1])
    window = VolumeWindow()
    if args.file:
        stage1 = rv._load_stage1()
        data, meta = stage1.load_scan([Path(path) for path in args.file])
        plan = stage1.detect_segments(data, stage1.DEFAULT_HALF_TURN_BLINES,
                                      depth_window=tuple(meta["depth_window"]))
        if args.turn_pages:
            plan["turn_pages"] = int(args.turn_pages)
        window.adopt_scan(data, meta, plan)
        window.voxel_lateral.setValue(float(args.voxel_um))
        window.voxel_axial.setValue(float(args.voxel_axial_um))
    window.show()
    return application.exec_()


if __name__ == "__main__":
    sys.exit(main())







