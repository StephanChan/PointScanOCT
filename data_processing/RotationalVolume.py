# -*- coding: utf-8 -*-
"""Rotational C-scan 3D reconstruction prototype - STAGE 2, offline.

Stage 1 (:mod:`RotationalReconstruct`) splits the recording into full turns and
shows that the turns repeat. This module turns those repeats into a volume, and
it does **not** assume that the rotation axis lies in the B-scan plane - which is
the normal case, since the spindle axis is set by the mount, not by the scan.

Geometry
--------
The B-scan frame is fixed in the lab and the same for every page:

    P(s, t) = O + s*dx*u + t*dz*d          u = lateral, d = depth, n = u x d

A page is a plane through the object; the sample rotates about the spindle axis
``L = {q + lam*a}``. A page at angle ``theta`` means: the object point ``X`` sits
at the lab position ``q + R(theta)(X - q)``, so the object point imaged at pixel
``(s, t)`` is ``X = q + R(-theta)(P(s, t) - q)``. That one line is the whole
forward model (:func:`page_to_object`), and it is why a pixel of a page is a
*sample* of the object and not a line integral.

If ``a`` lies in the B-scan plane the pages are meridian planes and the classic
polar assembly works. If ``a`` is out of the plane the pages are oblique
("laminography"): they cut the object at a fixed angle to the axis and never pass
through it. Every voxel is still sampled - each page is a fully sampled 2D image,
so the pages cover the volume - and reconstructing is therefore a *resampling*
problem for a known geometry, not an ill-posed inversion. The hard part is the
axis pose: a fraction of a pixel of pose error smears out-of-plane detail, and
finding the pose (:func:`autofocus`, or a phantom) is what this prototype is for.

What is here (offline, numpy only - no Qt, no GUI):
    * the forward model and its inverse: :func:`page_to_object`,
      :func:`object_to_page`
    * multi-turn stacking: :func:`segment_spans`, :func:`refine_offsets`,
      :func:`average_turns`
    * trilinear splatting reconstruction on any grid: :func:`reconstruct`
    * the turn split into its two halves, which image every point twice, and the
      automatic tilt window: :func:`half_turn_spans`, :func:`reconstruct_halves`,
      :func:`combine_halves`, :func:`halves_agreement`, :func:`fine_tilt`
    * a phantom of revolution and its simulated pages: :func:`phantom_volume`,
      :func:`simulate_pages`
    * page-match axis calibration, bounded where it needs to be: :func:`autofocus`
      (see :data:`POSE_PARAMETERS`) and :func:`focus_curve`
    * a self-test that round-trips a *tilted* phantom, and an inspector that
      estimates the pose of a real file.

Usage::

    python RotationalVolume.py --self-test
    python RotationalVolume.py --file Cscan-2-Y3750-X4800-Z420.tif
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np

# Kept in sync with RotationalReconstruct; the depth step is the optical (in-air)
# sampling of the B-line, 6.6 um per row with the index of air taken as 1, so the
# depth axis of the volume needs no refractive-index correction.
LATERAL_PIXEL_SIZE_UM = 10.0
DEFAULT_AXIAL_STEP_UM = 6.6
DEFAULT_DEPTH_WINDOW = (100, 350)
# Measured on the rig: the spindle axis lies in the B-scan plane, i.e. 90 degrees
# from the plane normal. That is the classic meridian case, so it is the default
# pose here and the tilt scan only has to confirm it.
DEFAULT_TILT_DEG = 90.0

# The pose terms the fit is allowed to move, with the steps it probes them with: the
# lateral position of the piercing point in pixels, the in-plane angle in degrees and
# the offset along the plane normal in micrometres. Two parameters are deliberately
# absent:
#   * ``centre_t`` is a gauge. Moving the piercing point *along* the axis leaves the
#     line the object turns about where it was, so the mapping does not change at all
#     (R(theta)*a = a); fitting it only walks the grid around for nothing.
#   * ``tilt_deg`` is owned by the automatic tilt window (:func:`fine_tilt`).
POSE_PARAMETERS = ("centre_s", "azimuth_deg", "offset_n_um")
POSE_STEPS = (4.0, 15.0, 50.0)
# A bound on the offset, so a flat or monotone objective cannot walk it away.
DEFAULT_OFFSET_LIMIT_UM = 1000.0
# Room the grid keeps for the fit to move the axis inside it, on top of one voxel.
POSE_MARGIN_UM = 250.0
# How much the page match has to improve before a fit step is believed: the score is
# a normalised residual in 0..1, and a cheap reconstruction jitters in its fourth
# decimal, so without this a fit would walk the pose on micro-improvements that carry
# no information (measured: a centre drifting 8 px for a change of 1e-4).
FIT_TOLERANCE = 1e-3
# The lateral pierce point is the one pose term that pulls the two half-turns apart -
# the two observations of a point sit at opposite ends of the B-line, so an error
# moves them opposite ways, by twice the error - and it is the one the page match
# cannot see. So it gets its own scan, scored by :func:`halves_radial_split` /
# :func:`halves_mismatch`. The window is set in *pixels of the B-line* rather than in
# micrometres, because that is the frame the error lives in and it then scales with the
# lateral pixel size; the wide pair is for a fresh mount and the fine pair polishes the
# parabola. Measured on the rig: the scan lands within a pixel of the best position,
# and the two-half split falls from 181 um to 6 um when it does.
CENTRE_SCAN_SPAN_PX = 5.0
CENTRE_SCAN_STEP_PX = 1.0
CENTRE_SCAN_FINE_SPAN_PX = 1.5
CENTRE_SCAN_FINE_STEP_PX = 0.5
CENTRE_FINE_TUNE_VOXELS = 2.5
CENTRE_FINE_TUNE_STEP_VOXELS = 0.5


# --------------------------------------------------------------------------- #
# Vectors, the axis pose and the grid
# --------------------------------------------------------------------------- #


def rotation_matrix(axis, theta):
    """Rodrigues rotation about the unit ``axis`` by ``theta`` radians, 3x3."""
    x, y, z = (float(value) for value in axis)
    c, s = math.cos(float(theta)), math.sin(float(theta))
    rest = 1.0 - c
    return np.array(
        [[c + x * x * rest, x * y * rest - z * s, x * z * rest + y * s],
         [y * x * rest + z * s, c + y * y * rest, y * z * rest - x * s],
         [z * x * rest - y * s, z * y * rest + x * s, c + z * z * rest]],
        dtype=np.float64,
    )


class AxisPose:
    """Pose of the spindle axis in the B-scan frame.

    ``centre`` is the pixel where the axis pierces the B-scan plane, ``tilt_deg``
    is the angle between the axis and the plane normal ``n`` (90 deg = the axis
    lies in the plane, the classic case; 0 deg = perpendicular to it, the one
    pose that ever sees a single plane), and ``azimuth_deg`` says into which
    in-plane direction the axis leans. ``offset_n_um`` shifts the axis along
    ``n``, which is what an axis parallel to the plane needs.
    """

    __slots__ = ("centre", "tilt_deg", "azimuth_deg", "offset_n_um")

    def __init__(self, centre=(600.0, 210.0), tilt_deg=0.0, azimuth_deg=0.0,
                 offset_n_um=0.0):
        self.centre = (float(centre[0]), float(centre[1]))
        self.tilt_deg = float(tilt_deg)
        self.azimuth_deg = float(azimuth_deg)
        self.offset_n_um = float(offset_n_um)

    def copy(self):
        return AxisPose(self.centre, self.tilt_deg, self.azimuth_deg,
                        self.offset_n_um)

    def adjusted(self, name, delta):
        """Copy with one named parameter moved by ``delta`` (friendly names)."""
        other = self.copy()
        if name in ("centre_s", "s_c"):
            other.centre = (self.centre[0] + float(delta), self.centre[1])
        elif name in ("centre_t", "t_c"):
            other.centre = (self.centre[0], self.centre[1] + float(delta))
        elif name == "tilt_deg":
            other.tilt_deg = self.tilt_deg + float(delta)
        elif name == "azimuth_deg":
            other.azimuth_deg = self.azimuth_deg + float(delta)
        elif name == "offset_n_um":
            other.offset_n_um = self.offset_n_um + float(delta)
        else:
            raise KeyError(name)
        return other

    def unit_vector(self):
        """Axis direction in the ``(u, d, n)`` basis."""
        tilt = math.radians(self.tilt_deg)
        azimuth = math.radians(self.azimuth_deg)
        return np.array([math.sin(tilt) * math.cos(azimuth),
                         math.sin(tilt) * math.sin(azimuth),
                         math.cos(tilt)], dtype=np.float64)

    def point_um(self, dx, dz):
        """A point of the axis, in micrometres in the ``(u, d, n)`` basis."""
        s_c, t_c = self.centre
        return np.array([s_c * dx, t_c * dz, self.offset_n_um], dtype=np.float64)

    def describe(self):
        return (f"axis through pixel ({self.centre[0]:.1f}, {self.centre[1]:.1f})"
                f", tilt {self.tilt_deg:.2f} deg from the plane normal"
                f", azimuth {self.azimuth_deg:.1f} deg"
                f", n offset {self.offset_n_um:.1f} um")


def frame_from_axis(axis, along_axis_z=True):
    """Orthonormal grid frame (rows ``e0, e1, e2``) built around the axis.

    With ``along_axis_z`` the third grid axis is the rotation axis, so the
    sample stands upright in the volume instead of leaning with the mount.
    """
    a = axis.unit_vector()
    if not along_axis_z:
        return np.eye(3, dtype=np.float64)
    seed = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(seed, a))) > 0.9:
        seed = np.array([0.0, 1.0, 0.0])
    e0 = seed - float(np.dot(seed, a)) * a
    e0 = e0 / float(np.linalg.norm(e0))
    e1 = np.cross(a, e0)
    return np.stack([e0, e1, a], axis=0)


class Grid:
    """Voxel grid: ``position = origin + (i*sp0)e0 + (j*sp1)e1 + (k*sp2)e2``.

    ``frame`` rows are the grid axes in the ``(u, d, n)`` basis, ``spacing`` is
    micrometres per voxel and ``shape`` counts voxels along ``e0, e1, e2``.
    """

    __slots__ = ("frame", "origin", "spacing", "shape")

    def __init__(self, frame, origin, spacing, shape):
        self.frame = np.asarray(frame, dtype=np.float64).reshape(3, 3)
        self.origin = np.asarray(origin, dtype=np.float64).reshape(3)
        self.spacing = np.asarray(spacing, dtype=np.float64).reshape(3)
        self.shape = tuple(int(value) for value in shape)

    def size(self):
        return int(np.prod(self.shape))

    def centre(self):
        """Voxel centre of the grid, in micrometres in the ``(u, d, n)`` basis."""
        return self.origin + 0.5 * (np.asarray(self.shape, np.float64) - 1.0) \
            * self.spacing @ self.frame


def grid_of_points(points, voxel_um, frame=None, margin_um=None):
    """Grid that holds all ``points`` (micrometres), optionally in ``frame``."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if frame is None:
        frame = np.eye(3, dtype=np.float64)
    local = points @ frame.T
    low, high = local.min(axis=0), local.max(axis=0)
    if margin_um is not None:
        low = low - float(margin_um)
        high = high + float(margin_um)
    spacing = np.full(3, float(voxel_um))
    shape = np.maximum(1, np.ceil((high - low) / spacing).astype(int) + 1)
    return Grid(frame, low, spacing, shape)


# --------------------------------------------------------------------------- #
# Forward model
# --------------------------------------------------------------------------- #


def page_to_object(axis, dx, dz, theta, s, t):
    """Object-frame positions (micrometres) of the pixels of one page.

    ``s`` and ``t`` are the lateral and depth pixel coordinates (arrays of any
    shape); ``theta`` is that page's rotation angle in radians. The object frame
    is the lab frame at ``theta = 0``, so the axis appears tilted inside it.
    """
    q = axis.point_um(dx, dz)
    s_px = np.asarray(s, dtype=np.float64)
    t_px = np.asarray(t, dtype=np.float64)
    y = np.stack([s_px * dx, t_px * dz, np.zeros_like(s_px)], axis=-1)
    rot = rotation_matrix(axis.unit_vector(), -float(theta))
    return q + (y - q) @ rot.T


def object_to_page(axis, dx, dz, theta, points):
    """Pixels of one page for object-frame ``points``; also the plane offset.

    Returns ``(s, t, off_plane_um)``. ``off_plane_um`` is how far the point sits
    from the page's plane along ``n``: zero means that page really images it.
    """
    q = axis.point_um(dx, dz)
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    rot = rotation_matrix(axis.unit_vector(), float(theta))
    y = q + (points - q) @ rot.T
    return y[:, 0] / dx, y[:, 1] / dz, y[:, 2]


def page_angles(count, direction=1.0, first_deg=0.0):
    """Rotation angle of every page of one turn, in radians."""
    count = int(count)
    return (math.radians(float(first_deg)) + float(direction) * 2.0 * math.pi
            * np.arange(count, dtype=np.float64) / float(max(1, count)))


def _spacing_from(voxel_um):
    """Voxel edge of the three grid axes from a scalar or a ``(lateral, axial)``.

    The first two grid axes span the plane across the rotation axis (the XY slice
    of the volume) and the third one is the axis itself, so a two-number request
    is read as "one voxel across, another one along".
    """
    requested = np.asarray(voxel_um, dtype=np.float64).reshape(-1)
    if requested.size >= 3:
        spacing = requested[:3].astype(np.float64).copy()
    elif requested.size == 2:
        spacing = np.array([requested[0], requested[0], requested[1]],
                           dtype=np.float64)
    elif requested.size == 1:
        spacing = np.full(3, float(requested[0]))
    else:
        spacing = np.full(3, 30.0)
    return np.maximum(spacing, 1e-6)


def default_grid(axis, dx, dz, x_pixels, z_pixels, voxel_um=30.0,
                 along_axis_z=True, row_window=None, angle_samples=24,
                 margin_um=None):
    """Grid that holds everything the pages can contain around ``axis``.

    ``voxel_um`` is one edge for all three grid axes, or ``(lateral, axial)``: the
    two axes across the rotation axis (the XY plane of the volume) share the
    lateral voxel and the axis itself takes the axial one. The two pixel pitches
    of a recording are not the same size - 10 um along the B-line, ~6.6 um in
    depth - so a single number for both would over-sample one of them.

    The pages sweep the B-scan rectangle around the axis, so what they can ever
    reach is that rectangle's *solid of revolution*: exactly the rectangle
    projected on the axis in depth, and a disc of radius ``rho_max`` around the
    axis across it. The grid is sized from that and not from a few sampled
    rotation angles, because a sampled box misses the extremes in between and then
    quietly discards every page that falls outside it - which is what an axis
    lying *in* the B-scan plane (tilt 90 deg) used to do here.

    ``row_window`` is the depth rows that carry data (``(first, stop)``); those are
    the rows the pages of any axis pose have to fit.
    """
    frame = frame_from_axis(axis, along_axis_z)
    unit = axis.unit_vector()
    q = axis.point_um(dx, dz)
    rows = ((0.0, float(z_pixels) - 1.0) if row_window is None
            else (float(row_window[0]), float(row_window[1]) - 1.0))
    corners = np.array([[s, t] for s in (0.0, float(x_pixels) - 1.0)
                        for t in rows], dtype=np.float64)
    # One page already shows the whole revolve: where a pixel lands does not depend
    # on the page angle, only its radius and its height above the pierce point do.
    points = page_to_object(axis, dx, dz, 0.0, corners[:, 0], corners[:, 1])
    delta = points - q
    axial = delta @ unit
    radial = delta - axial[:, None] * unit
    rho_max = float(np.linalg.norm(radial, axis=1).max())
    axial_low, axial_high = float(axial.min()), float(axial.max())
    spacing = _spacing_from(voxel_um)
    margin = float(np.min(spacing)) if margin_um is None else float(margin_um)
    centre = q + 0.5 * (axial_low + axial_high) * unit
    if along_axis_z:
        # The third grid axis is the rotation axis, so the first two have to hold
        # the disc around it and the axis runs through the middle of the volume.
        size = np.array([2.0 * (rho_max + margin), 2.0 * (rho_max + margin),
                         2.0 * (0.5 * (axial_high - axial_low) + margin)])
        shape = np.maximum(1, np.ceil(size / spacing).astype(int) + 1)
        origin = centre - (0.5 * (shape - 1) * spacing) @ frame
        return Grid(frame, origin, spacing, shape)
    # Any other frame: bound the revolve there, per axis.
    low = np.empty(3)
    for position in range(3):
        along = float(frame[position] @ unit)
        span = (abs(along) * 0.5 * (axial_high - axial_low)
                + rho_max * math.sqrt(max(0.0, 1.0 - along * along)))
        low[position] = float(frame[position] @ centre) - span - margin
    size = np.array([2.0 * (float(frame[position] @ centre) - low[position])
                     for position in range(3)])
    shape = np.maximum(1, np.ceil(size / spacing).astype(int) + 1)
    return Grid(frame, low, spacing, shape)




# --------------------------------------------------------------------------- #
# Multi-turn stacking: align the turns, then average the same phase
# --------------------------------------------------------------------------- #


def segment_spans(plan, page_count, turn_pages, min_fraction=0.9, notes=None):
    """Full turns of the segmentation, as ``(start, length)`` in pages.

    ``plan`` is what stage 1 returns. A segment is kept only when it really holds
    a whole turn (``min_fraction`` of it): a partial turn would leave a gap of
    phases that the averaging cannot fill.

    ``notes`` (a list, when given) is filled with one line per segment of the plan:
    its B-line range, how long it is and whether it was kept. A segment that is
    dropped here - a short one, or one that runs past the end of the file - is
    otherwise invisible in the stack that is averaged, and a plan whose segments
    do not line up with the B-lines the file holds is exactly what has to be seen.
    """
    starts = list((plan or {}).get("offsets") or [])
    lengths = list((plan or {}).get("lengths") or [])
    minimum = int(round(float(min_fraction) * int(turn_pages)))
    spans = []
    for index, start in enumerate(starts):
        length = int(lengths[index]) if index < len(lengths) else int(turn_pages)
        usable = min(int(length), int(turn_pages),
                     max(0, int(page_count) - int(start)))
        kept = usable >= minimum
        if notes is not None:
            notes.append(
                f"#{index + 1} B-lines {int(start)}..{int(start) + max(0, length - 1)}"
                f" ({length} page(s))"
                + ("" if kept else f": shorter than {minimum} page(s), not averaged"))
        if kept:
            spans.append((int(start), usable))
    if not spans:
        fallback = [(0, min(int(turn_pages), int(page_count)))]
        if notes is not None:
            notes.append(
                f"no segment of the plan holds {minimum} page(s): falling back to "
                f"B-lines {fallback[0][0]}..{fallback[0][1] - 1} of the file")
        spans = fallback
    return spans


def _signature(page, step=4):
    """Small, high-passed, unit-length signature of one page (for alignment).

    The smooth part of a page looks the same at every phase, so the mean of every
    row and of every column is taken out first: what is left is the fine structure,
    which is what changes from page to page and therefore what can tell whether two
    pages show the same direction.
    """
    data = np.array(page[::step, ::step], dtype=np.float64, copy=True)
    if data.size == 0:
        return data.ravel()
    data -= data.mean(axis=1, keepdims=True)
    data -= data.mean(axis=0, keepdims=True)
    flat = data.ravel()
    norm = float(np.linalg.norm(flat))
    return flat / norm if norm > 0.0 else flat


def refine_offsets(data, spans, turn_pages, max_shift=2, sample_pages=24,
                   progress=None):
    """Nudge every turn onto the reference phase; returns ``(spans, matches)``.

    Stage 1 finds the turns with a coarse signature, so a start can sit a page or
    two off - and one page is a third of a degree, exactly the error the
    reconstruction turns into a smear. Every start is probed over ``+-max_shift``
    pages and the best one is kept; the reference is the longest segment, so the
    alignment does not depend on where the file happens to start.
    """
    reference = max(range(len(spans)), key=lambda index: spans[index][1])
    step = max(1, int(turn_pages) // max(1, int(sample_pages)))
    phases = list(range(0, int(turn_pages), step))
    reference_signatures = [_signature(np.asarray(data[spans[reference][0] + phase]))
                            for phase in phases]
    refined = []
    matches = []
    for index, (start, length) in enumerate(spans):
        if index == reference:
            refined.append((start, length))
            matches.append(1.0)
            continue
        best = (None, -np.inf)
        for shift in range(-int(max_shift), int(max_shift) + 1):
            candidate = int(start) + shift
            if candidate < 0 or candidate + int(turn_pages) > int(data.shape[0]):
                continue
            score = 0.0
            for phase, signature in zip(phases, reference_signatures):
                score += float(np.dot(signature,
                                      _signature(np.asarray(data[candidate + phase]))))
            if score > best[1]:
                best = (candidate, score)
        if best[0] is None:
            refined.append((int(start), length))
            matches.append(float("nan"))
        else:
            refined.append((int(best[0]), length))
            matches.append(best[1] / float(max(1, len(phases))))
    if progress is not None:
        progress(len(spans), len(spans), "turns aligned")
    return refined, matches


def average_turns(data, spans, turn_pages, row_window=None, sample_pages=24):
    """Average every turn at the same in-turn phase; ``(pages, counts, match)``.

    Page ``k`` of every turn shows the same direction, so averaging those pages
    multiplies the SNR of each view; and because a splat reconstruction is linear,
    averaging first and reconstructing once gives exactly the same volume as
    reconstructing every turn and averaging afterwards - for a fraction of the
    time. ``counts`` says how many turns fed each phase and ``match`` how well they
    agreed, which is where a wrong alignment or a drifting spindle shows up first.
    ``pages`` is float32: a turn of 1082 pages of 1200 x 250 rows is 1.3 GB.
    """
    row_count = (int(data.shape[2]) if row_window is None
                 else int(row_window[1]) - int(row_window[0]))
    rows = slice(None) if row_window is None else slice(int(row_window[0]),
                                                       int(row_window[1]))
    starts = [int(start) for start, _length in spans
              if int(start) + int(turn_pages) <= int(data.shape[0])]
    if not starts:
        return None, None, 0.0
    pages = np.zeros((int(turn_pages), int(data.shape[1]), row_count),
                     dtype=np.float32)
    counts = np.zeros(int(turn_pages), dtype=np.int32)
    for start in starts:
        pages += np.asarray(data[start:start + int(turn_pages), :, rows],
                            dtype=np.float32)
        counts += 1
    pages /= np.maximum(counts, 1)[:, None, None]
    step = max(1, int(turn_pages) // max(1, int(sample_pages)))
    agreement = []
    for phase in range(0, int(turn_pages), step):
        page = pages[phase].astype(np.float64)
        for start in starts:
            agreement.append(correlation(page, np.asarray(data[start + phase, :, rows],
                                                          dtype=np.float64)))
    return pages, counts, (float(np.mean(agreement)) if agreement else 0.0)


# --------------------------------------------------------------------------- #
# Reconstruction: trilinear splat of every page
# --------------------------------------------------------------------------- #


def pixel_axes(page_shape, decimate=(1, 1)):
    """Pixel indices of a page along the lateral and the depth axis."""
    rows, columns = int(page_shape[0]), int(page_shape[1])
    step_s = max(1, int(decimate[0]))
    step_t = max(1, int(decimate[1]))
    return (np.arange(0, rows, step_s, dtype=np.float64),
            np.arange(0, columns, step_t, dtype=np.float64))


def _splat(total, weight, grid, coords, values):
    """Trilinear scatter-add of ``values`` at ``coords`` (micrometres)."""
    n0, n1, n2 = grid.shape
    position = ((coords - grid.origin) @ grid.frame.T) / grid.spacing
    base = np.floor(position)
    frac = position - base
    base = base.astype(np.int64)
    flat_total = total.reshape(-1)
    flat_weight = weight.reshape(-1)
    values = np.asarray(values, dtype=np.float64)
    for di in (0, 1):
        wi = frac[:, 0] if di else 1.0 - frac[:, 0]
        for dj in (0, 1):
            wj = frac[:, 1] if dj else 1.0 - frac[:, 1]
            for dk in (0, 1):
                wk = frac[:, 2] if dk else 1.0 - frac[:, 2]
                index = base + (di, dj, dk)
                inside = ((index[:, 0] >= 0) & (index[:, 0] < n0)
                          & (index[:, 1] >= 0) & (index[:, 1] < n1)
                          & (index[:, 2] >= 0) & (index[:, 2] < n2))
                if not bool(inside.any()):
                    continue
                brick = index[inside]
                weights = (wi * wj * wk)[inside]
                flat = (brick[:, 0] * n1 + brick[:, 1]) * n2 + brick[:, 2]
                np.add.at(flat_total, flat, weights * values[inside])
                np.add.at(flat_weight, flat, weights)


def reconstruct(pages, angles, axis, dx, dz, grid, decimate=(1, 1), page_step=1,
                row_offset=0.0, progress=None):
    """Resample ``pages`` into ``grid``; returns ``(volume, weight)``.

    Every page is placed with the current axis pose, so a wrong pose mixes
    neighbouring planes and blurs out-of-plane detail - exactly what
    :func:`sharpness` measures. ``weight`` counts the samples that landed in a
    voxel (zero outside the scanned region) and the volume is ``sum / weight``.
    ``row_offset`` is the depth row of the file that ``pages[:, :, 0]`` holds, so
    a caller that reconstructs only the measured rows keeps the same object frame
    as a caller that passes the whole B-line.
    """
    total = np.zeros(grid.shape, dtype=np.float64)
    weight = np.zeros(grid.shape, dtype=np.float64)
    angles = list(angles)
    count = len(angles)
    step = max(1, int(page_step))
    step_s = max(1, int(decimate[0]))
    step_t = max(1, int(decimate[1]))
    s_axis, t_axis = pixel_axes(np.asarray(pages[0]).shape, decimate)
    t_axis = t_axis + float(row_offset)
    mesh_s, mesh_t = np.meshgrid(s_axis, t_axis, indexing="ij")
    flat_s, flat_t = mesh_s.ravel(), mesh_t.ravel()
    for index in range(0, count, step):
        if progress is not None:
            progress(index, count, f"page {index + 1} of {count}")
        page = np.asarray(pages[index])[::step_s, ::step_t]
        coords = page_to_object(axis, dx, dz, angles[index], flat_s, flat_t)
        _splat(total, weight, grid, coords, page.ravel())
    volume = np.zeros(grid.shape, dtype=np.float64)
    filled = weight > 0.0
    volume[filled] = total[filled] / weight[filled]
    return volume, weight


# --------------------------------------------------------------------------- #
# Axis calibration: how sharp does the volume get for a given pose?
# --------------------------------------------------------------------------- #


def weight_mask(weight, quantile=0.5):
    """Voxels sampled at least as well as that quantile of the sampled voxels.

    The weight of a voxel counts the page samples that landed in it, and a tilted
    geometry spreads that count over an order of magnitude, so the mask is taken
    from a quantile of the non-zero weights: a fraction of the *maximum* would keep
    only the handful of voxels the pages happen to cross most often and leave the
    rest of the sample out of every comparison.
    """
    weight = np.asarray(weight, dtype=np.float64)
    sampled = weight[weight > 0.0]
    if sampled.size == 0:
        return np.zeros(weight.shape, dtype=bool)
    return weight >= float(np.quantile(sampled, float(quantile)))


def sharpness(volume, mask=None):
    """Gradient energy of a reconstruction: high when the axis pose is right.

    The three grid axes are summed so the metric has no preferred orientation
    and it is averaged over the compared voxel pairs only, so an empty rim does
    not dilute it. A wrong pose mixes neighbouring planes and drops this number.
    """
    data = np.asarray(volume, dtype=np.float64)
    if mask is None:
        mask = np.ones(data.shape, dtype=bool)
    else:
        mask = np.asarray(mask, dtype=bool)
    total = 0.0
    count = 0
    for axis in range(3):
        diff = np.diff(data, axis=axis)
        if not diff.size:
            continue
        size = data.shape[axis]
        first = np.take(mask, np.arange(0, size - 1), axis=axis)
        second = np.take(mask, np.arange(1, size), axis=axis)
        pair = np.logical_and(first, second)
        total += float(np.sum(diff[pair] ** 2))
        count += int(pair.sum())
    return total / float(max(1, count))


def _page_match(measured, model):
    """Fraction of a page's energy that the volume explains (1 - normalised SSE).

    Deliberately not a correlation: a pose with too small a tilt reconstructs only
    part of the sample, and a correlation over the page would then score it badly
    and a *larger* tilt always better (more of the page is covered). The residual
    punishes both mistakes - the uncovered part, and the ghosting a too large tilt
    produces - so the peak is the pose that reproduces the recording.
    """
    denominator = float(np.sum(measured * measured))
    if denominator <= 0.0:
        return 0.0
    error = measured - model
    return 1.0 - float(np.sum(error * error)) / denominator


def _score_and_volume(pages, angles, axis, dx, dz, grid, decimate=(4, 4),
                      page_step=1, row_offset=0.0, probe_pages=24, progress=None):
    """``(page match, volume, weight)`` for one axis pose.

    The volume is rebuilt for the pose and then asked to *predict* the recording:
    a few pages are re-sampled out of it with the same geometry and compared with
    what was measured (:func:`_page_match`). A pose that is off smears the volume,
    the smeared volume predicts every page worse, and the mean match drops - which
    is a far better criterion than the gradient of the volume, because ghosting
    from a wrong pose can make a volume look *sharper* than the truth.
    """
    volume, weight = reconstruct(pages, angles, axis, dx, dz, grid,
                                 decimate=decimate, page_step=page_step,
                                 row_offset=row_offset, progress=progress)
    step_s = max(1, int(decimate[0]))
    step_t = max(1, int(decimate[1]))
    shape = np.asarray(pages[0]).shape
    probe_s = np.arange(0, int(shape[0]), step_s, dtype=np.float64)
    probe_t = np.arange(0, int(shape[1]), step_t, dtype=np.float64) \
        + float(row_offset)
    mesh_s, mesh_t = np.meshgrid(probe_s, probe_t, indexing="ij")
    flat_s, flat_t = mesh_s.ravel(), mesh_t.ravel()
    indices = list(range(0, len(angles), max(1, int(page_step))))
    if 1 < int(probe_pages) < len(indices):
        take = np.linspace(0, len(indices) - 1, int(probe_pages)).round()
        indices = [indices[int(value)] for value in take]
    scores = []
    for index in indices:
        measured = np.asarray(pages[index])[::step_s, ::step_t].ravel()
        model = sample_trilinear(volume, grid,
                                 page_to_object(axis, dx, dz, angles[index],
                                                flat_s, flat_t))
        scores.append(_page_match(measured, model))
    return (float(np.mean(scores)) if scores else 0.0), volume, weight


def consistency_score(pages, angles, axis, dx, dz, grid, decimate=(4, 4),
                      page_step=1, row_offset=0.0, probe_pages=24):
    """How well a pose explains the recording (see :func:`_score_and_volume`)."""
    return _score_and_volume(pages, angles, axis, dx, dz, grid,
                             decimate=decimate, page_step=page_step,
                             row_offset=row_offset, probe_pages=probe_pages)[0]


def focus_curve(pages, angles, dx, dz, grid, axis, parameter="tilt_deg",
                values=(0.0, 2.0, 4.0, 6.0, 8.0, 10.0), decimate=(4, 4),
                page_step=1, row_offset=0.0, progress=None):
    """Page match for a list of one-parameter axis poses (a scan of the metric).

    ``progress`` reports every trial as ``(count, total, text)``.
    """
    current = float(getattr(axis, parameter))
    total = len(values)
    curve = []
    for index, value in enumerate(values):
        if progress is not None:
            progress(index + 1, total,
                     f"{parameter} {float(value):.2f} ({index + 1}/{total})")
        pose = axis.adjusted(parameter, float(value) - current)
        curve.append((float(value),
                      consistency_score(pages, angles, pose, dx, dz, grid,
                                        decimate=decimate,
                                        page_step=page_step,
                                        row_offset=row_offset)))
    return curve


def autofocus(pages, angles, dx, dz, grid, axis,
              parameters=("centre_s", "centre_t", "tilt_deg", "azimuth_deg"),
              steps=(8.0, 8.0, 4.0, 30.0), rounds=6, shrink=0.5,
              decimate=(4, 4), page_step=1, row_offset=0.0, limits=None,
              tolerance=0.0, progress=None):
    """Find the axis pose that explains the recording best.

    Plain coordinate descent: every round each parameter is probed both ways, the
    best probe is kept, and a parameter whose probes all fail has its step
    halved. The objective is :func:`consistency_score` of a *cheap* reconstruction
    (decimated grid and pixels), so this is a focus loop and not the final render;
    the returned volume is the best cheap render seen on the way. Returns a dict
    with ``axis``, ``score``, ``volume``, ``weight`` and ``history``
    (``(round, axis, score)``).

    ``limits`` is an optional ``{parameter: (low, high)}`` map for parameters that
    have to stay inside a physical range; a probe outside it is simply not tried.
    ``offset_n_um`` needs that: with the tilt at exactly 90 degrees the objective is
    symmetric in it (the two directions image the same points mirrored), so an
    unbounded descent can walk it anywhere without ever being corrected.

    ``tolerance`` is how much a probe has to add to the score before it is accepted
    (see :data:`FIT_TOLERANCE`): on a cheap reconstruction the tail of the objective
    is jitter, and without it the pose drifts on numbers that mean nothing.

    ``progress`` reports every probe of the descent as ``(count, total, text)``.
    """
    parameters = list(parameters)
    steps = [float(value) for value in steps]
    floors = [max(0.05, 0.05 * value) for value in steps]
    limits = dict(limits or {})

    def allowed(pose, name):
        if name not in limits:
            return True
        low, high = limits[name]
        return float(low) <= float(getattr(pose, name)) <= float(high)

    def score(pose):
        return _score_and_volume(pages, angles, pose, dx, dz, grid,
                                 decimate=decimate, page_step=page_step,
                                 row_offset=row_offset)

    # Every probe is one cheap reconstruction, so the report counts probes and the
    # budget is the most the descent can do: two directions per parameter per round.
    budget = 1 + 2 * len(parameters) * max(0, int(rounds))
    probes = [0]

    def probe(pose, text):
        probes[0] += 1
        if progress is not None:
            progress(min(probes[0], budget), budget, text)
        return score(pose)

    value, volume, weight = probe(axis, "the starting pose")
    best = axis.copy()
    history = [(0, best.copy(), value)]
    for round_index in range(1, int(rounds) + 1):
        improved = False
        for position, name in enumerate(parameters):
            if steps[position] <= floors[position]:
                continue
            for sign in (1.0, -1.0):
                trial = best.adjusted(name, sign * steps[position])
                if not allowed(trial, name):
                    continue
                probe_score, probe_volume, probe_weight = probe(
                    trial,
                    f"round {round_index}: {name} {sign * steps[position]:+.2f}")
                if probe_score > value + float(tolerance):
                    best, value = trial, probe_score
                    volume, weight = probe_volume, probe_weight
                    improved = True
                    break
            else:
                steps[position] = 0.5 * steps[position]
        history.append((round_index, best.copy(), value))
        if not improved and all(step <= floor for step, floor
                                in zip(steps, floors)):
            break
    return {"axis": best, "score": value, "volume": volume,
            "weight": weight, "history": history}


# --------------------------------------------------------------------------- #
# Half turns and the small automatic tilt
# --------------------------------------------------------------------------- #


def half_turn_spans(turn_pages):
    """The two halves of one turn, as ``(start, stop)`` page ranges."""
    count = max(2, int(turn_pages))
    half = count // 2
    return [(0, half), (half, count)]


def fine_tilt(pages, angles, dx, dz, grid, axis, span_deg=2.0, step_deg=0.5,
              decimate=(4, 4), page_step=1, row_offset=0.0, progress=None):
    """Best tilt within ``+-span_deg`` of the pose, by page match; ``(axis, curve)``.

    The rig puts the axis in the B-scan plane (90 degrees from the plane normal) and
    only the last degree or two are left to fit, so the search is kept inside a small
    window on purpose: a bounded scan cannot wander into a wrong local optimum the
    way an unbounded descent can, and the returned curve shows whether the window was
    wide enough (a best value sitting on either end means it was not).
    """
    base = float(axis.tilt_deg)
    values = list(np.arange(-float(span_deg),
                            float(span_deg) + 0.5 * float(step_deg),
                            float(step_deg)) + base)
    curve = focus_curve(pages, angles, dx, dz, grid, axis, "tilt_deg", values=values,
                        decimate=decimate, page_step=page_step,
                        row_offset=row_offset, progress=progress)
    best, _score = max(curve, key=lambda item: item[1])
    return axis.adjusted("tilt_deg", float(best) - base), curve


def combine_halves(halves):
    """Weight-weighted mean of half-turn volumes: ``(volume, weight)``.

    The splat sums the samples and their weights, so the sum of the two halves' sums
    divided by the sum of their weights is exactly what reconstructing the whole turn
    in one go gives.
    """
    total = None
    weight = None
    for half in halves:
        values = np.asarray(half["volume"], dtype=np.float64)
        weights = np.asarray(half["weight"], dtype=np.float64)
        if total is None:
            total = values * weights
            weight = weights.copy()
        else:
            total = total + values * weights
            weight = weight + weights
    if total is None:
        return None, None
    volume = np.zeros_like(total)
    filled = weight > 0.0
    volume[filled] = total[filled] / weight[filled]
    return volume, weight


def reconstruct_halves(pages, angles, axis, dx, dz, grid, decimate=(1, 1),
                       page_step=1, row_offset=0.0, progress=None):
    """Reconstruct the two half-turns separately; ``(halves, combined)``.

    A full turn images every object point twice, at ``theta`` and at ``theta + 180``
    degrees, so the two halves are two independent looks at the same sample. Where
    they agree the structure is real; where they disagree the difference is the
    residual pose error, or a feature only one half ever saw.

    ``progress`` runs the two halves as one bar: the second half picks the count up
    where the first one left it, and every report names the B-lines it belongs to.
    """
    pages = np.asarray(pages)
    count = int(pages.shape[0])
    step = max(1, int(page_step))
    spans = []
    for start, stop in half_turn_spans(count):
        start = max(0, min(count - 1, start))
        stop = max(start + 1, min(count, stop))
        spans.append((start, stop))
    visited = [len(range(start, stop, step)) for start, stop in spans]
    total = max(1, int(sum(visited)))
    done = [0]
    halves = []
    for start, stop in spans:
        def report(index, _count, text, start=start, stop=stop):
            if progress is not None:
                progress(done[0] + int(index), total,
                         f"B-lines {start + 1}-{stop}: {text}")

        match, volume, weight = _score_and_volume(
            pages[start:stop], angles[start:stop], axis, dx, dz, grid,
            decimate=decimate, page_step=page_step, row_offset=row_offset,
            progress=report)
        halves.append({"start": int(start), "stop": int(stop), "volume": volume,
                       "weight": weight, "match": float(match)})
        done[0] += visited[len(halves) - 1]
    return halves, combine_halves(halves)


def halves_agreement(halves, quantile=0.5):
    """Correlation between the two half-turn volumes over the voxels both filled."""
    if halves is None or len(halves) < 2:
        return 0.0
    first, second = halves[0], halves[1]
    mask = np.logical_and(weight_mask(first["weight"], quantile),
                          weight_mask(second["weight"], quantile))
    return correlation(first["volume"], second["volume"], mask)


def halves_radial_split(halves, grid, sectors=12, z_slices=5, band_um=None):
    """Radial difference between two half-turn volumes, and its cosine amplitude.

    A wrong lateral axis position (``centre_s``) does not move the two half-turns
    together: the two observations of a point sit at *opposite ends* of the B-line,
    so an error of ``du`` shifts them opposite ways, by ``2*du``, along a direction
    that turns with the azimuth of the material. On a shell that reads as

        delta_rho(phi) = A*cos(phi) + B*sin(phi) + C,    sqrt(A^2+B^2) = 2*du

    which is what makes this amplitude the objective for the axis position: measured
    on a real recording it gave 78 / 29 / 119 um for ``centre_s`` at -95 / optimum /
    +25 um, while the page match cannot see the same error at all. The constant term
    soaks up a common radius error and the fitted sinusoid rejects the page noise.

    Returns a dict with the per-sector radii of both halves, their difference, the
    band that was used and the amplitude/phase of the fitted sinusoid.
    """
    if halves is None or len(halves) < 2:
        raise ValueError("two half-turn volumes are needed")
    # The radii live in the plane across the axis, so the axial voxel (which is
    # finer) must not enter the step from a radius to a voxel index.
    edges = np.asarray(grid.spacing, dtype=np.float64).reshape(-1)
    spacing = float(np.mean(edges[:2])) if edges.size >= 2 else float(edges[0])
    first = np.asarray(halves[0]["volume"], dtype=np.float64)
    second = np.asarray(halves[1]["volume"], dtype=np.float64)
    shape = first.shape
    if min(shape) < 3:
        raise ValueError("the volumes are too small to look at shells")
    # The axis runs through the middle of the grid (see default_grid), and the
    # azimuth directions live in the first two grid axes.
    mid = 0.5 * (np.asarray(shape[:2], dtype=np.float64) - 1.0)
    z_index = np.unique(np.clip(
        np.rint(np.linspace(0.35, 0.65, max(1, int(z_slices))) * (shape[2] - 1))
        .astype(int), 0, shape[2] - 1))
    radii = np.arange(1.0, max(2.0, 0.5 * float(min(shape[0], shape[1])))) * spacing
    angle = 2.0 * math.pi * np.arange(int(sectors)) / float(max(1, int(sectors)))
    step = radii / spacing
    rows = np.clip(np.rint(mid[0] + np.outer(np.cos(angle), step)), 0,
                   shape[0] - 1).astype(int)
    columns = np.clip(np.rint(mid[1] + np.outer(np.sin(angle), step)), 0,
                      shape[1] - 1).astype(int)

    def radial_of(volume):
        out = np.zeros(radii.size, dtype=np.float64)
        for z in z_index:
            out += np.mean(np.maximum(volume[rows, columns, z],
                                      second[rows, columns, z]), axis=0)
        return out / float(z_index.size)

    band = band_um
    if band is None:
        # The dominant shell decides the band: an inner surface or the noise floor
        # must not drag the centroid around.
        radial = np.convolve(radial_of(first), np.ones(3) / 3.0, mode="same")
        peak = float(radii[int(np.argmax(radial))])
        band = (0.85 * peak, 1.15 * peak)
    keep = (radii >= float(band[0])) & (radii <= float(band[1]))
    if not bool(keep.any()):
        keep = np.ones(radii.size, dtype=bool)
    band_radii = radii[keep]

    def radius_of(volume):
        per_slice = []
        for z in z_index:
            values = np.asarray(volume[rows, columns, z], dtype=np.float64)[:, keep]
            cut = 0.5 * values.max(axis=1, keepdims=True)
            weight = np.where(values > cut, values - cut, 0.0)
            total = weight.sum(axis=1)
            with np.errstate(invalid="ignore", divide="ignore"):
                per_slice.append((weight * band_radii).sum(axis=1)
                                 / np.where(total > 0.0, total, np.nan))
        if not per_slice:
            return np.full(int(sectors), np.nan)
        return np.nanmean(np.stack(per_slice, axis=0), axis=0)

    rho_first = radius_of(first)
    rho_second = radius_of(second)
    delta = rho_first - rho_second
    good = np.isfinite(delta)
    amplitude = float("nan")
    phase = float("nan")
    coefficients = [float("nan"), float("nan"), float("nan")]
    if int(good.sum()) >= max(4, int(sectors) // 2):
        design = np.stack([np.cos(angle[good]), np.sin(angle[good]),
                           np.ones(int(good.sum()))], axis=1)
        coefficients, *_ = np.linalg.lstsq(design, delta[good], rcond=None)
        coefficients = [float(value) for value in coefficients]
        amplitude = float(np.hypot(coefficients[0], coefficients[1]))
        phase = float(math.degrees(math.atan2(coefficients[1], coefficients[0])))
    return {"amplitude_um": amplitude, "phase_deg": phase,
            "coefficients": coefficients, "delta_um": delta.tolist(),
            "first_um": rho_first.tolist(), "second_um": rho_second.tolist(),
            "band_um": [float(band[0]), float(band[1])],
            "sectors": int(sectors), "z_slices": int(z_index.size)}


def _box_mean(volume, half_width, axis):
    """Moving average along one axis, numpy only (integral image, O(n))."""
    half_width = max(0, int(half_width))
    data = np.asarray(volume, dtype=np.float64)
    if half_width == 0:
        return data
    width = 2 * half_width + 1
    pad = [(0, 0), (0, 0), (0, 0)]
    pad[int(axis)] = (half_width, half_width)
    padded = np.pad(data, pad, mode="edge")
    cumulative = np.cumsum(padded, axis=int(axis))
    zero = np.zeros_like(np.take(cumulative, [0], axis=int(axis)))
    cumulative = np.concatenate([zero, cumulative], axis=int(axis))
    length = int(padded.shape[int(axis)]) - width + 1
    upper = np.take(cumulative, np.arange(width, width + length), axis=int(axis))
    lower = np.take(cumulative, np.arange(0, length), axis=int(axis))
    return (upper - lower) / float(width)


def halves_mismatch(halves, grid, smooth_voxels=2.0, quantile=0.5):
    """How far apart the two half-turns sit: the *structured* part of their difference.

    The two halves image the same sample from opposite sides, so the pose terms that
    treat them differently - a wrong lateral pierce point moves them opposite ways, by
    twice the error - leave a low-frequency residual in their difference while the page
    noise stays high-frequency. The objective is the ratio of the smoothed difference
    to the raw one: it grows with the misregistration, has its minimum where the two
    halves sit on top of each other, and it assumes nothing about the sample - unlike a
    shell radius, which a solid object does not offer.
    """
    if halves is None or len(halves) < 2:
        raise ValueError("two half-turn volumes are needed")
    first = np.asarray(halves[0]["volume"], dtype=np.float64)
    second = np.asarray(halves[1]["volume"], dtype=np.float64)
    mask = np.logical_and(weight_mask(halves[0]["weight"], quantile),
                          weight_mask(halves[1]["weight"], quantile))
    if int(np.count_nonzero(mask)) < 8:
        return {"structured": float("nan"), "raw": float("nan"), "ratio": float("nan")}
    difference = np.where(mask, first - second, 0.0)
    raw = float(np.sqrt(np.mean(difference[mask] ** 2)))
    smoothed = difference
    for axis in range(3):
        smoothed = _box_mean(smoothed, int(round(float(smooth_voxels))), axis)
    structured = float(np.sqrt(np.mean(smoothed[mask] ** 2)))
    return {"structured": structured, "raw": raw,
            "ratio": float(structured / raw) if raw > 0.0 else float("nan")}


def _parabola_vertex(points):
    """Vertex of the parabola through three points, or the middle one if unusable.

    The points may carry more than the two coordinates it needs (the scans hand over
    whole curve entries), so only the first two are used.
    """
    points = [(float(entry[0]), float(entry[1])) for entry in points]
    if len(points) < 3:
        return float(points[0][0]) if points else float("nan")
    (x1, y1), (x2, y2), (x3, y3) = points
    denom = (x1 - x2) * (x1 - x3) * (x2 - x3)
    if denom == 0.0 or not all(np.isfinite([y1, y2, y3])):
        return float(x2)
    a = (x3 * (y2 - y1) + x2 * (y1 - y3) + x1 * (y3 - y2)) / denom
    b = (x3 * x3 * (y1 - y2) + x2 * x2 * (y3 - y1) + x1 * x1 * (y2 - y3)) / denom
    if a <= 0.0:
        return float(x2)
    vertex = -b / (2.0 * a)
    if not (min(x1, x2, x3) <= vertex <= max(x1, x2, x3)):
        return float(x2)
    return float(vertex)


def centre_scan(pages, angles, dx, dz, grid, axis, span_um, step_um, decimate=(4, 4),
                page_step=4, row_offset=0.0, sectors=12, z_slices=5, band_um=None,
                progress=None):
    """Score lateral axis positions around the pose by how the two halves line up.

    The objective is the ratio :func:`halves_mismatch` returns (see there for why it
    needs no shell): smaller means the two half-turns sit closer on top of each other.
    The shell radius difference of :func:`halves_radial_split` is carried along as a
    diagnostic, because on a shell it is the sharper reading of the same error.

    Every candidate is one pair of half-turn reconstructions, so this stays affordable
    while ``page_step`` is a few pages. The grid has to be the render grid - a feature
    the voxels cannot resolve cannot be measured either.

    Returns ``{"curve": [(centre_s_px, objective, agreement, split_um), ...], ...}``
    with the best entry, plus the radius band that was used for the diagnostic.
    """
    step_um = max(1e-6, float(step_um))
    half_count = int(round(abs(float(span_um)) / step_um))
    offsets = np.arange(-half_count, half_count + 1, 1, dtype=np.float64) * step_um
    start_s = float(axis.centre[0])
    curve = []
    band = band_um
    for index, offset in enumerate(offsets):
        pose = axis.adjusted("centre_s", offset / float(dx))
        # Splatting only: the page match is not what this scan is about, and computing it
        # here would make every candidate cost as much as a full comparison render.
        halves = []
        for start_page, stop_page in half_turn_spans(len(pages)):
            volume, weight = reconstruct(
                pages[start_page:stop_page], angles[start_page:stop_page], pose,
                dx, dz, grid, decimate=decimate, page_step=page_step,
                row_offset=row_offset)
            halves.append({"start": int(start_page), "stop": int(stop_page),
                           "volume": volume, "weight": weight})
        mismatch = halves_mismatch(halves, grid)
        try:
            split = halves_radial_split(halves, grid, sectors=sectors,
                                        z_slices=z_slices, band_um=band)
            if band is None:
                band = split["band_um"]
            split_um = float(split["amplitude_um"])
        except ValueError:
            split_um = float("nan")
        curve.append((float(pose.centre[0]), float(mismatch["ratio"]),
                      float(halves_agreement(halves)), split_um))
        if progress is not None:
            progress(index + 1, len(offsets), f"centre scan at {pose.centre[0]:.1f} px")
    usable = [entry for entry in curve if np.isfinite(entry[1])]
    best = min(usable, key=lambda entry: entry[1]) if usable else curve[0]
    return {"curve": curve, "start_centre_s": start_s,
            "best_centre_s": float(best[0]), "best_ratio": float(best[1]),
            "best_agreement": float(best[2]), "best_split_um": float(best[3]),
            "edge": bool(best is curve[0] or best is curve[-1]),
            "band_um": band, "step_um": step_um, "offsets_um": offsets.tolist()}


def calibrate_centre_s(pages, angles, dx, dz, grid, axis,
                       span_um=None, step_um=None, fine_span_um=None,
                       fine_step_um=None, decimate=(4, 4),
                       page_step=4, row_offset=0.0, sectors=12, z_slices=5,
                       tolerance=0.05, progress=None):
    """Find the lateral axis position that makes the two half-turns agree.

    Two stages: a window that brackets the minimum, then a fine one around the vertex of
    the parabola through the first. The window defaults to
    :data:`CENTRE_SCAN_SPAN_PX` pixels either side of the pose (and the fine stage to
    :data:`CENTRE_SCAN_FINE_SPAN_PX`), because the error lives in the pixel frame and it
    is small once the mount is anywhere near right: the wider it is, the more pairs of
    half-turn reconstructions it costs.

    The objective is the mismatch ratio of :func:`halves_mismatch`, because it is the one
    metric that responds to this error at all (the page match does not, measured) and it
    needs no assumption about the sample; the shell radius split of
    :func:`halves_radial_split` is scored alongside and used instead whenever it is the
    one that separates the candidates, which it is on anything with a rim.

    The result is accepted only when it improves on the starting position by
    ``tolerance`` (a fraction), so a flat curve - nothing to align, no signal, a window
    that misses the minimum - leaves the pose alone and says so. A best value sitting at
    the edge of the window is flagged: that means the window needs to be wider. Set
    ``fine_span_um`` to 0 for the single-stage version the panel runs before every
    render. Returns a dict with the accepted ``axis``, the position it settled on, both
    objectives, both shell splits and both curves.
    """
    span_um = CENTRE_SCAN_SPAN_PX * float(dx) if span_um is None else span_um
    step_um = CENTRE_SCAN_STEP_PX * float(dx) if step_um is None else step_um
    if fine_span_um is None:
        fine_span_um = CENTRE_SCAN_FINE_SPAN_PX * float(dx)
    if fine_step_um is None:
        fine_step_um = CENTRE_SCAN_FINE_STEP_PX * float(dx)
    start_axis = axis.copy()
    start_s = float(start_axis.centre[0])
    coarse = centre_scan(pages, angles, dx, dz, grid, start_axis, span_um, step_um,
                         decimate=decimate, page_step=page_step,
                         row_offset=row_offset, sectors=sectors, z_slices=z_slices,
                         progress=progress)

    # Every candidate reports two metrics and they see different things: the shell radius
    # split is sharp on a shell (a lens, a cup - anything with a rim) and meaningless on
    # a solid, while the mismatch ratio works on anything but is limited by the page
    # noise on a real recording. So the scan goes with whichever one actually separates
    # the candidates over the window, and reports both either way.
    def contrast(values):
        finite = [float(value) for value in values if np.isfinite(value)]
        if len(finite) < 5 or min(finite) <= 0.0:
            return 0.0
        return float(max(finite) / min(finite))

    ratio_contrast = contrast([entry[1] for entry in coarse["curve"]])
    shell_contrast = contrast([entry[3] for entry in coarse["curve"]])
    column = 3 if shell_contrast > ratio_contrast else 1
    metric = ("shell radius split" if column == 3 else "two-half mismatch ratio")

    def entry_at(scan, position_px):
        """The curve entry nearest a position, so both metrics can be read together."""
        curve = scan["curve"]
        return min(curve, key=lambda entry: abs(entry[0] - float(position_px)))

    start_entry = entry_at(coarse, start_s)
    start_objective = float(start_entry[column])

    def vertex_of(scan):
        curve = scan["curve"]
        order = sorted(range(len(curve)),
                       key=lambda index: curve[index][column]
                       if np.isfinite(curve[index][column]) else float("inf"))
        if not order or not np.isfinite(curve[order[0]][column]):
            return float(scan["start_centre_s"]), curve[0]
        position = order[0]
        return _parabola_vertex(curve[max(0, position - 1):position + 2]), curve[position]

    coarse_s, best_entry = vertex_of(coarse)
    best_s = coarse_s
    fine = None
    edge = bool(coarse.get("edge"))
    if float(fine_span_um) > 0.0:
        # The fine stage repeats the same scan around the vertex, with the band of the
        # coarse one so both stages read the same feature.
        fine = centre_scan(pages, angles, dx, dz, grid,
                           start_axis.adjusted("centre_s", coarse_s - start_s),
                           float(fine_span_um), float(fine_step_um),
                           decimate=decimate, page_step=page_step,
                           row_offset=row_offset, sectors=sectors,
                           z_slices=z_slices, band_um=coarse["band_um"],
                           progress=progress)
        best_s, best_entry = vertex_of(fine)
        edge = bool(fine.get("edge"))
    best_objective = float(best_entry[column])
    accepted = (np.isfinite(best_objective) and np.isfinite(start_objective)
                and start_objective > 0.0
                and best_objective < (1.0 - float(tolerance)) * start_objective
                and max(ratio_contrast, shell_contrast) > 1.2)
    if not accepted:
        best_s = start_s
        best_entry = start_entry
        best_objective = start_objective
    return {"axis": start_axis.adjusted("centre_s", best_s - start_s),
            "accepted": bool(accepted), "start_centre_s": start_s,
            "centre_s": float(best_s), "metric": metric, "column": int(column),
            "start_objective": float(start_objective),
            "objective": float(best_objective),
            "start_ratio": float(start_entry[1]), "ratio": float(best_entry[1]),
            "start_amplitude_um": float(start_entry[3]),
            "amplitude_um": float(best_entry[3]),
            "ratio_contrast": float(ratio_contrast),
            "shell_contrast": float(shell_contrast), "edge": bool(edge),
            "coarse": coarse, "fine": fine, "band_um": coarse["band_um"],
            "tolerance": float(tolerance)}


# --------------------------------------------------------------------------- #
# Phantom of revolution, and the pages it would produce
# --------------------------------------------------------------------------- #


def grid_axis_points(grid, position):
    """Micrometre offsets of the voxels along one grid axis, shape (n, 3).

    The offsets start at the grid origin, so :func:`grid_positions` adds the
    origin exactly once.
    """
    count = grid.shape[position]
    steps = np.arange(count, dtype=np.float64) * grid.spacing[position]
    return np.outer(steps, grid.frame[position])


def grid_positions(grid):
    """Micrometre position of every voxel of ``grid``, shape ``shape + (3,)``."""
    along = [grid_axis_points(grid, position) for position in range(3)]
    return (grid.origin + along[0][:, None, None, :]
            + along[1][None, :, None, :] + along[2][None, None, :, :])


def visible_reach_um(axis, radius_um):
    """Axial half-range the pages can image at ``radius_um`` from the axis.

    A page can image an object point only where its plane passes through it, so
    with a tilted axis the pages ever see the double cone ``|z'| <= rho*tan(tilt)``
    around the axis: nothing outside it is in any B-line. The sample therefore has
    to sit far enough from the axis for the depth range of interest to fit.
    """
    return np.abs(np.asarray(radius_um, dtype=np.float64)) \
        * math.tan(math.radians(axis.tilt_deg))



def sample_trilinear(volume, grid, coords, fill=0.0):
    """Trilinear read of ``volume`` at the micrometre positions ``coords``."""
    position = ((coords - grid.origin) @ grid.frame.T) / grid.spacing
    base = np.floor(position)
    frac = position - base
    base = base.astype(np.int64)
    n0, n1, n2 = volume.shape
    data = np.asarray(volume, dtype=np.float64)
    out = np.full(len(position), float(fill), dtype=np.float64)
    for di in (0, 1):
        wi = frac[:, 0] if di else 1.0 - frac[:, 0]
        for dj in (0, 1):
            wj = frac[:, 1] if dj else 1.0 - frac[:, 1]
            for dk in (0, 1):
                wk = frac[:, 2] if dk else 1.0 - frac[:, 2]
                index = base + (di, dj, dk)
                inside = ((index[:, 0] >= 0) & (index[:, 0] < n0)
                          & (index[:, 1] >= 0) & (index[:, 1] < n1)
                          & (index[:, 2] >= 0) & (index[:, 2] < n2))
                if not bool(inside.any()):
                    continue
                brick = index[inside]
                out[inside] += (wi * wj * wk)[inside] * data[brick[:, 0],
                                                            brick[:, 1],
                                                            brick[:, 2]]
    return out


def phantom_volume(grid, axis, dx, dz, radius_um=2600.0, wall_um=100.0,
                   core_um=900.0, holder_offset_um=1500.0, speckle=0.5,
                   texture_um=600.0, visible_frac=0.9, seed=1):
    """A phantom of revolution about ``axis``: bright wall, core, planar holder.

    Everything is written in the axis frame, so the phantom is round about the
    axis *by construction* whatever the pose - which is what makes it a fair test
    of a calibration that has to find a tilted pose. ``holder_offset_um`` puts a
    bright plane perpendicular to the axis at that axial position: it appears as a
    straight line in every page, so its blur is the clearest sign of a wrong pose.

    ``visible_frac`` cuts the phantom down to the cone the pages can actually
    reach (``|z'| <= visible_frac * rho * tan(tilt)``, see :func:`visible_reach_um`)
    with a small margin, so the self-test does not ask the reconstruction for
    object that no B-line ever touched.
    """
    points = grid_positions(grid)
    direction = axis.unit_vector()
    q = axis.point_um(dx, dz)
    delta = points - q
    axial = delta @ direction
    radial = delta - axial[..., None] * direction
    r = np.sqrt(np.sum(radial * radial, axis=-1))
    rng = np.random.default_rng(int(seed))
    texture = np.zeros(r.shape, dtype=np.float64)
    for _ in range(12):
        wave = rng.normal(size=3)
        wave = wave / float(np.linalg.norm(wave))
        texture += np.cos(2.0 * math.pi * (delta @ wave) / float(texture_um)
                          + float(rng.uniform(0.0, 2.0 * math.pi)))
    texture /= 12.0
    wall = np.exp(-0.5 * ((r - float(radius_um)) / float(wall_um)) ** 2)
    core = np.exp(-0.5 * (r / float(core_um)) ** 2)
    holder = np.exp(-0.5 * ((axial - float(holder_offset_um))
                            / (2.0 * float(wall_um))) ** 2)
    volume = (wall * (1.0 + float(speckle) * texture)
              + 0.6 * core * (1.0 + 0.5 * float(speckle) * texture)
              + 1.5 * holder)
    if visible_frac:
        inside = np.abs(axial) <= float(visible_frac) * visible_reach_um(axis, r)
        volume = np.where(inside, volume, 0.0)
    return np.maximum(volume, 0.0)


def simulate_pages(volume, grid, axis, dx, dz, angles, page_shape,
                   decimate=(1, 1), noise=0.0, seed=2):
    """The pages an OCT scan would record: one plane sample per pixel.

    ``page_shape`` is ``(x_pixels, z_pixels)`` before ``decimate``, and the
    returned list holds one 2-D ``float32`` array per angle.
    """
    s_axis, t_axis = pixel_axes(page_shape, decimate)
    mesh_s, mesh_t = np.meshgrid(s_axis, t_axis, indexing="ij")
    flat_s, flat_t = mesh_s.ravel(), mesh_t.ravel()
    rng = np.random.default_rng(int(seed))
    pages = []
    for value in angles:
        coords = page_to_object(axis, dx, dz, float(value), flat_s, flat_t)
        data = sample_trilinear(volume, grid, coords)
        if noise:
            data = data + rng.normal(scale=float(noise), size=data.shape)
        pages.append(data.reshape(len(s_axis), len(t_axis)).astype(np.float32))
    return pages


def correlation(first, second, mask=None):
    """Pearson correlation of two volumes (over ``mask`` when given)."""
    a = np.asarray(first, dtype=np.float64).reshape(-1)
    b = np.asarray(second, dtype=np.float64).reshape(-1)
    if mask is not None:
        keep = np.asarray(mask, dtype=bool).reshape(-1)
        a, b = a[keep], b[keep]
    if a.size < 2:
        return 0.0
    a = a - a.mean()
    b = b - b.mean()
    norm = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b / norm) if norm > 0.0 else 0.0


# --------------------------------------------------------------------------- #
# Self-test: a tilted phantom has to come back
# --------------------------------------------------------------------------- #


def _phantom_case(tilt_deg=20.0, azimuth_deg=20.0, turn_pages=320,
                  x_pixels=200, z_pixels=150, voxel_um=30.0, dx=10.0, dz=10.0,
                  texture_um=350.0):
    """Phantom of revolution about a *tilted* axis, plus the pages it makes.

    The axis pierces the B-line near its centre. How much of the object the pages
    can ever see is set by the tilt: they reach the cone ``|z'| <= rho*tan(tilt)``
    and nothing else, so 20 degrees leaves about a millimetre of a sample that
    sits a couple of millimetres off the axis - enough for a real test of both the
    reconstruction and the pose search.
    """
    axis = AxisPose((0.5 * x_pixels, 0.5 * z_pixels), tilt_deg, azimuth_deg)
    grid = default_grid(axis, dx, dz, x_pixels, z_pixels, voxel_um=voxel_um)
    phantom = phantom_volume(grid, axis, dx, dz, radius_um=1450.0, wall_um=60.0,
                             core_um=250.0, holder_offset_um=0.0,
                             texture_um=float(texture_um))
    angles = page_angles(turn_pages)
    pages = simulate_pages(phantom, grid, axis, dx, dz, angles,
                           (x_pixels, z_pixels))
    return {"axis": axis, "grid": grid, "phantom": phantom, "angles": angles,
            "pages": pages, "dx": dx, "dz": dz}


def _turns_case(turn_pages=160, noise=0.25, offsets=(9, 21), seed=5,
                texture_um=120.0, tail=24, tilt_deg=20.0):
    """A three-turn stack with known turn starts, built from the phantom pages.

    The filler between the turns is noise, so an alignment that is off by a page
    cannot be lucky, and the per-page noise is what makes the averaging
    measurable. The phantom texture is finer than in the other checks: the
    alignment can only find the phase if the pages really differ from phase to
    phase.
    """
    case = _phantom_case(turn_pages=turn_pages, texture_um=texture_um,
                         tilt_deg=float(tilt_deg))
    clean = case["pages"]
    rng = np.random.default_rng(int(seed))
    scale = float(np.std(clean[0]))
    shape_like = clean[0].shape
    pieces = []
    starts = []
    for position, offset in enumerate((0,) + tuple(offsets)):
        if position:
            pieces.append(rng.normal(scale=1.5 * scale,
                                     size=(int(offset),) + shape_like
                                     ).astype(np.float32))
        starts.append(sum(piece.shape[0] for piece in pieces))
        pieces.append(np.stack([page + rng.normal(scale=float(noise) * scale,
                                                  size=shape_like)
                                for page in clean]).astype(np.float32))
    pieces.append(rng.normal(scale=1.5 * scale, size=(int(tail),) + shape_like
                             ).astype(np.float32))
    return case, np.concatenate(pieces, axis=0), starts, \
        np.stack(clean).astype(np.float64)


def run_self_test(verbose=True):
    """Round-trip a tilted phantom; returns the list of ``(name, ok, detail)``."""
    checks = []

    def report(name, ok, detail=""):
        checks.append((name, bool(ok), str(detail)))
        if verbose:
            print(f"[{'PASS' if ok else 'FAIL'}] {name}"
                  + (f"  {detail}" if detail else ""))
        return bool(ok)

    # 1. The forward model is invertible.
    pose = AxisPose((40.0, 30.0), 12.0, 55.0)
    s = np.array([0.0, 17.5, 63.0])
    t = np.array([5.0, 44.0, 88.0])
    worst = 0.0
    for theta in (0.0, 0.9, 3.7):
        points = page_to_object(pose, 10.0, 10.0, theta, s, t)
        back_s, back_t, off = object_to_page(pose, 10.0, 10.0, theta, points)
        worst = max(worst, float(np.max(np.abs(back_s - s))),
                    float(np.max(np.abs(back_t - t))),
                    float(np.max(np.abs(off))))
    report("a page pixel maps to the object frame and back", worst < 1e-9,
           f"worst error {worst:.2e} um or px")

    # 2. The classic in-plane pose must reproduce the polar assembly stage 2 was
    #    built on: with the axis in the B-scan plane, a pixel's distance to the
    #    axis is its lateral offset and the azimuth is the page angle.
    classic = AxisPose((30.0, 20.0), tilt_deg=90.0, azimuth_deg=90.0)
    worst_classic = 0.0
    for theta in (0.0, 0.6, 2.2):
        point = page_to_object(classic, 10.0, 10.0, theta,
                               np.array([45.0]), np.array([20.0]))[0]
        delta = point - classic.point_um(10.0, 10.0)
        worst_classic = max(
            worst_classic,
            abs(float(np.hypot(delta[0], delta[2])) - 150.0),
            abs(float(delta[1])),
            abs(float(math.atan2(delta[2], delta[0])) - theta),
        )
    report("the in-plane pose reduces to the polar assembly",
           worst_classic < 1e-9,
           f"worst deviation {worst_classic:.2e} (radius 150 um, azimuth = "
           "page angle)")

    case = _phantom_case()
    grid, axis, phantom = case["grid"], case["axis"], case["phantom"]
    pages, angles = case["pages"], case["angles"]
    dx, dz = case["dx"], case["dz"]

    # 3. Sampling has to be exact at the voxel centres (it feeds the phantom).
    nodes = grid_positions(grid).reshape(-1, 3)
    error = float(np.max(np.abs(
        sample_trilinear(phantom, grid, nodes).reshape(phantom.shape)
        - phantom)))
    report("the sampler reads the volume back at its own voxel centres",
           error < 1e-6, f"worst error {error:.2e}; grid {phantom.shape}")

    # 4. Splatting normalisation: constant pages give that constant back.
    flat_pages = [np.ones((40, 30), dtype=np.float32) for _ in range(24)]
    constant, constant_weight = reconstruct(flat_pages, page_angles(24), axis,
                                            dx, dz, grid, decimate=(2, 2))
    filled = constant_weight > 0.0
    spread = float(np.max(np.abs(constant[filled] - 1.0))) if filled.any() else 1.0
    report("constant pages reconstruct to that constant",
           bool(filled.any()) and spread < 1e-9,
           f"{int(filled.sum())} voxels filled, worst deviation {spread:.2e}")

    # 5. The tilted phantom itself has to come back.
    volume, weight = reconstruct(pages, angles, axis, dx, dz, grid,
                                 decimate=(2, 2))
    mask = weight_mask(weight, 0.75)
    best_correlation = correlation(volume, phantom, mask)
    report("the reconstruction of the tilted phantom matches it",
           best_correlation > 0.85,
           f"correlation {best_correlation:.3f} over {int(mask.sum())} voxels")

    # 5b. The pages can only ever reach the cone the tilt allows: every pixel they
    #     sample has to satisfy |z'| <= rho*tan(tilt). That cone is what limits a
    #     tilted-axis recording, so it is checked on the samples themselves.
    direction = axis.unit_vector()
    q = axis.point_um(dx, dz)
    probe_s, probe_t = pixel_axes(pages[0].shape, (5, 5))
    mesh_s, mesh_t = np.meshgrid(probe_s, probe_t, indexing="ij")
    slope = math.tan(math.radians(axis.tilt_deg))
    steepest = 0.0
    for theta in angles[::17]:
        coords = page_to_object(axis, dx, dz, float(theta), mesh_s.ravel(),
                                mesh_t.ravel())
        delta = coords - q
        axial = delta @ direction
        radial = delta - axial[:, None] * direction
        ratio = np.abs(axial) / np.maximum(np.linalg.norm(radial, axis=1), 1e-12)
        steepest = max(steepest, float(ratio.max()))
    report("every sample lands inside the cone the tilt allows",
           steepest <= slope + 1e-9,
           f"steepest sample |z'|/rho = {steepest:.6f} vs tan(tilt) "
           f"{slope:.6f}; reach at 1.5 mm is "
           f"{visible_reach_um(axis, 1500.0):.0f} um")

    # 6. A pose that ignores the tilt must be visibly worse.
    untilted = AxisPose(axis.centre, 0.0, 0.0)
    flat_volume, flat_weight = reconstruct(pages, angles, untilted, dx, dz, grid,
                                           decimate=(2, 2), page_step=2)
    flat_correlation = correlation(flat_volume, phantom,
                                   weight_mask(flat_weight, 0.75))
    report("ignoring the tilt blurs the reconstruction",
           flat_correlation < best_correlation - 0.05,
           f"correlation {flat_correlation:.3f} untilted vs "
           f"{best_correlation:.3f} at the true pose")

    # 7. The focus metric has to prefer the true tilt.
    curve = focus_curve(pages, angles, dx, dz, grid, untilted, "tilt_deg",
                        values=(0.0, 10.0, 20.0, 30.0, 40.0), decimate=(4, 4),
                        page_step=2)
    best_value = max(curve, key=lambda item: item[1])
    report("the page match peaks at the true tilt",
           abs(best_value[0] - axis.tilt_deg) <= 5.0,
           "page match " + ", ".join(f"{value:.0f}deg:{score:.3f}"
                                     for value, score in curve)
           + f" -> best {best_value[0]:.0f} deg, truth {axis.tilt_deg:.0f} deg")

    # 8. Autofocus has to find the tilt on its own, starting from an untilted
    #    guess. The azimuth means nothing while the tilt is zero, so the tilt is
    #    searched first.
    result = autofocus(pages, angles, dx, dz, grid, untilted,
                       parameters=("tilt_deg", "azimuth_deg", "centre_s",
                                   "centre_t"),
                       steps=(4.0, 30.0, 8.0, 8.0), rounds=5, decimate=(4, 4),
                       page_step=2)
    found = result["axis"]
    recovered, recovered_weight = reconstruct(pages, angles, found, dx, dz, grid,
                                              decimate=(2, 2))
    found_correlation = correlation(recovered, phantom,
                                    weight_mask(recovered_weight, 0.75))
    report("autofocus recovers the tilt from an untilted start",
           abs(found.tilt_deg - axis.tilt_deg) <= 3.0
           and found_correlation >= 0.95 * best_correlation,
           f"{found.describe()}; correlation {found_correlation:.3f} vs "
           f"{best_correlation:.3f}; tilt walk "
           + " -> ".join(f"{item[1].tilt_deg:.1f}"
                         for item in result["history"]))

    # 9b. Three noisy turns have to align onto one common phase, and averaging
    #     them has to give a cleaner turn than any single turn.
    turns_case, stack, truth, clean = _turns_case()
    turn_pages = int(clean.shape[0])
    turns = len(truth)
    deliberate = [start + 3 for start in truth]
    plan = {"offsets": deliberate, "lengths": [turn_pages] * turns}
    spans = segment_spans(plan, stack.shape[0], turn_pages)
    refined, _matches = refine_offsets(stack, spans, turn_pages, max_shift=4)
    found = [start for start, _length in refined]
    averaged_pages, turn_counts, agreement = average_turns(stack, refined,
                                                           turn_pages)
    single = correlation(np.asarray(stack[truth[0] + 5]).astype(np.float64),
                         clean[5])
    averaged = correlation(averaged_pages[5].astype(np.float64), clean[5])
    relative_truth = [start - truth[0] for start in truth]
    relative_found = [start - found[0] for start in found]
    report("the turns align onto one common phase and average cleaner",
           relative_found == relative_truth and int(turn_counts.min()) == turns
           and abs(averaged) > abs(single),
           f"starts {deliberate} (3 pages off) -> {found}: the relative starts "
           f"{relative_truth} come back as {relative_found} (one common phase "
           f"offset stays - that is only where the object frame starts, so it "
           f"changes nothing); every phase fed by {int(turn_counts.min())} turns, "
           f"turn match {agreement:.3f}; page 5 matches the clean page at "
           f"{averaged:.3f} averaged vs {single:.3f} single")

    # 9c. Averaging the turns before the reconstruction has to give the same
    #     volume as reconstructing every turn and averaging afterwards (a splat is
    #     linear), which is what makes the averaging worth doing at all.
    grid_turns = turns_case["grid"]
    axis_turns = turns_case["axis"]
    dx_turns, dz_turns = turns_case["dx"], turns_case["dz"]
    angles_turns = page_angles(turn_pages)
    average_volume, _weight = reconstruct(averaged_pages, angles_turns,
                                          axis_turns, dx_turns, dz_turns,
                                          grid_turns, decimate=(2, 2))
    stacked = np.zeros_like(average_volume)
    for start, _usable in refined:
        turn_volume, _turn_weight = reconstruct(stack[start:start + turn_pages],
                                                angles_turns, axis_turns,
                                                dx_turns, dz_turns, grid_turns,
                                                decimate=(2, 2))
        stacked += turn_volume
    stacked /= float(len(refined))
    difference = float(np.max(np.abs(average_volume - stacked))) \
        / max(1e-12, float(np.max(np.abs(stacked))))
    report("averaging the turns first equals reconstructing them all",
           difference < 1e-3,
           f"largest relative voxel difference {difference:.2e}")

    passed = sum(1 for _, ok, _ in checks if ok)
    if verbose:
        print(f"\n{passed}/{len(checks)} checks passed")
    return checks

# --------------------------------------------------------------------------- #
# Inspector for a real recording
# --------------------------------------------------------------------------- #


def _load_stage1():
    """Import the stage 1 tool from this folder (importing it brings no Qt)."""
    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
    import RotationalReconstruct  # noqa: E402 - deliberate late import
    return RotationalReconstruct


def save_slice_png(volume, path, spacing=None, levels=None):
    """Write the three central slices of ``volume`` as one PNG, if matplotlib is there.

    ``spacing`` is the voxel edge of the three grid axes (micrometres); it sets the
    display aspect of the two slices that mix a lateral axis with the axial one, so
    a volume with a finer axial voxel is not drawn stretched.

    ``levels`` is the display window ``(vmin, vmax)`` the caller paints its own
    slices with - the panel hands over whatever its brightness slider leaves the
    window at - so the file looks like the screen. ``None`` lets matplotlib choose.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional dependency
        print(f"no slice image: matplotlib unavailable ({exc})")
        return None
    shape = volume.shape
    edges = (np.ones(3) if spacing is None
             else np.asarray(spacing, dtype=np.float64).reshape(-1))
    if edges.size < 3:
        edges = np.ones(3)
    low, high = (None, None) if levels is None else (levels[0], levels[1])
    figure, boxes = plt.subplots(1, 3, figsize=(12, 4), dpi=110)
    for position, box in enumerate(boxes):
        plane = np.take(volume, shape[position] // 2, axis=position)
        # np.rot90 leaves the last remaining grid axis on the horizontal and the
        # other one on the vertical, so the aspect is vertical over horizontal.
        others = [axis for axis in range(3) if axis != position]
        aspect = float(edges[others[-1]]) / float(edges[others[0]])
        box.imshow(np.rot90(plane), cmap="gray", aspect=aspect,
                   vmin=low, vmax=high)
        box.set_title(f"grid axis {position}, index {shape[position] // 2}",
                      fontsize=8)
        box.axis("off")
    figure.tight_layout()
    figure.savefig(Path(path))
    plt.close(figure)
    print(f"slice image written: {path}")
    return Path(path)


def inspect_file(path, turn_pages=None, voxel_um=30.0, decimate=(4, 4),
                 tilt_values=(DEFAULT_TILT_DEG,), azimuth_deg=90.0,
                 centre=None, row_window=None, page_step=1, direction=1.0,
                 axial_step_um=DEFAULT_AXIAL_STEP_UM, autofocus_rounds=0,
                 save=None, slice_png=None, single_turn=False, max_shift=2,
                 verbose=True):
    """Estimate the rotation axis of a real recording and render it.

    The file is loaded whole (stage 1 keeps every depth row) and segmented exactly
    as the GUI does. Every full turn is then aligned onto one phase and averaged,
    so the reconstruction works on a turn that has the SNR of the whole recording;
    it is splatted into a grid around a trial axis and the page match of the result
    is printed for every trial tilt, because a wrong tilt smears the volume along
    the out-of-plane direction and nothing else in the pipeline would notice.
    Returns the report dict.
    """
    stage1 = _load_stage1()
    data, meta = stage1.load_scan([Path(path)])
    plan = stage1.detect_segments(data, stage1.DEFAULT_HALF_TURN_BLINES,
                                  depth_window=tuple(meta["depth_window"]))
    pages_per_turn = int(turn_pages or plan.get("turn_pages") or data.shape[0])
    rows = (int(meta["depth_window"][0]), int(meta["depth_window"][1])) \
        if row_window is None else (int(row_window[0]), int(row_window[1]))
    rows = (max(0, rows[0]), min(int(data.shape[2]), max(rows[0] + 1, rows[1])))
    dx = float(meta["x_step_um"])
    dz = float(axial_step_um)
    x_pixels = int(data.shape[1])
    spans = segment_spans(plan, int(data.shape[0]), pages_per_turn)
    spans, span_matches = refine_offsets(data, spans, pages_per_turn,
                                         max_shift=int(max_shift))
    if single_turn:
        start, usable = max(spans, key=lambda item: item[1])
        length = max(2, min(int(usable), pages_per_turn))
        pages = np.asarray(data[start:start + length, :, rows[0]:rows[1]])
        counts = np.ones(length, dtype=np.int32)
        agreement = 1.0
    else:
        averaged, counts, agreement = average_turns(data, spans, pages_per_turn,
                                                    row_window=rows)
        if averaged is None:
            raise SystemExit("no full turn in the recording - nothing to stack")
        pages = averaged
        length = pages_per_turn
    angles = page_angles(length, direction=direction)
    centre_px = ((0.5 * (x_pixels - 1), 0.5 * (rows[0] + rows[1] - 1))
                 if centre is None else (float(centre[0]), float(centre[1])))
    axis0 = AxisPose(centre_px, tilt_deg=DEFAULT_TILT_DEG,
                     azimuth_deg=azimuth_deg)
    # The pages of one pose only ever reach the cone |z'| <= rho*tan(tilt) around
    # the axis, so the grid has to be sized for the *most* tilted pose that will be
    # tried - an untilted pose fills a plane, and a grid built from it could not
    # hold any tilted trial.
    envelope_tilt = max([1.0] + [abs(float(value)) for value in tilt_values])
    envelope = AxisPose(centre_px, tilt_deg=envelope_tilt,
                        azimuth_deg=azimuth_deg)
    grid = default_grid(envelope, dx, dz, x_pixels, int(data.shape[2]),
                        voxel_um=voxel_um, row_window=rows)
    if verbose:
        print(f"file        : {Path(path).name}")
        print(f"stack       : {tuple(data.shape)} pages x lateral x depth, "
              f"rows {rows[0]}..{rows[1] - 1} of {data.shape[2]} used "
              f"(window {tuple(meta['depth_window'])})")
        print(f"turns       : {len(spans)} full turn(s) at page "
              f"{[int(start) for start, _length in spans]} of {pages_per_turn} "
              f"pages, aligned within +-{int(max_shift)} page(s) -> every phase "
              f"fed by {int(np.min(counts))} turn(s), page match {agreement:.3f}, "
              f"direction {direction:+.0f}")
        if envelope_tilt >= 89.9:
            print("cone        : the axis lies in the B-scan plane, so the pages "
                  "sweep the whole depth range of the row window")
        else:
            print(f"cone        : at {envelope_tilt:.0f} deg a point "
                  f"{0.5 * x_pixels * dx / 1000.0:.1f} mm from the axis is visible "
                  f"over +-{float(visible_reach_um(envelope, 0.5 * x_pixels * dx)):.0f}"
                  f" um of depth only, so the tilt decides how much of the sample "
                  f"any page can hold")
        print(f"grid        : {grid.shape} voxels of {voxel_um:.1f} um "
              f"({grid.size() / 1e6:.1f} M), decimation {decimate}, page step "
              f"{page_step}")
        print(f"start pose  : {axis0.describe()}")
    curve = focus_curve(pages, angles, dx, dz, grid, axis0, "tilt_deg",
                        values=tilt_values, decimate=decimate,
                        page_step=page_step, row_offset=rows[0])
    best_tilt, _ = max(curve, key=lambda item: item[1])
    axis = axis0.adjusted("tilt_deg", best_tilt - axis0.tilt_deg)
    if autofocus_rounds:
        result = autofocus(pages, angles, dx, dz, grid, axis,
                           parameters=("tilt_deg", "azimuth_deg", "centre_s",
                                       "centre_t"),
                           steps=(2.0, 30.0, 8.0, 8.0),
                           rounds=autofocus_rounds, decimate=decimate,
                           page_step=page_step, row_offset=rows[0])
        axis = result["axis"]
    volume, weight = reconstruct(pages, angles, axis, dx, dz, grid,
                                 decimate=decimate, page_step=page_step,
                                 row_offset=rows[0])
    mask = weight_mask(weight, 0.5)
    if verbose:
        print("tilt scan   : " + ", ".join(f"{value:.0f}deg {score:.3g}"
                                           for value, score in curve))
        print(f"best pose   : {axis.describe()}")
        if bool(mask.any()):
            samples = weight[mask]
            print(f"volume      : {int(mask.sum())} of {grid.size()} voxels "
                  f"filled, values {volume[mask].min():.3g}.."
                  f"{volume[mask].max():.3g}, median "
                  f"{float(np.median(samples)):.0f} sample(s) per voxel")
        else:
            print("volume      : nothing filled - the pose misses the data")
    report = {"meta": meta, "plan": plan, "grid": grid, "axis": axis,
              "tilt_curve": curve, "volume": volume.astype(np.float32),
              "weight": weight.astype(np.float32),
              "turns": {"spans": [(int(start), int(usable))
                                  for start, usable in spans],
                        "matches": [float(value) for value in span_matches],
                        "counts": counts, "match": float(agreement)}}
    if save is not None:
        from tifffile import imwrite
        imwrite(Path(save), report["volume"])
        print(f"volume written: {save}")
    if slice_png is not None:
        save_slice_png(report["volume"], slice_png)
    return report


# --------------------------------------------------------------------------- #
# Command line
# --------------------------------------------------------------------------- #


def main(argv=None):
    """``--file`` inspects a recording, anything else runs the phantom self-test."""
    parser = argparse.ArgumentParser(
        description="Rotational C-scan 3D reconstruction prototype (stage 2, "
                    "offline): tilted-axis forward model, splatting "
                    "reconstruction, sharpness autofocus and a phantom test.")
    parser.add_argument("--self-test", action="store_true",
                        help="round-trip a tilted phantom (the default)")
    parser.add_argument("--file", type=Path, help="C-scan TIFF to inspect")
    parser.add_argument("--turn-pages", type=int,
                        help="pages per turn (default: as segmented)")
    parser.add_argument("--voxel-um", type=float, default=30.0,
                        help="volume voxel size in micrometres")
    parser.add_argument("--decimate", default="4,4",
                        help="page pixel step 'lateral,depth'")
    parser.add_argument("--page-step", type=int, default=1,
                        help="use every Nth page of the turn")
    parser.add_argument("--tilts", default=f"{DEFAULT_TILT_DEG:.0f}",
                        help="tilt scan, degrees from the plane normal; 90 is the "
                             "axis lying in the B-scan plane (the rig's pose)")
    parser.add_argument("--azimuth-deg", type=float, default=90.0,
                        help="in-plane direction the axis leans into; 90 = along "
                             "the depth axis")
    parser.add_argument("--centre", default="",
                        help="axis pixel 's,t' where it pierces the B-line")
    parser.add_argument("--rows", default="",
                        help="depth rows 'first,stop' to reconstruct")
    parser.add_argument("--direction", type=float, default=1.0,
                        help="rotation sense: +1 or -1")
    parser.add_argument("--axial-step-um", type=float,
                        default=DEFAULT_AXIAL_STEP_UM,
                        help="depth sampling of the file (scales the depth axis)")
    parser.add_argument("--autofocus-rounds", type=int, default=0,
                        help="coordinate-descent rounds on the axis pose")
    parser.add_argument("--single-turn", action="store_true",
                        help="use one turn instead of aligning and averaging all")
    parser.add_argument("--max-shift", type=int, default=2,
                        help="pages a turn may be nudged onto the reference phase")
    parser.add_argument("--save", type=Path, help="volume TIFF to write")
    parser.add_argument("--slice-png", type=Path,
                        help="PNG of the three central slices")
    args = parser.parse_args(argv)
    if args.self_test or args.file is None:
        checks = run_self_test()
        return 0 if all(ok for _, ok, _ in checks) else 1
    centre = (tuple(float(part) for part in args.centre.split(","))
              if args.centre else None)
    rows = (tuple(int(part) for part in args.rows.split(","))
            if args.rows else None)
    decimate = tuple(int(part) for part in args.decimate.split(","))
    tilt_values = tuple(float(part) for part in args.tilts.split(","))
    inspect_file(args.file, turn_pages=args.turn_pages, voxel_um=args.voxel_um,
                 decimate=decimate, tilt_values=tilt_values,
                 azimuth_deg=args.azimuth_deg, centre=centre, row_window=rows,
                 page_step=args.page_step, direction=args.direction,
                 axial_step_um=args.axial_step_um,
                 autofocus_rounds=args.autofocus_rounds, save=args.save,
                 slice_png=args.slice_png, single_turn=args.single_turn,
                 max_shift=args.max_shift)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())






