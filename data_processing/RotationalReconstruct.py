# -*- coding: utf-8 -*-
"""Rotational C-scan processing - STAGE 1: segmentation only.

This module currently stops after the segmentation. It reads a rotational C-scan
with every depth row - the display needs them all - splits the recording into its
full turns and puts those turns on top of each other in the window: the pages of
every full turn are added, so the sum is sharp only while the turns really are
repeats. The turns are lined up first by maximising the page match between them;
whatever shift that takes is how many pages a detected boundary is off, which is
the check that the segmentation is right. Nothing is reconstructed yet.

Only the *calculation* is restricted to the depth rows that hold the sample (the
depth window, default 100-350): the sample holder does not turn with the spindle,
so measuring it only dilutes the signatures and the page match. It is still
loaded and still shown - the top panel marks the window with dashed lines.

The window holds only the scan import (path, depth window, Load) and the result
(the summed pages at one angle, the summed turn, and one line naming the measured
turn): no segmentation parameters, because the detector measures the turn itself,
and no shift controls, because the alignment is automatic. Everything else - the
segmentation report and the page match of every segment - is printed on stdout.

The order is deliberate: the reconstruction needs the rotation centre, and that
axis does not even have to lie inside the B-line. Until the centre is settled -
and until the segment boundaries are known to be stable - the polar assembly,
the Cartesian reconstruction, the export and everything that displays a volume
are commented out (search for "STAGE 2"), so the running program never reaches
them.

How the acquisition works (unchanged): the Y galvo is parked (YStepSize = 0)
while the sample is rotated, so the C-scan's Y index (one TIFF page per acquired
B-line) becomes the rotation index. Each page is a B-line whose X axis crosses
the rotation axis at a selectable pixel column ``Xc``.

A B-line is a diameter, so one half-turn of rotation is enough to cover the whole
sample: the right half of a page looks in one direction and its left half looks
180 degrees the other way. The second half-turn in the file therefore only
repeats directions that were already measured, and because the OCT line centre
never sits exactly on the lens axis those repeats are mutually offset. This tool
uses *only* the first half-turn and discards the rest, so the two offset repeats
are never averaged together:

    page k (k < half-turn count)  ->  180 * k / (half-turn count)  degrees

The right half of page k supplies that direction and the left half supplies the
direction 180 degrees away, giving ``2 * half-turn count`` evenly spaced
directions over 360 degrees, each measured exactly once. No binning and no
averaging is performed. The profiles are resampled onto a radial grid centred on
``Xc`` and then onto a Cartesian XY grid, which yields the reconstructed volume
``(Ny, Nx, Z)``.

The number of B-lines spanning 180 degrees is an input parameter, adjusted until
the image looks right: too few or too many shows up directly in the result.

Fixed conventions:
    * rotation direction: counter-clockwise, start angle 0 degrees
    * rotation axis parallel to the imaging depth axis
    * lateral pixel size: LATERAL_PIXEL_SIZE_UM
    * amplitude display only; regions without data are exactly black

Nothing is reconstructed on load: the default parameters are only a starting
guess, so the first reconstruction happens when Apply is pressed.

Run ``python RotationalReconstruct.py --self-test`` to check the reconstruction
against a synthetic volume with a known ground truth.
"""

import argparse
import json
import math
import re
import sys
import tempfile
from pathlib import Path

import numpy as np

try:
    import tifffile as TIFF
except Exception as error:  # pragma: no cover - environment issue
    raise RuntimeError(f"tifffile is required: {error}")

try:
    from scipy import ndimage as NDIMAGE
except Exception:  # optional: only used by the speckle filter
    NDIMAGE = None

try:
    from PyQt5 import QtCore, QtWidgets
    from PyQt5.QtCore import Qt, QThread, pyqtSignal

    QT_AVAILABLE = True
    QT_IMPORT_ERROR = None
except Exception as _qt_error:  # pragma: no cover - the UI needs PyQt5
    QT_AVAILABLE = False
    QT_IMPORT_ERROR = _qt_error
    QtCore = QtWidgets = None
    Qt = None

    class QThread:  # placeholder so the module stays importable for --self-test
        def __init__(self, *args, **kwargs):
            pass

    def pyqtSignal(*args, **kwargs):  # noqa: N802 - mirrors the Qt name
        return None


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

# Lateral sampling of a B-line. Used to size the Cartesian grid and to report
# physical extents. Tune here when the real optics/scan geometry is measured.
LATERAL_PIXEL_SIZE_UM = 10.0

# Number of B-lines that span 180 degrees of rotation. One half-turn already
# covers the whole sample because each B-line is a diameter: its two halves look
# 180 degrees apart. Only this first half-turn is used and the rest of the file
# is discarded, so the repeat pass is never averaged into the result.
MIN_HALF_TURN_BLINES = 1
MAX_HALF_TURN_BLINES = 1000
DEFAULT_HALF_TURN_BLINES = 550

# Hard cap on the Cartesian grid edge length, and a soft cap on the volume
# memory footprint. Exceeding the byte limit raises the decimation factor.
MAX_GRID_SIZE = 2048
VOLUME_BYTE_LIMIT = 4 * 1024 ** 3

# Z chunk used by the reconstruction gather, expressed as the maximum number of
# simultaneous output-pixel/slice samples.
GATHER_BUDGET_ELEMENTS = 2_000_000

CSCAN_PATTERN = re.compile(
    r"^Cscan-(?P<num>\d+)-Y(?P<y>\d+)-X(?P<x>\d+)-Z(?P<z>\d+)\.tiff?$",
    re.IGNORECASE,
)

# Depth rows of a B-line (pixels along Z, half-open ``[start, stop)``) that the
# *calculations* of this module may use: the segmentation signatures, the
# silhouettes and the page match.
#
# A C-scan window usually also contains the sample holder ("base"). It does not
# turn with the spindle, so it is not part of the sample's surface of revolution
# and it is useless for the reconstruction - but because it is bright and wide it
# dominates any measurement taken over the whole depth range: the outer edge, the
# radial profiles used to find the rotation centre and the segmentation
# signatures. Restricting the measurements to the sample rows keeps them on the
# sample only.
#
# The rows outside the window are loaded and displayed like any other (the window
# is not a crop): they are only left out of the measurements, so nothing is lost
# from the picture.
#
# Rows are counted from the top of the B-line. Set the window to ``None`` (or call
# ``load_scan(..., depth_window=None)``) to measure over every row. It is clamped
# to the rows that actually exist, so a file with fewer depth pixels is never cut
# down to one row.
DEFAULT_DEPTH_WINDOW = (100, 350)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def resolve_depth_window(z_pixels, window=DEFAULT_DEPTH_WINDOW):
    """Clamp a ``(start, stop)`` depth window to ``0 <= start < stop <= z_pixels``.

    ``None`` means "keep every row". A window that does not overlap the data at
    all also falls back to "keep every row" - returning an empty (or one-row)
    stack would be far more surprising than ignoring a window that cannot apply.
    """
    total = int(z_pixels)
    if window is None or total <= 1:
        return 0, max(1, total)
    try:
        start = int(round(float(window[0])))
        stop = int(round(float(window[1])))
    except (TypeError, ValueError, IndexError):
        return 0, total
    if start >= total or stop <= 0:
        return 0, total
    start = max(0, start)
    stop = min(total, stop)
    if stop <= start:
        return 0, total
    return start, stop

def depth_slice(depth_window, z_pixels):
    """``slice`` of the depth rows a measurement may use.

    ``None`` means "every row". Anything else is clamped to the rows the stack
    really has, so a window that lies outside the file never selects nothing.
    """
    if depth_window is None:
        return slice(0, max(1, int(z_pixels)))
    start, stop = depth_window
    rows = max(1, int(z_pixels))
    start = max(0, min(int(start), rows - 1))
    stop = max(start + 1, min(int(stop), rows))
    return slice(start, stop)


def parse_cscan_filename(path):
    """Return the ``num``/``y``/``x``/``z`` fields encoded in a C-scan name."""
    match = CSCAN_PATTERN.match(Path(path).name)
    if match is None:
        return None
    return {key: int(value) for key, value in match.groupdict().items()}


def read_page_stack(path):
    """Read every TIFF page of ``path`` as a 2D array.

    The acquisition appends one page per B-line, and tifffile stores each of
    those appends as its own series, so ``imread`` would only return the first
    one. Iterating ``TiffFile.pages`` is the reliable route.
    """
    with TIFF.TiffFile(str(path)) as tif:
        pages = [np.asarray(page.asarray()) for page in tif.pages]
    if not pages:
        raise ValueError(f"TIFF contains no pages: {path}")
    cleaned = []
    for page in pages:
        if page.ndim == 3:
            page = page[..., 0]
        if page.ndim != 2:
            raise ValueError(f"Unsupported page shape {page.shape} in {path}")
        cleaned.append(page)
    return cleaned


def page_to_amplitude(page, z_pixels):
    """Reduce one saved page to its amplitude image of shape ``(X, Z)``.

    Complex acquisitions interleave magnitude then phase along the page width,
    so a width of ``2 * Z`` means the magnitude is the first half. A width of
    ``Z`` means the page already holds real (amplitude) data.
    """
    rows, cols = page.shape
    if z_pixels is not None:
        if cols == 2 * z_pixels and rows != 2 * z_pixels:
            return page[:, :z_pixels]
        if cols == z_pixels and rows != z_pixels:
            return page
    # Fall back to a shape-only guess when the filename did not tell us Z.
    if rows == cols and cols > 1 and cols % 2 == 0:
        return page[:, : cols // 2]
    return page


def load_scan(paths, progress=None, depth_window=DEFAULT_DEPTH_WINDOW):
    """Load and concatenate C-scan files in numeric-then-page order.

    Every depth row of every file is kept: the stack is complete, so the views can
    show the whole B-line including the sample holder ("base"). ``depth_window``
    does not crop anything here - it is resolved and reported in
    ``meta["depth_window"]`` (see :data:`DEFAULT_DEPTH_WINDOW`) so that the
    callers can hand it to the measurements, which must not see the non-rotating
    holder. ``meta`` reports both the window to measure over and the full depth
    extent of the file. Pass ``depth_window=None`` to measure over every row.
    """
    def ordered(path_list):
        numbered, unnumbered = [], []
        for path in path_list:
            info = parse_cscan_filename(path)
            if info is None:
                unnumbered.append(path)
            else:
                numbered.append((info["num"], str(path), path))
        numbered.sort(key=lambda item: (item[0], item[1]))
        unnumbered.sort(key=lambda item: str(item))
        return [item[2] for item in numbered] + unnumbered

    files = ordered(list(paths))
    if not files:
        raise ValueError("No input files were selected.")

    amplitudes = []
    z_from_name = None
    depth_start, depth_stop = None, None
    full_z_pixels = None
    for index, path in enumerate(files):
        if progress is not None:
            progress(index, len(files), f"Reading {Path(path).name}")
        info = parse_cscan_filename(path)
        if info is not None:
            z_from_name = info["z"]
        for page in read_page_stack(path):
            amplitude = page_to_amplitude(page, z_from_name)
            if amplitude.shape[0] < 3 or amplitude.shape[1] < 2:
                raise ValueError(
                    f"Page in {Path(path).name} is too small: {amplitude.shape}"
                )
            if full_z_pixels is None:
                full_z_pixels = int(amplitude.shape[1])
                depth_start, depth_stop = resolve_depth_window(
                    full_z_pixels, depth_window
                )
            # The whole B-line is kept, holder rows included: they belong in the
            # picture. Only the measurements skip them, and the window that says
            # which rows those are travels in ``meta``.
            amplitudes.append(np.ascontiguousarray(amplitude, dtype=np.float32))

    shapes = {array.shape for array in amplitudes}
    if len(shapes) != 1:
        raise ValueError(f"All pages must share one shape, got {sorted(shapes)}")
    x_pixels, z_pixels = amplitudes[0].shape

    data = np.stack(amplitudes, axis=0)
    meta = {
        "files": [str(path) for path in files],
        "pages": int(data.shape[0]),
        "x_pixels": int(x_pixels),
        "z_pixels": int(z_pixels),
        "z_pixels_full": int(full_z_pixels if full_z_pixels is not None else z_pixels),
        # Not a crop: the rows the measurements may use, clamped to the file.
        "depth_window": [
            int(0 if depth_start is None else depth_start),
            int(z_pixels if depth_stop is None else depth_stop),
        ],
        "x_step_um": LATERAL_PIXEL_SIZE_UM,
    }
    return data, meta


# ===== STAGE 2 - commented out while only the segmentation is developed =====
# # --------------------------------------------------------------------------- #
# # Polar assembly (one half-turn only)
# # --------------------------------------------------------------------------- #
#
# def resample_columns(page, positions, x_pixels):
#     """Linearly resample ``page[x, z]`` at fractional ``positions`` along x."""
#     base = np.floor(positions)
#     frac = (positions - base)[:, None].astype(np.float32)
#     lower = np.clip(base.astype(np.int64), 0, x_pixels - 1)
#     upper = np.clip(lower + 1, 0, x_pixels - 1)
#     return (1.0 - frac) * page[lower] + frac * page[upper]
#
#
# def build_half_turn_stacks(data, xc, half_turn_blines, page_offset=0, page_count=None,
#                            progress=None):
#     """Turn one half-turn of B-lines into the full 360 degrees of radial profiles.
#
#     Only the first ``half_turn_blines`` pages are used; every page after them is
#     ignored. A B-line is a diameter through the rotation axis, so its right half
#     supplies the radial profile at ``k * 180 / half_turn_blines`` degrees and its
#     left half supplies the profile 180 degrees away. One half-turn therefore
#     fills 360 degrees with ``2 * half_turn_blines`` evenly spaced directions,
#     each measured exactly once.
#
#     There is deliberately no binning and no averaging here. The second half-turn
#     in the file is a repeat of the first, and when the OCT line centre does not
#     coincide with the lens axis the two repeats are mutually offset; averaging
#     them is what blurred the reconstruction. Directions come from the user's
#     count rather than from a measured angle, so a count that is too small or too
#     large shows up directly in the reconstructed image.
#
#     Returns ``(stack, slot_pages, slot_angles, slot_of_bin, info)``. ``stack``
#     has shape ``(2 * half_turn_blines, R + 1, Z)`` and slots with no page behind
#     them are left at zero.
#
#     ``page_offset`` and ``page_count`` choose which pages form the half-turn, so
#     one call also handles each detected segment of a multi-turn recording. The
#     defaults (0 / ``None``) reproduce the historical "first pages of the file".
#     """
#     n_pages, x_pixels, z_pixels = data.shape
#     n_half = max(1, int(round(half_turn_blines)))
#     step_deg = 180.0 / float(n_half)
#     n_slots = 2 * n_half
#
#     r_max = radial_pixel_limit(x_pixels, xc)
#     radial = np.arange(0, r_max + 1, dtype=np.float64)
#     right_positions = xc + radial
#     left_positions = np.clip(xc - radial, 0.0, float(x_pixels - 1))
#
#     start = int(max(0, min(int(page_offset), n_pages)))
#     if page_count is None:
#         stop = n_pages
#     else:
#         stop = int(min(n_pages, start + max(0, int(page_count))))
#     used = int(max(0, min(n_half, stop - start)))
#     stack = np.zeros((n_slots, radial.size, z_pixels), dtype=np.float32)
#     slot_pages = [[] for _ in range(n_slots)]
#
#     report_every = max(1, used // 50)
#     for local_index in range(used):
#         page_index = start + local_index
#         if progress is not None and local_index % report_every == 0:
#             progress(local_index, used, f"Filling half-turn {local_index + 1}/{used}")
#         page = data[page_index]
#         stack[local_index] = resample_columns(page, right_positions, x_pixels)
#         slot_pages[local_index].append(page_index)
#         stack[local_index + n_half] = resample_columns(page, left_positions, x_pixels)
#         slot_pages[local_index + n_half].append(page_index)
#
#     slot_angles = np.arange(n_slots, dtype=np.float64) * step_deg
#     slot_of_bin = {slot: slot for slot in range(n_slots)}
#     info = {
#         "half_turn_blines": int(n_half),
#         "degrees_per_bline": float(step_deg),
#         "angular_samples": int(n_slots),
#         "pages_used": int(used),
#         "pages_total": int(n_pages),
#         "pages_discarded": int(max(0, n_pages - used)),
#         "missing_slots": int(n_slots - 2 * used),
#         "radial_pixels": int(r_max),
#         "page_offset": int(start),
#         "page_count": int(used),
#         "page_window": [int(start), int(start + used)],
#     }
#     return stack, slot_pages, slot_angles, slot_of_bin, info
#
#
# --------------------------------------------------------------------------- #
# Multi-turn segmentation (cross-correlation of the rotation repeats)
# --------------------------------------------------------------------------- #
#
# A recording usually spans more than one full turn. Because the sample only
# rotates, page ``k`` and page ``k + T`` (T = one full turn = 2 * half-turn
# B-lines) show the same direction and therefore look alike, while page ``k`` and
# page ``k + T/2`` look alike after mirroring about the rotation axis. Both facts
# are used here: the mirror relation pins the half-turn count / centre, and the
# turn relation locates the segment boundaries of the whole stack.

# Signature resolution used for the cross-correlation search (cheap and enough).
SEGMENT_SIGNATURE_SIZE = 64
# Never split into more than this many segments.
SEGMENT_MAX_COUNT = 4
# Segments shorter than this fraction of a turn are reported as partial.
SEGMENT_MIN_LENGTH_FRACTION = 0.25
# Search window for the turn length and the segment starts (fraction of a turn).
SEGMENT_SEARCH_TOLERANCE = 0.05
# Page pairs averaged per candidate lag (spread over the whole overlap).
SEGMENT_SAMPLE_COUNT = 48
# When several candidates score within this margin of the best, the one closest to
# the nominal value wins. Perfectly periodic stretches make the score profile flat,
# and then the nominal boundary is the safer choice.
SEGMENT_TIE_MARGIN = 0.02
# Below this turn score the peak is not trusted; the search is widened, and if that
# stays weak the nominal parameter is used and the plan is flagged as nominal.
SEGMENT_WEAK_SCORE = 0.30
SEGMENT_WIDE_TOLERANCE = 0.20
# A segment counts as a full turn when it is at least this fraction of a turn
# (rotation wobble makes real segments a little shorter or longer than nominal).
SEGMENT_FULL_FRACTION = 0.95
# A sub-period that correlates at least this well relative to the multi-turn peak
# is taken as the fundamental turn: a parameter that is off by an integer factor
# otherwise yields segments that are several turns long.
SEGMENT_FUNDAMENTAL_FACTOR = 0.75
# Bumped when the GUI or the segmentation behaviour changes, so the copy that is
# actually running can be identified from the window title and the console banner.
BUILD_TAG = "2026-09-22 stage 1: segmentation only"
SCRIPT_PATH = Path(__file__).resolve()


def _block_mean_step(count, target):
    """Block size used by :func:`_block_mean_axis`: closest to ``count/target``."""
    return max(1, int(round(int(count) / float(max(1, int(target))))))


def _block_mean_axis(image, axis, target):
    """Block-mean downsample one axis of ``image`` to about ``target`` samples.

    The two axes are handled separately so anisotropic pages (many columns, few
    depth samples) keep both dimensions instead of being cropped.
    """
    count = int(image.shape[axis])
    step = _block_mean_step(count, target)
    crop = max(step, (count // step) * step)
    if axis == 0:
        trimmed = image[:crop]
        return trimmed.reshape(crop // step, step, trimmed.shape[1]).mean(axis=1)
    trimmed = image[:, :crop]
    return trimmed.reshape(trimmed.shape[0], crop // step, step).mean(axis=2)


def page_signature(page, size=SEGMENT_SIGNATURE_SIZE):
    """Downsample one ``(lateral, depth)`` page to its correlation signature.

    Streaming a file page by page through this function never holds the whole
    stack in memory, which matters for the multi-gigabyte rotations.
    """
    values = np.asarray(page, dtype=np.float32)
    lateral_target = max(2, min(int(size), int(values.shape[0])))
    depth_target = max(2, min(int(size), int(values.shape[1])))
    return _block_mean_axis(_block_mean_axis(values, 0, lateral_target), 1, depth_target)


def normalize_signatures(signatures, remove_rotational_mean=True):
    """Zero-mean / unit-norm every page signature.

    The rotation-invariant part (the mean over all pages) is dropped so only the
    angle-dependent content drives the correlation, and the per-page normalisation
    removes gain, brightness and laser drift.
    """
    values = np.asarray(signatures, dtype=np.float32)
    if remove_rotational_mean and values.shape[0] > 1:
        values = values - values.mean(axis=0, keepdims=True)
    flat = values.reshape(values.shape[0], -1)
    norm = np.linalg.norm(flat, axis=1, keepdims=True)
    norm[norm <= 0.0] = 1.0
    return (flat / norm).reshape(values.shape)


def page_signatures(
    data,
    size=SEGMENT_SIGNATURE_SIZE,
    remove_rotational_mean=True,
    progress=None,
    depth_window=None,
):
    """Signature stack of a whole ``(pages, x, z)`` array.

    Every page is reduced by :func:`page_signature` and then normalised by
    :func:`normalize_signatures`. Returns ``(pages, h, w)`` float32, where ``h``
    and ``w`` are the block-mean sizes of the lateral and depth axes.

    ``depth_window`` restricts every page to the depth rows the segmentation may
    measure (see :data:`DEFAULT_DEPTH_WINDOW`). The stack itself may well hold
    all of them: only the measurement leaves the holder out.
    """
    n_pages = int(data.shape[0])
    rows = depth_slice(depth_window, int(data.shape[2]))
    signatures = None
    report_every = max(1, n_pages // 50)
    for index in range(n_pages):
        if progress is not None and index % report_every == 0:
            progress(index, n_pages, f"Signatures {index + 1}/{n_pages}")
        small = page_signature(data[index][:, rows], size)
        if signatures is None:
            # Take the shape from the data itself: the block size is an integer,
            # so the analytic size is not guaranteed to match exactly.
            signatures = np.empty((n_pages,) + small.shape, dtype=np.float32)
        signatures[index] = small
    return normalize_signatures(signatures, remove_rotational_mean)


def _sample_page_indices(start, stop, count):
    """Evenly spaced integer page indices inside ``[start, stop)``."""
    start = int(start)
    stop = int(stop)
    if stop <= start:
        return np.zeros(0, dtype=np.int64)
    count = max(1, int(count))
    return np.linspace(start, stop - 1, min(count, stop - start)).astype(np.int64)


def rotation_similarity(signatures, lag, start=0, stop=None,
                        sample_count=SEGMENT_SAMPLE_COUNT):
    """Mean NCC between page k and page k+lag over evenly sampled pairs.

    ``start``/``stop`` bound the pages that exist for BOTH partners, so the score
    only uses pairs that are actually inside the recording.
    """
    lag = int(lag)
    n_pages = int(signatures.shape[0])
    if lag <= 0 or n_pages - lag < 2:
        return float("nan")
    start = max(0, int(start))
    stop = n_pages - lag if stop is None else min(int(stop), n_pages - lag)
    first = _sample_page_indices(start, stop, sample_count)
    if first.size == 0:
        return float("nan")
    second = first + lag
    return float(np.mean(np.einsum("nij,nij->n", signatures[first], signatures[second])))


def _best_candidate(profile, nominal, margin=SEGMENT_TIE_MARGIN):
    """Key of the best-scoring candidate, preferring ``nominal`` among near ties."""
    scored = {key: value for key, value in profile.items()
              if value is not None and not math.isnan(value)}
    if not scored:
        return int(nominal), float("nan")
    best_value = max(scored.values())
    floor = best_value - max(0.0, float(margin))
    near = [key for key, value in scored.items() if value >= floor]
    best_key = min(near, key=lambda key: (abs(key - int(nominal)), key))
    return int(best_key), float(scored[best_key])


def find_turn_length(signatures, expected_pages, tolerance=SEGMENT_SEARCH_TOLERANCE,
                     sample_count=SEGMENT_SAMPLE_COUNT):
    """Best full-turn length (in pages) around ``expected_pages``.

    Returns ``(turn_pages, score, profile)`` with ``profile`` mapping ``lag -> score``.
    """
    expected = max(2.0, float(expected_pages))
    low = max(2, int(math.floor(expected * (1.0 - float(tolerance)))))
    high = max(low, int(math.ceil(expected * (1.0 + float(tolerance)))))
    profile = {}
    for lag in range(low, high + 1):
        profile[lag] = rotation_similarity(signatures, lag, sample_count=sample_count)
    best_lag, best_score = _best_candidate(profile, round(expected))
    return int(best_lag), float(best_score), profile


def find_segment_start(signatures, guess_pages, turn_pages, window_pages=None,
                       sample_count=SEGMENT_SAMPLE_COUNT):
    """Best start page of a segment whose offset is known to about ``guess_pages``.

    Scores ``NCC(page k, page k+turn)`` for candidate starts inside the window and
    returns ``(start, score, profile)``.
    """
    turn = max(2, int(turn_pages))
    window = max(1, int(window_pages if window_pages is not None
                         else round(turn * SEGMENT_SEARCH_TOLERANCE)))
    n_pages = int(signatures.shape[0])
    low = max(0, int(round(guess_pages)) - window)
    high = max(low, min(n_pages - 1, int(round(guess_pages)) + window))
    profile = {}
    for start in range(low, high + 1):
        # Pairs must stay inside the segment AND be separated by a full turn.
        profile[start] = rotation_similarity(
            signatures, turn, start=start, stop=min(n_pages - turn, start + turn),
            sample_count=sample_count,
        )
    best_start, best_score = _best_candidate(profile, int(round(guess_pages)))
    return int(best_start), float(best_score), profile


def find_turn_broad(signatures, sample_count=SEGMENT_SAMPLE_COUNT, min_lag=16,
                    max_fraction=0.98, prominence=0.05):
    """Strongest *peaked* turn correlation over every plausible lag.

    Neighbouring pages always correlate (the sample changes slowly with the angle),
    so a plain argmax would return a tiny lag. Only local maxima that stand
    ``prominence`` above their own neighbourhood count as a period, which is what
    distinguishes a repeat from gradual decorrelation. Returns ``(pages, score)``.
    """
    n_pages = int(signatures.shape[0])
    limit = max(min_lag + 2, int(n_pages * float(max_fraction)))
    if n_pages < 4 * min_lag:
        return 0, float("nan")
    step = max(1, int(round((limit - min_lag) / 2000.0)))
    scores = {}
    for lag in range(min_lag, limit + 1, step):
        value = rotation_similarity(signatures, lag, sample_count=sample_count)
        if not math.isnan(value):
            scores[lag] = value
    if not scores:
        return 0, float("nan")

    best_lag, best_score = 0, float("nan")
    for lag, value in scores.items():
        window = max(2 * step, int(round(lag * 0.01)))
        left, right = lag - window, lag + window
        if left < min_lag or right > limit:
            continue                          # a peak has to be interior
        neighbours = [scores[key] for key in (left, right) if key in scores]
        if not neighbours or value - max(neighbours) < prominence:
            continue
        if math.isnan(best_score) or value > best_score:
            best_lag, best_score = lag, value
    if not best_lag:
        return 0, float("nan")

    window = max(2 * step, int(round(best_lag * 0.01)))
    for lag in range(max(min_lag, best_lag - window), best_lag + window + 1):
        value = rotation_similarity(signatures, lag, sample_count=sample_count)
        if not math.isnan(value) and value > best_score:
            best_lag, best_score = lag, value
    return int(best_lag), float(best_score)


def _prefer_fundamental_turn(signatures, turn_pages, turn_score,
                             factor=SEGMENT_FUNDAMENTAL_FACTOR,
                             sample_count=SEGMENT_SAMPLE_COUNT):
    """Divide the turn by 2, 3 or 4 while a shorter period correlates as well.

    A correlation peak at two or three turns is exactly as strong as the
    fundamental one, so a parameter that is off by an integer factor would
    otherwise produce segments several turns long, plus a wrong
    B-lines-per-180-deg figure. Returns ``(pages, score)``.
    """
    pages = int(turn_pages)
    score = float(turn_score)
    if math.isnan(score) or score <= 0.0:
        return pages, score
    for _ in range(4):                        # allow T -> T/2 -> T/4 -> T/8
        improved = False
        for divisor in (2, 3, 4):
            centre = int(round(pages / float(divisor)))
            if centre < 16 or centre >= pages:
                continue
            window = max(2, int(round(centre * 0.01)))
            profile = {
                lag: rotation_similarity(signatures, lag, sample_count=sample_count)
                for lag in range(centre - window, centre + window + 1)
            }
            candidate, candidate_score = _best_candidate(profile, centre)
            if (not math.isnan(candidate_score)
                    and candidate_score >= max(SEGMENT_WEAK_SCORE, factor * score)):
                pages, score = int(candidate), float(candidate_score)
                improved = True
                break
        if not improved:
            break
    return pages, score


def detect_segments(
    data,
    half_turn_blines,
    max_segments=SEGMENT_MAX_COUNT,
    tolerance=SEGMENT_SEARCH_TOLERANCE,
    signature_size=SEGMENT_SIGNATURE_SIZE,
    remove_rotational_mean=True,
    progress=None,
    depth_window=None,
):
    """Find the full-turn segments of a ``(pages, x, z)`` recording.

    Thin wrapper around :func:`detect_segments_from_signatures` that builds the
    signatures from the loaded stack first. ``depth_window`` limits those
    signatures to the depth rows that hold the sample, so a stack that still
    carries the (non-rotating) sample holder is segmented from the sample alone;
    the stack itself is not modified.
    """
    signatures = page_signatures(
        data, size=signature_size, remove_rotational_mean=remove_rotational_mean,
        progress=progress, depth_window=depth_window,
    )
    return detect_segments_from_signatures(
        signatures, half_turn_blines, page_count=int(data.shape[0]),
        max_segments=max_segments, tolerance=tolerance,
    )


def detect_segments_from_signatures(
    signatures,
    half_turn_blines,
    page_count=None,
    max_segments=SEGMENT_MAX_COUNT,
    tolerance=SEGMENT_SEARCH_TOLERANCE,
):
    """Locate the full-turn segments from an already built signature stack.

    Returns a plan dict with ``offsets`` (start page of each segment), ``lengths``
    (pages, capped at one turn), ``is_full`` flags, the phases (degrees of each
    segment's first page), the per-segment scores, the searched ``curve`` for the
    UI and human-readable ``notes``. Nothing is reconstructed here, and nothing is
    averaged: a full turn already contains every direction once.
    """
    signatures = np.asarray(signatures, dtype=np.float32)
    n_pages = int(signatures.shape[0] if page_count is None else page_count)
    n_half = max(1, int(round(float(half_turn_blines))))
    expected_turn = 2 * n_half
    min_length = max(1, int(round(expected_turn * SEGMENT_MIN_LENGTH_FRACTION)))

    notes = []
    if n_pages < min_length:
        notes.append(f"Only {n_pages} page(s): too short to segment; using one segment.")
        length = int(min(n_pages, expected_turn))
        return {
            "offsets": [0],
            "lengths": [length],
            "is_full": [bool(length >= expected_turn)],
            "phases_deg": [0.0],
            "scores": [float("nan")],
            "nominal": False,
            "turn_pages": int(expected_turn),
            "turn_score": float("nan"),
            "half_turn_blines": int(n_half),
            "expected_turn_pages": int(expected_turn),
            "curve": {"turn": {}, "starts": []},
            "page_curve": {"pages": [], "scores": [], "lag": 0},
            "notes": notes,
            "pages_total": int(n_pages),
            "pages_used": int(min(length, expected_turn)),
            "pages_discarded": int(max(0, n_pages - expected_turn)),
        }

    turn_pages, turn_score, turn_profile = find_turn_length(
        signatures, expected_turn, tolerance=tolerance,
    )
    nominal = False
    if math.isnan(turn_score) or turn_score < SEGMENT_WEAK_SCORE:
        wide_pages, wide_score, wide_profile = find_turn_length(
            signatures, expected_turn, tolerance=SEGMENT_WIDE_TOLERANCE,
        )
        if not math.isnan(wide_score) and wide_score >= SEGMENT_WEAK_SCORE:
            notes.append(
                f"Weak peak inside +/-{tolerance * 100:.0f}% of the parameter "
                f"({turn_score:.3f}); widened to "
                f"+/-{SEGMENT_WIDE_TOLERANCE * 100:.0f}% and found {wide_pages} "
                f"pages per turn ({wide_score:.3f})."
            )
            turn_pages, turn_score, turn_profile = wide_pages, wide_score, wide_profile
        else:
            score_text = "not scoreable" if math.isnan(wide_score) else f"{wide_score:.3f}"
            broad_pages, broad_score = find_turn_broad(signatures)
            if broad_pages and not math.isnan(broad_score) and broad_score >= SEGMENT_WEAK_SCORE:
                notes.append(
                    f"No usable peak near the parameter (best {score_text}); a full "
                    f"search found {broad_pages} pages per turn "
                    f"({broad_score:.3f}) - the parameter is off by a factor or more."
                )
                turn_pages, turn_score, turn_profile = broad_pages, broad_score, turn_profile
            else:
                notes.append(
                    f"No usable turn-to-turn correlation (best score {score_text}); "
                    f"falling back to the parameter: {expected_turn} pages per turn. "
                    f"The segment boundaries are then nominal, so check B-lines per "
                    f"180 deg by hand."
                )
                turn_pages, turn_score, turn_profile = expected_turn, wide_score, wide_profile
                nominal = True
    fundamental_pages, fundamental_score = _prefer_fundamental_turn(
        signatures, turn_pages, turn_score,
    )
    if fundamental_pages < turn_pages:
        notes.append(
            f"The peak is a multiple of the true period: {turn_pages} pages turned "
            f"out to be {turn_pages / max(1, fundamental_pages):.0f} turns, so the "
            f"fundamental turn is {fundamental_pages} pages "
            f"({fundamental_pages / 2.0:.1f} B-lines per 180 deg, score "
            f"{fundamental_score:.3f})."
        )
        turn_pages, turn_score = fundamental_pages, fundamental_score

    if not nominal and turn_pages != expected_turn:
        notes.append(
            f"Correlation peak at {turn_pages} pages per turn "
            f"({turn_pages / 2.0:.1f} B-lines per 180 deg) instead of the "
            f"{expected_turn} pages implied by the parameter; set B-lines per "
            f"180 deg to {turn_pages / 2.0:.0f} and press \"Detect now\" to search "
            f"again with the narrower window."
        )

    offsets = [0]
    scores = [float("nan")]  # the first segment has no earlier twin to compare
    starts_curve = []
    # How short a trailing stretch may be before it is treated as noise: this uses
    # the turn that was actually found, not the parameter.
    min_length = max(1, int(round(turn_pages * SEGMENT_MIN_LENGTH_FRACTION)))
    while len(offsets) < max(1, int(max_segments)):
        previous = offsets[-1]
        guess = previous + turn_pages
        if n_pages - guess < min_length:
            break
        start, score, profile = find_segment_start(
            signatures, guess, turn_pages,
            window_pages=int(round(turn_pages * tolerance)),
        )
        starts_curve.append(profile)
        if (not math.isnan(turn_score) and not math.isnan(score)
                and score < 0.5 * turn_score):
            notes.append(
                f"Segment {len(offsets) + 1} at page {start} matches only weakly "
                f"(score {score:.3f} vs {turn_score:.3f}); kept but flagged."
            )
        offsets.append(int(start))
        scores.append(float(score))

    # One half-turn of B-lines already fills 360 degrees, so only the first
    # half_turn_blines pages of a segment are needed for the reconstruction; the
    # rest of that turn is the repeat pass.
    lengths = []
    is_full = []
    for index, offset in enumerate(offsets):
        available = max(0, n_pages - offset)
        length = int(min(turn_pages, available))
        if index > 0:
            length = int(min(length, offsets[index] - offsets[index - 1]))
        lengths.append(length)
        is_full.append(bool(length >= turn_pages * SEGMENT_FULL_FRACTION))

    phases = [(offset * 180.0 / float(n_half)) % 360.0 for offset in offsets]
    pages_used = int(sum(min(length, expected_turn) for length in lengths))
    if not any(is_full):
        notes.append("No complete turn was found; the longest segment is partial.")

    # Per-page version of the turn correlation, drawn by the window: it stays high
    # inside a segment and drops where consecutive pages stop lining up.
    curve_pages = [int(value)
                   for value in _sample_page_indices(0, max(1, n_pages - turn_pages), 240)
                   if value + turn_pages < n_pages]
    page_curve = {
        "pages": curve_pages,
        "scores": [float(np.dot(signatures[index].ravel(),
                                signatures[index + turn_pages].ravel()))
                   for index in curve_pages],
        "lag": int(turn_pages),
    }

    return {
        "offsets": [int(value) for value in offsets],
        "lengths": [int(value) for value in lengths],
        "is_full": [bool(value) for value in is_full],
        "phases_deg": [float(value) for value in phases],
        "scores": [float(value) for value in scores],
        "turn_pages": int(turn_pages),
        "turn_score": float(turn_score),
        "nominal": bool(nominal),
        "half_turn_blines": int(n_half),
        "expected_turn_pages": int(expected_turn),
        "curve": {"turn": dict(turn_profile), "starts": starts_curve},
        "page_curve": page_curve,
        "notes": notes,
        "pages_total": int(n_pages),
        "pages_used": int(pages_used),
        "pages_discarded": int(max(0, n_pages - pages_used)),
    }


def segment_page_window(plan, index, half_turn_blines):
    """Pages to hand to :func:`build_half_turn_stacks` for segment ``index``.

    A segment holds a full turn, but half a turn already covers 360 degrees, so
    the window is ``half_turn_blines`` pages long and simply starts at the
    segment's first page. Returns ``(offset, count)``; ``(0, None)`` means "first
    pages of the file", i.e. the historical behaviour.
    """
    n_half = max(1, int(round(float(half_turn_blines))))
    if not plan:
        return 0, None
    offsets = plan.get("offsets", [])
    if index is None or index < 0 or index >= len(offsets):
        return 0, None
    offset = int(offsets[index])
    if index == 0 and offset <= 0:
        return 0, None
    return offset, int(n_half)


def describe_segment_plan(plan):
    """One-line summary of a segment plan for the status label / JSON sidecar.

    Every segment is listed with the B-lines it spans (``B-lines a..b (n)``), so the
    ranges can be checked against the recording: one page is one B-line here, and a
    segment that looks too short (or that overlaps its neighbour) is visible before
    anything is stacked.
    """
    if not plan:
        return "no segment plan"
    parts = []
    for index, offset in enumerate(plan.get("offsets", [])):
        lengths = plan.get("lengths", [])
        scores = plan.get("scores", [])
        is_full = plan.get("is_full", [])
        length = int(lengths[index]) if index < len(lengths) else 0
        score = scores[index] if index < len(scores) else float("nan")
        flag = "" if (index < len(is_full) and is_full[index]) else "/partial"
        score_text = "--" if math.isnan(score) else f"{score:.2f}"
        parts.append(f"#{index + 1} B-lines {int(offset)}..{int(offset) + max(0, length - 1)}"
                     f" ({length}{flag}, ncc {score_text})")
    return (
        "segments " + "  ".join(parts)
        + f" | turn {int(plan.get('turn_pages', 0))} pages"
        + f" | used {int(plan.get('pages_used', 0))}/{int(plan.get('pages_total', 0))}"
        + (" | nominal (weak correlation)" if plan.get("nominal") else "")
    )



# ===== STAGE 2 - commented out while only the segmentation is developed =====
# # --------------------------------------------------------------------------- #
# # Cartesian reconstruction
# # --------------------------------------------------------------------------- #
#
# def radial_pixel_limit(x_pixels, xc):
#     """Largest symmetric radius, in B-line pixels, fully covered by the scan."""
#     r_max = int(math.floor(min(xc, (x_pixels - 1) - xc)))
#     if r_max < 1:
#         raise ValueError(
#             "Rotation centre must stay at least one pixel away from the B-line "
#             f"edges; got Xc={xc:.3f} for X={x_pixels}."
#         )
#     return r_max
#
#
# def plan_cartesian_grid(
#     x_pixels,
#     xc,
#     z_pixels,
#     decimate=1,
#     max_grid=MAX_GRID_SIZE,
#     byte_limit=VOLUME_BYTE_LIMIT,
# ):
#     """Choose an isotropic output grid, shrinking it until it fits the limits.
#
#     The grid spacing starts at the native radial sampling so the XY view is
#     neither over- nor under-sampled. Both the edge length and the resulting
#     float32 volume size are bounded.
#     """
#     r_max = radial_pixel_limit(x_pixels, xc)
#     requested = max(1, int(decimate))
#     step = requested
#     while step <= r_max:
#         n_grid = 2 * (r_max // step) + 1
#         if n_grid <= max_grid and n_grid * n_grid * z_pixels * 4 <= byte_limit:
#             notes = []
#             if step != requested:
#                 notes.append(
#                     f"Decimation raised to {step} to respect the grid/memory limits."
#                 )
#             return n_grid, float(step), notes
#         step += 1
#
#     # Even the coarsest decimation is too large: clamp the edge length instead,
#     # which leaves a fractional spacing. The grid stays odd so the rotation axis
#     # lands exactly on the centre pixel.
#     n_grid = int(math.sqrt(max(1.0, float(byte_limit) / 4.0 / max(1, z_pixels))))
#     n_grid = max(3, min(max_grid, n_grid))
#     if n_grid % 2 == 0:
#         n_grid -= 1
#     n_grid = max(3, n_grid)
#     spacing = (2.0 * r_max) / max(1, n_grid - 1)
#     return n_grid, spacing, [f"Grid clamped to {n_grid} px ({spacing:.3f} px spacing)."]
#
#
# def build_sampling_geometry(n_grid, spacing, n_bins, n_slots, slot_of_bin, r_max):
#     """Precompute the polar->Cartesian gather indices and weights.
#
#     The geometry does not depend on depth, so it is built once per Apply and
#     reused for every slice, which is what keeps the reconstruction fast.
#     """
#     centre = (n_grid - 1) / 2.0
#     offsets = np.arange(n_grid, dtype=np.float64) - centre
#     u, v = np.meshgrid(offsets, offsets)
#     radius = np.hypot(u, v) * spacing
#     angle = np.degrees(np.arctan2(v, u)) % 360.0
#
#     bin_width = 360.0 / float(n_bins)
#     position = angle / bin_width
#     base = np.floor(position)
#     weight_angle = (position - base).astype(np.float32)
#     bin_a = base.astype(np.int64) % n_bins
#     bin_b = (bin_a + 1) % n_bins
#
#     dense_slot = np.full(n_bins, -1, dtype=np.int64)
#     for bin_index, slot in slot_of_bin.items():
#         dense_slot[bin_index] = slot
#     slot_a = dense_slot[bin_a]
#     slot_b = dense_slot[bin_b]
#
#     only_a = (slot_a >= 0) & (slot_b < 0)
#     only_b = (slot_b >= 0) & (slot_a < 0)
#     empty = (slot_a < 0) & (slot_b < 0)
#     valid = (~empty) & (radius <= float(r_max))
#
#     slot_a = np.where(slot_a >= 0, slot_a, np.where(slot_b >= 0, slot_b, 0))
#     slot_b = np.where(slot_b >= 0, slot_b, np.where(slot_a >= 0, slot_a, 0))
#     weight_angle = np.where(only_a, np.float32(0.0), weight_angle)
#     weight_angle = np.where(only_b, np.float32(1.0), weight_angle)
#
#     radial_base = np.floor(radius)
#     weight_radius = (radius - radial_base).astype(np.float32)
#     radial_low = np.clip(radial_base.astype(np.int64), 0, r_max)
#     radial_high = np.minimum(radial_low + 1, r_max)
#
#     return {
#         "n_grid": int(n_grid),
#         "spacing": float(spacing),
#         "slot_a": slot_a.ravel(),
#         "slot_b": np.clip(slot_b, 0, max(0, n_slots - 1)).ravel(),
#         "weight_angle": weight_angle.ravel(),
#         "radial_low": radial_low.ravel(),
#         "radial_high": radial_high.ravel(),
#         "weight_radius": weight_radius.ravel(),
#         "valid": valid.ravel(),
#     }
#
#
# def reconstruct_volume(stack, geometry, progress=None):
#     """Resample the binned polar stack onto the Cartesian grid."""
#     n_grid = geometry["n_grid"]
#     n_pixels = n_grid * n_grid
#     z_pixels = stack.shape[2]
#
#     volume = np.zeros((n_grid, n_grid, z_pixels), dtype=np.float32)
#
#     slot_a = geometry["slot_a"]
#     slot_b = geometry["slot_b"]
#     wa = geometry["weight_angle"]
#     r0 = geometry["radial_low"]
#     r1 = geometry["radial_high"]
#     wr = geometry["weight_radius"]
#     valid = geometry["valid"]
#
#     one_minus_wa = (1.0 - wa).astype(np.float32)[:, None]
#     one_minus_wr = (1.0 - wr).astype(np.float32)[:, None]
#     chunk = max(1, min(z_pixels, int(GATHER_BUDGET_ELEMENTS // max(1, n_pixels))))
#     wa = wa[:, None]
#     wr = wr[:, None]
#
#     for start in range(0, z_pixels, chunk):
#         stop = min(start + chunk, z_pixels)
#         if progress is not None:
#             progress(start, z_pixels, f"Reconstructing depth {start + 1}-{stop}/{z_pixels}")
#         a_low = stack[slot_a, r0, start:stop]
#         a_high = stack[slot_a, r1, start:stop]
#         b_low = stack[slot_b, r0, start:stop]
#         b_high = stack[slot_b, r1, start:stop]
#         blended = (one_minus_wr * ((one_minus_wa * a_low) + wa * b_low)
#                    + wr * ((one_minus_wa * a_high) + wa * b_high))
#         blended[~valid] = 0.0
#         volume[:, :, start:stop] = blended.reshape(n_grid, n_grid, stop - start)
#
#     return volume
#
#
# # --------------------------------------------------------------------------- #
# # Export
# # --------------------------------------------------------------------------- #
#
# def default_volume_filename(source_path, n_grid, z_pixels, directory=None):
#     """Build ``RotVol-<num>-Y<N>-X<N>-Z<Z>.tif`` next to the source scan."""
#     source = Path(source_path)
#     info = parse_cscan_filename(source)
#     number = info["num"] if info else 1
#     name = f"RotVol-{number}-Y{n_grid}-X{n_grid}-Z{int(z_pixels)}.tif"
#     folder = Path(directory) if directory else source.parent
#     return folder / name
#
#
# def export_volume(volume, path, meta, params):
#     """Write the float32 volume as a multi-page TIFF plus a JSON sidecar."""
#     path = Path(path)
#     data = np.ascontiguousarray(volume, dtype=np.float32)
#     TIFF.imwrite(str(path), data, photometric="minisblack")
#
#     sidecar = path.with_suffix(".json")
#     segments = params.get("segments") or {}
#     phases = [float(value) for value in (segments.get("phases_deg") or [])]
#     segment_index = int(params.get("segment_index", 0))
#     start_angle = phases[segment_index] if 0 <= segment_index < len(phases) else 0.0
#     scores = segments.get("scores") or []
#     turn_score = segments.get("turn_score")
#     payload = {
#         "source_files": list(meta.get("files", [])),
#         "volume_shape_YXZ": [int(value) for value in data.shape],
#         "lateral_pixel_size_um": float(meta.get("x_step_um", LATERAL_PIXEL_SIZE_UM)),
#         "depth_window_pixels": [int(value) for value in (meta.get("depth_window") or [])],
#         # Every depth row is loaded now: "used" is the window, not the shape.
#         "depth_pixels_used": int(meta.get("depth_window", [0, data.shape[2]])[1]
#                                  - meta.get("depth_window", [0, data.shape[2]])[0]),
#         "depth_pixels_full": int(meta.get("z_pixels_full", data.shape[2])),
#         "half_turn_blines": float(params.get("half_turn_blines", 0.0)),
#         "rotation_centre_pixel": float(params.get("xc", 0.0)),
#         "degrees_per_bline": float(params.get("degrees_per_bline", 0.0)),
#         "angular_samples": int(params.get("angular_samples", 0)),
#         "pages_used": int(params.get("pages_used", 0)),
#         "pages_total": int(params.get("pages_total", 0)),
#         "pages_discarded": int(params.get("pages_discarded", 0)),
#         "grid_spacing_pixels": float(params.get("spacing", 1.0)),
#         "grid_spacing_um": float(params.get("spacing", 1.0)) * float(meta.get("x_step_um", LATERAL_PIXEL_SIZE_UM)),
#         "rotation_direction": "counter-clockwise",
#         "start_angle_deg": float(start_angle),
#         "page_offset": int(params.get("page_offset", 0)),
#         "page_count": int(params.get("page_count", 0)),
#         "segment_index": segment_index,
#         "segment_count": len(segments.get("offsets") or []),
#         "segment_offsets": [int(value) for value in (segments.get("offsets") or [])],
#         "segment_lengths": [int(value) for value in (segments.get("lengths") or [])],
#         "segment_is_full": [bool(value) for value in (segments.get("is_full") or [])],
#         "segment_phases_deg": phases,
#         "segment_scores": [None if value is None else float(value) for value in scores],
#         "segment_turn_pages": int(segments.get("turn_pages", 0)),
#         "segment_turn_score": (None if turn_score is None else float(turn_score)),
#         "segment_nominal": bool(segments.get("nominal", False)),
#         "segment_notes": [str(value) for value in (segments.get("notes") or [])],
#     }
#     with open(str(sidecar), "w", encoding="utf-8") as handle:
#         json.dump(payload, handle, indent=2, sort_keys=True)
#     return path, sidecar
#
#
# --------------------------------------------------------------------------- #
# Background workers
# --------------------------------------------------------------------------- #

class _Cancelled(Exception):
    """Raised inside a progress callback to abort a running worker."""


class LoadWorker(QThread):
    """Read and stack the C-scan pages without blocking the GUI."""

    progressed = pyqtSignal(int, int, str)
    loaded = pyqtSignal(object, object)
    failed = pyqtSignal(str)

    def __init__(self, paths, depth_window=DEFAULT_DEPTH_WINDOW, parent=None):
        super().__init__(parent)
        self._paths = list(paths)
        self._depth_window = (None if depth_window is None else tuple(depth_window))
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        def report(index, total, message):
            if self._cancelled:
                raise _Cancelled()
            self.progressed.emit(int(index), int(total), message)

        try:
            data, meta = load_scan(self._paths, progress=report,
                                   depth_window=self._depth_window)
        except _Cancelled:
            self.failed.emit("cancelled")
        except Exception as error:  # noqa: BLE001 - surfaced in the GUI
            self.failed.emit(f"{error}")
        else:
            self.loaded.emit(data, meta)


class DetectWorker(QThread):
    """Search the loaded stack for repeated turns, without reconstructing.

    The search measures the depth rows in ``depth_window`` only (the sample); the
    stack is complete and stays as it is.
    """

    progressed = pyqtSignal(int, int, str)
    detected = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, data, half_turn_blines, depth_window=None, parent=None):
        super().__init__(parent)
        self._data = data
        self._half_turn_blines = float(half_turn_blines)
        self._depth_window = (None if depth_window is None else tuple(depth_window))
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        def report(count, total, message):
            if self._cancelled:
                raise _Cancelled()
            self.progressed.emit(int(count), int(total), message)

        try:
            plan = detect_segments(self._data, self._half_turn_blines,
                                   progress=report,
                                   depth_window=self._depth_window)
        except _Cancelled:
            self.failed.emit("cancelled")
        except Exception as error:  # noqa: BLE001 - surfaced in the GUI
            self.failed.emit(f"{error}")
        else:
            self.detected.emit(plan)


# ===== STAGE 2 - commented out while only the segmentation is developed =====
# class ReconstructWorker(QThread):
#     """Bin, plan the grid and reconstruct the volume off the GUI thread."""
#
#     progressed = pyqtSignal(int, int, str)
#     detected = pyqtSignal(object)
#     done = pyqtSignal(object, object, object, object)
#     failed = pyqtSignal(str)
#
#     def __init__(self, data, xc, half_turn_blines, decimate, detect=True,
#                  segment_index=0, parent=None):
#         super().__init__(parent)
#         self._data = data
#         self._xc = float(xc)
#         self._half_turn_blines = float(half_turn_blines)
#         self._decimate = int(decimate)
#         self._detect = bool(detect)
#         self._segment_index = int(segment_index)
#         self._cancelled = False
#
#     def cancel(self):
#         self._cancelled = True
#
#     def run(self):
#         def report(count, total, message):
#             if self._cancelled:
#                 raise _Cancelled()
#             self.progressed.emit(int(count), int(total), message)
#
#         try:
#             data = self._data
#             x_pixels, z_pixels = data.shape[1], data.shape[2]
#             plan = None
#             if self._detect:
#                 report(0, 1, "Looking for the rotation repeats...")
#                 plan = detect_segments(data, self._half_turn_blines, progress=report)
#                 # Report the boundaries straight away: the reconstruction of a
#                 # full-size file takes minutes, the segmentation only seconds.
#                 self.detected.emit(plan)
#             page_offset, page_count = segment_page_window(
#                 plan, self._segment_index, self._half_turn_blines
#             )
#             stack, slot_pages, slot_angles, slot_of_bin, info = build_half_turn_stacks(
#                 data, self._xc, self._half_turn_blines, page_offset=page_offset,
#                 page_count=page_count, progress=report,
#             )
#             planar_grid, spacing, notes = plan_cartesian_grid(
#                 x_pixels, self._xc, z_pixels, decimate=self._decimate
#             )
#             angular_samples = int(info["angular_samples"])
#             geometry = build_sampling_geometry(
#                 planar_grid, spacing, angular_samples, angular_samples,
#                 slot_of_bin, info["radial_pixels"],
#             )
#             volume = reconstruct_volume(stack, geometry, progress=report)
#         except _Cancelled:
#             self.failed.emit("cancelled")
#         except Exception as error:  # noqa: BLE001 - surfaced in the GUI
#             self.failed.emit(f"{error}")
#         else:
#             info = dict(info)
#             info["n_grid"] = int(planar_grid)
#             info["spacing"] = float(spacing)
#             info["notes"] = list(notes)
#             info["grid_um"] = float(spacing) * LATERAL_PIXEL_SIZE_UM
#             info["radius_um"] = float(info["radial_pixels"]) * LATERAL_PIXEL_SIZE_UM
#             info["segment_index"] = int(self._segment_index)
#             info["segment_plan"] = plan
#             self.done.emit(volume, slot_pages, slot_angles, info)
#
#
# --------------------------------------------------------------------------- #
# Synthetic self-test
# --------------------------------------------------------------------------- #

def _sample_plane(image, xs, ys):
    """Bilinear sample of a 2D image at fractional (column, row) positions."""
    height, width = image.shape
    x0 = np.floor(xs).astype(np.int64)
    y0 = np.floor(ys).astype(np.int64)
    fx = xs - x0
    fy = ys - y0
    x0c = np.clip(x0, 0, width - 1)
    x1c = np.clip(x0 + 1, 0, width - 1)
    y0c = np.clip(y0, 0, height - 1)
    y1c = np.clip(y0 + 1, 0, height - 1)
    top = (1.0 - fx) * image[y0c, x0c] + fx * image[y0c, x1c]
    bottom = (1.0 - fx) * image[y1c, x0c] + fx * image[y1c, x1c]
    return (1.0 - fy) * top + fy * bottom


def _synthetic_volume(n_grid, z_pixels):
    """Smooth blobs whose brightness also varies with depth."""
    centre = (n_grid - 1) / 2.0
    rows, cols = np.mgrid[0:n_grid, 0:n_grid].astype(np.float64)
    blobs = [
        (centre - 18.0, centre + 12.0, 7.0, 1.0),
        (centre + 14.0, centre - 16.0, 9.0, 0.65),
        (centre + 22.0, centre + 20.0, 5.0, 0.8),
    ]
    base = np.zeros((n_grid, n_grid), dtype=np.float64)
    for blob_y, blob_x, sigma, amplitude in blobs:
        base += amplitude * np.exp(
            -(((rows - blob_y) ** 2) + ((cols - blob_x) ** 2)) / (2.0 * sigma * sigma)
        )
    depth = 0.6 + 0.4 * np.cos(np.linspace(0.0, np.pi, z_pixels))
    return (base[:, :, None] * depth[None, None, :]).astype(np.float32)


def _project_pages(volume, xc, half_turn_blines, pages=None, noise=0.0, seed=0):
    """Forward-project a Cartesian volume into rotational B-line pages.

    This is the exact inverse of the reconstruction's sampling model, which is
    what makes the round-trip check meaningful: column ``xc + r`` holds the
    profile at the page angle and column ``xc - r`` the profile 180 degrees
    opposite. ``pages`` defaults to a full turn, i.e. twice the half-turn count,
    so the tests exercise a file that contains the repeat pass as well.
    """
    n_grid, _, z_pixels = volume.shape
    centre = (n_grid - 1) / 2.0
    r_max = int(centre)
    column = int(round(xc))
    n_pages = int(2 * half_turn_blines if pages is None else pages)
    step_deg = 180.0 / float(half_turn_blines)
    radii = np.arange(0, r_max + 1, dtype=np.float64)
    rng = np.random.default_rng(seed)

    result = np.zeros((n_pages, n_grid, z_pixels), dtype=np.float32)
    for page_index in range(n_pages):
        angle = math.radians(page_index * step_deg)
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        forward_x = centre + radii * cos_a
        forward_y = centre + radii * sin_a
        backward_x = centre - radii * cos_a
        backward_y = centre - radii * sin_a
        for z in range(z_pixels):
            plane = volume[:, :, z]
            result[page_index, column:column + r_max + 1, z] = _sample_plane(
                plane, forward_x, forward_y
            )
            result[page_index, column - r_max:column + 1, z] = _sample_plane(
                plane, backward_x, backward_y
            )[::-1]
    if noise:
        result += rng.normal(0.0, noise, size=result.shape).astype(np.float32)
    return result


# ===== STAGE 2 - commented out while only the segmentation is developed =====
# def _legacy_averaged_stacks(pages, xc, half_turn):
#     """Test-only: the removed behaviour, which averaged the two repeat passes.
#
#     Kept purely so the self-test can demonstrate why the averaging was dropped.
#     The reconstruction itself never calls this. Direction ``j`` is measured by
#     the right half of page ``j`` and by the left half of the page half a turn
#     away; the removed code blended those two mutually offset measurements.
#     """
#     x_pixels = pages.shape[1]
#     n_slots = 2 * half_turn
#     r_max = radial_pixel_limit(x_pixels, xc)
#     radial = np.arange(0, r_max + 1, dtype=np.float64)
#     right = np.zeros((n_slots, radial.size, pages.shape[2]), dtype=np.float32)
#     left = np.zeros_like(right)
#     for page_index in range(min(n_slots, pages.shape[0])):
#         page = pages[page_index]
#         right[page_index] = resample_columns(page, xc + radial, x_pixels)
#         left[page_index] = resample_columns(
#             page, np.clip(xc - radial, 0.0, float(x_pixels - 1)), x_pixels
#         )
#     return 0.5 * (right + np.roll(left, half_turn, axis=0))
#
#
def run_self_test():
    """Round-trip a synthetic volume and report the checks."""
    print("Rotational reconstruction self-test")
    print("-" * 60)

    results = []

    def check(name, ok, detail=""):
        results.append((name, bool(ok)))
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" :: {detail}" if detail else ""))

    radius = 64
    n_grid = 2 * radius + 1
    z_pixels = 16
    axis_true = float(radius)
    half_turn = 64                        # B-lines spanning 180 degrees

    truth = _synthetic_volume(n_grid, z_pixels)
    scale = float(np.max(truth))

    # ===== STAGE 2 - commented out while only the segmentation is developed =====
#     def rebuild(stack, centre):
#         r_max = radial_pixel_limit(n_grid, centre)
#         grid, spacing, notes = plan_cartesian_grid(n_grid, centre, z_pixels, decimate=1)
#         slots = stack.shape[0]
#         geometry = build_sampling_geometry(
#             grid, spacing, slots, slots, {s: s for s in range(slots)}, r_max
#         )
#         return reconstruct_volume(stack, geometry), grid, spacing, notes
#
#     def embed(volume, size=n_grid):
#         """Centre a smaller reconstruction grid inside the full-size grid.
#
#         A displaced centre leaves a smaller usable radius, so the grid shrinks;
#         padding it back (both grids are centred on the assumed axis) keeps the
#         shift itself as the error rather than a shape mismatch.
#         """
#         if volume.shape[0] == size:
#             return volume
#         pad = (size - volume.shape[0]) // 2
#         out = np.zeros((size, size, volume.shape[2]), dtype=volume.dtype)
#         out[pad:pad + volume.shape[0], pad:pad + volume.shape[0], :] = volume
#         return out
#
#     def error_of(volume):
#         return float(np.sqrt(np.mean((embed(volume) - truth) ** 2))) / scale
#
#     def sharpness_of(volume):
#         return float(np.mean(np.abs(np.diff(volume, axis=0)))
#                      + np.mean(np.abs(np.diff(volume, axis=1))))
#
#     # ------------------------------------------------------------------ #
#     # Structure: one half-turn in, 360 degrees out, nothing averaged
#     # ------------------------------------------------------------------ #
#     pages = _project_pages(truth, axis_true, half_turn)   # a full turn on disk
#     stack, slot_pages, slot_angles, slot_of_bin, info = build_half_turn_stacks(
#         pages, axis_true, half_turn
#     )
#     check("only the first half-turn is used",
#           info["pages_used"] == half_turn and info["pages_discarded"] == half_turn,
#           f"used={info['pages_used']} discarded={info['pages_discarded']}")
#     check("2N directions cover 360 degrees",
#           info["angular_samples"] == 2 * half_turn
#           and abs(info["degrees_per_bline"] - 180.0 / half_turn) < 1e-12,
#           f"samples={info['angular_samples']} step={info['degrees_per_bline']:.4f} deg")
#     check("every direction has a B-line", info["missing_slots"] == 0)
#     check("each direction is a single B-line: nothing is averaged",
#           max(len(entry) for entry in slot_pages) == 1,
#           f"max per direction={max(len(entry) for entry in slot_pages)}")
#     check("directions are evenly spaced over 360 degrees",
#           bool(np.allclose(np.diff(slot_angles), 180.0 / half_turn, atol=1e-9)))
#
#     radial = np.arange(0, radial_pixel_limit(n_grid, axis_true) + 1, dtype=np.float64)
#     left_positions = np.clip(axis_true - radial, 0.0, float(n_grid - 1))
#     check("the left half of page k fills direction k + N",
#           slot_pages[half_turn][0] == 0
#           and np.allclose(stack[half_turn],
#                           resample_columns(pages[0], left_positions, n_grid), atol=1e-6))
#
#     volume, grid, spacing, notes = rebuild(stack, axis_true)
#     check("grid matches the native radial sampling",
#           grid == n_grid and abs(spacing - 1.0) < 1e-9,
#           f"grid={grid} spacing={spacing} notes={notes}")
#     rms = error_of(volume)
#     check("round-trip RMS error is small", rms < 0.03, f"rms={rms:.4f}")
#
#     rows, cols = np.mgrid[0:n_grid, 0:n_grid]
#     outside = ((rows - radius) ** 2 + (cols - radius) ** 2) > (radius + 1) ** 2
#     check("region beyond the data radius is exactly black",
#           float(np.max(np.abs(volume[outside]))) == 0.0)
#     check("depth axis is preserved",
#           bool(np.allclose(truth[radius, radius, :], volume[radius, radius, :], atol=1e-3)))
#
#     # ------------------------------------------------------------------ #
#     # The repeat pass must be ignored, never blended in
#     # ------------------------------------------------------------------ #
#     spoiled = pages.copy()
#     spoiled[half_turn:] = np.random.default_rng(7).random(
#         spoiled[half_turn:].shape).astype(np.float32)
#     spoiled_stack, _sp, _sa, _sob, _i = build_half_turn_stacks(spoiled, axis_true, half_turn)
#     spoiled_volume, _g, _s, _n = rebuild(spoiled_stack, axis_true)
#     check("pages beyond the half-turn are ignored completely",
#           np.array_equal(spoiled_volume, volume))
#
#     short_stack, _sp, _sa, _sob, short_info = build_half_turn_stacks(
#         pages[:half_turn - 10], axis_true, half_turn
#     )
#     check("a capture shorter than requested leaves directions empty",
#           short_info["missing_slots"] == 20
#           and short_info["pages_used"] == half_turn - 10
#           and short_info["pages_discarded"] == 0,
#           f"missing={short_info['missing_slots']} used={short_info['pages_used']}")
#
#     # ------------------------------------------------------------------ #
#     # The point of the redesign: the offset repeat pass must not be averaged
#     # ------------------------------------------------------------------ #
#     missed = axis_true - 5.0              # assumed centre sits 5 px off the axis
#     single_stack, _sp, _sa, _sob, _i = build_half_turn_stacks(pages, missed, half_turn)
#     single_volume, _g, _s, _n = rebuild(single_stack, missed)
#     legacy_stack = _legacy_averaged_stacks(pages, missed, half_turn)
#     legacy_volume, _g, _s, _n = rebuild(legacy_stack, missed)
#     single_sharp = sharpness_of(single_volume)
#     legacy_sharp = sharpness_of(legacy_volume)
#     check("the offset repeat pass is not averaged in (no double image)",
#           single_sharp > 1.05 * legacy_sharp,
#           f"sharpness single={single_sharp:.5f} averaged={legacy_sharp:.5f}")
#     check("the result really is a single pass, not a blend",
#           not np.allclose(single_volume, legacy_volume, atol=1e-6))
#
#     shifted_error = error_of(single_volume)
#     check("a wrong rotation centre degrades the result (Xc is meaningful)",
#           shifted_error > 2.0 * rms, f"centred={rms:.4f} shifted={shifted_error:.4f}")
#
    # ------------------------------------------------------------------ #
    # Multi-turn recording: every turn is located by correlation
    # ------------------------------------------------------------------ #
    recording_pages = int(round(2.7 * 2 * half_turn))
    recording = _project_pages(truth, axis_true, half_turn, pages=recording_pages,
                               noise=0.02, seed=17)
    drift = 1.0 + 0.15 * np.sin(np.linspace(0.0, 3.0 * np.pi, recording_pages))
    recording = (recording * drift[:, None, None]).astype(np.float32)

    plan = detect_segments(recording, half_turn)
    expected_offsets = [0, 2 * half_turn, 4 * half_turn]
    check("a 2.7 turn recording splits at 0 / T / 2T",
          plan["offsets"] == expected_offsets,
          f"offsets={plan['offsets']} turn={plan['turn_pages']}")
    check("the detected turn length matches the parameter",
          plan["turn_pages"] == 2 * half_turn and not plan["nominal"],
          f"turn={plan['turn_pages']} score={plan['turn_score']:.3f}")
    check("two full turns plus one partial segment",
          plan["is_full"] == [True, True, False]
          and plan["lengths"][2] == recording_pages - 4 * half_turn,
          f"lengths={plan['lengths']} full={plan['is_full']}")
    check("the segments cover every page of the recording",
          plan["pages_used"] == recording_pages and plan["pages_discarded"] == 0,
          f"used={plan['pages_used']}/{plan['pages_total']} "
          f"discarded={plan['pages_discarded']}")
    check("the plan summary lists every segment",
          describe_segment_plan(plan).count("#") == 3, describe_segment_plan(plan))

    far_off = detect_segments(recording, 55)
    check("a parameter outside the window is recovered by widening the search",
          abs(far_off["turn_pages"] - 2 * half_turn) <= 4 and far_off["notes"],
          f"turn={far_off['turn_pages']} notes={len(far_off['notes'])}")

    doubled = detect_segments(recording, 2 * half_turn)
    check("a parameter that is a multiple of the turn is halved back",
          doubled["turn_pages"] == 2 * half_turn
          and any("multiple of the true period" in note for note in doubled["notes"]),
          f"turn={doubled['turn_pages']} offsets={doubled['offsets']} "
          f"notes={doubled['notes'][:1]}")

    nowhere = detect_segments(recording, 500)
    check("a wildly wrong parameter is rescued by the full search",
          abs(nowhere["turn_pages"] - 2 * half_turn) <= 4
          and not nowhere["nominal"]
          and any("full search" in note or "multiple of the true period" in note
                  for note in nowhere["notes"]),
          f"turn={nowhere['turn_pages']} nominal={nowhere['nominal']} "
          f"notes={nowhere['notes'][:1]}")

    # ===== STAGE 2 - commented out while only the segmentation is developed =====
#     second_offset, second_count = segment_page_window(plan, 1, half_turn)
#     second_stack, _sp2, _sa2, _sob2, second_info = build_half_turn_stacks(
#         recording, axis_true, half_turn, page_offset=second_offset,
#         page_count=second_count,
#     )
#     shifted_stack, _sp3, _sa3, _sob3, _i3 = build_half_turn_stacks(
#         recording, axis_true, half_turn, page_offset=second_offset + 5,
#         page_count=second_count,
#     )
#     aligned_error = float(np.mean(np.abs(second_stack - stack)))
#     shifted_difference = float(np.mean(np.abs(shifted_stack - stack)))
#     check("page_offset really selects the repeat pass",
#           second_info["page_window"] == [2 * half_turn, 3 * half_turn]
#           and aligned_error < shifted_difference,
#           f"window={second_info['page_window']} "
#           f"aligned={aligned_error:.4f} shifted={shifted_difference:.4f}")

    radii = np.arange(n_grid, dtype=np.float64) - axis_true
    profile = (np.exp(-(radii ** 2) / (2.0 * 20.0 ** 2))[:, None]
               * np.linspace(1.0, 0.5, z_pixels)[None, :])
    alike = np.tile(profile[None, :, :].astype(np.float32), (recording_pages, 1, 1))
    alike += np.random.default_rng(23).normal(0.0, 0.01, alike.shape).astype(np.float32)
    flat_plan = detect_segments(alike, half_turn)
    check("a rotation invariant recording falls back to the nominal turn",
          flat_plan["nominal"] and flat_plan["turn_pages"] == 2 * half_turn
          and any("No usable turn-to-turn correlation" in note
                  for note in flat_plan["notes"]),
          f"nominal={flat_plan['nominal']} turn={flat_plan['turn_pages']} "
          f"score={flat_plan['turn_score']:.4f}")

    # ------------------------------------------------------------------ #
    # Depth window: it steers the measurements, it never crops the data
    # ------------------------------------------------------------------ #
    check("the default depth window keeps the sample rows only",
          resolve_depth_window(420) == DEFAULT_DEPTH_WINDOW,
          f"window={resolve_depth_window(420)} of 420 rows")
    check("a depth window never cuts a file down to (almost) nothing",
          resolve_depth_window(16) == (0, 16)
          and resolve_depth_window(420, None) == (0, 420)
          and resolve_depth_window(420, (350, 100)) == (0, 420),
          f"small={resolve_depth_window(16)} open={resolve_depth_window(420, None)} "
          f"reversed={resolve_depth_window(420, (350, 100))}")
    check("a window is clamped to the rows that exist",
          resolve_depth_window(300) == (100, 300)
          and resolve_depth_window(420, (400, 900)) == (400, 420),
          f"short-file={resolve_depth_window(300)} partly-outside="
          f"{resolve_depth_window(420, (400, 900))}")

    with tempfile.TemporaryDirectory() as folder:
        # One page per B-line, shape (lateral, depth); the value stored in a pixel
        # is its depth row, so the rows that come back can be read exactly.
        rows = np.arange(420, dtype=np.float32)[None, :] * np.ones((8, 1), np.float32)
        block = np.repeat(rows[None, :, :], 3, axis=0)
        temp_path = Path(folder) / "Cscan-9-Y1-X1-Z420.tif"
        TIFF.imwrite(str(temp_path), block, photometric="minisblack")
        loaded, loaded_meta = load_scan([temp_path])
        check("loading keeps every depth row and only reports the window",
              loaded.shape == (3, 8, 420)
              and float(loaded[0, 0, 0]) == 0.0
              and float(loaded[0, 0, 100]) == 100.0
              and float(loaded[0, 0, -1]) == 419.0
              and loaded_meta["depth_window"] == [100, 350]
              and loaded_meta["z_pixels_full"] == 420
              and loaded_meta["z_pixels"] == 420,
              f"shape={loaded.shape} window={loaded_meta['depth_window']} "
              f"row0={float(loaded[0, 0, 0]):.0f} row100={float(loaded[0, 0, 100]):.0f}")

    # The window must reach the segmentation as a plain crop of each page and
    # nothing else: same plan as segmenting the already cropped stack.
    rng = np.random.default_rng(7)
    stack = rng.random((24, 10, 40), dtype=np.float32)
    window_plan = detect_segments(stack, 8, depth_window=(12, 30))
    cropped_plan = detect_segments(stack[:, :, 12:30], 8)
    check("the depth window reaches the segmentation as a plain crop",
          window_plan["offsets"] == cropped_plan["offsets"]
          and window_plan["lengths"] == cropped_plan["lengths"]
          and window_plan["turn_pages"] == cropped_plan["turn_pages"],
          f"window={window_plan['turn_pages']}p/{len(window_plan['offsets'])}seg "
          f"crop={cropped_plan['turn_pages']}p/{len(cropped_plan['offsets'])}seg")

    failed = [name for name, ok in results if not ok]
    print("-" * 60)
    print(f"{len(results) - len(failed)}/{len(results)} checks passed")
    for name in failed:
        print(f"  FAILED: {name}")
    print("SELF-TEST RESULT: " + ("PASS" if not failed else "FAIL"))
    return 1 if failed else 0


if QT_AVAILABLE:

    class ReconstructionWindow(QtWidgets.QMainWindow):
        """Interactive rotational C-scan segmentation window (stage 1)."""

        def __init__(self, initial_paths=None, initial_center=None,
                     initial_half_turn_blines=None, initial_depth_window=None):
            super().__init__()
            self.setWindowTitle(
                f"Rotational C-scan Segmentation  |  {BUILD_TAG}  |  "
                f"{SCRIPT_PATH.parent}"
            )
            self.resize(1180, 760)

            self.data = None
            self.meta = {}
            self.volume = None
            self.slot_pages = []
            self.slot_angles = np.zeros(0)
            self.info = {}
            self.segment_plan = None
            self.selected_paths = []
            self.current_source = None
            self.load_worker = None
            self.detect_worker = None
            self.reconstruct_worker = None
            self.applied_xc = None
            self.stage2_panel = None
            # Segment-overlap view: per-segment silhouettes (page x B-line column),
            # the shift each segment is drawn with along the page axis, and the page
            # row whose profiles the lower panel shows.
            self.segment_kymographs = None
            self.segment_kymograph_key = None
            self.segment_shifts = []
            self.row_cursor = 0
            self.pending_center = initial_center
            self.pending_depth_window = (tuple(initial_depth_window)
                                         if initial_depth_window
                                         else DEFAULT_DEPTH_WINDOW)
            # The nominal B-lines per 180 deg is not a control any more: the
            # detector only needs it as the centre of its search window (and
            # widens the search by itself when the peak is weak), so the nominal
            # value - or the one given on the command line - is used and the
            # measured turn is reported instead.
            self.half_turn_blines = float(
                initial_half_turn_blines
                if initial_half_turn_blines is not None
                else DEFAULT_HALF_TURN_BLINES)

            self._build_ui()
            if initial_paths:
                self.set_selected_paths(initial_paths)
                QtCore.QTimer.singleShot(0, self.load_files)

        # ------------------------------------------------------------------ #
        # Layout
        # ------------------------------------------------------------------ #

        def _build_ui(self):
            central = QtWidgets.QWidget(self)
            self.setCentralWidget(central)
            root = QtWidgets.QVBoxLayout(central)

            root.addWidget(self._build_file_group())
            root.addWidget(self._build_result_group())

            # Stage 2 opens from here: the scan that is loaded is handed over, and
            # the panel stacks the turns, calibrates the axis and reconstructs.
            root.addWidget(self._build_stage2_group())

            # Stage 1 output: every segment the detector found. The XY/XZ volume
            # views of stage 2 are commented out further down.
            root.addWidget(self._build_segment_view(), 1)

            self.progress = QtWidgets.QProgressBar(self)
            self.progress.setRange(0, 100)
            self.progress.setValue(0)
            root.addWidget(self.progress)

        def _build_file_group(self):
            group = QtWidgets.QGroupBox("Scan", self)
            layout = QtWidgets.QGridLayout(group)

            self.path_edit = QtWidgets.QLineEdit(self)
            self.path_edit.setReadOnly(True)
            self.path_edit.setPlaceholderText("Select one or more Cscan-*.tif files")
            layout.addWidget(self.path_edit, 0, 0, 1, 3)

            self.browse_button = QtWidgets.QPushButton("Browse...", self)
            self.browse_button.clicked.connect(self.browse)
            layout.addWidget(self.browse_button, 0, 3)

            self.load_button = QtWidgets.QPushButton("Load", self)
            self.load_button.clicked.connect(self.load_files)
            layout.addWidget(self.load_button, 0, 4)

            self.status_label = QtWidgets.QLabel("No scan loaded.", self)
            self.status_label.setWordWrap(True)
            layout.addWidget(self.status_label, 1, 0, 1, 5)

            # Depth window. Everything outside it - in practice the bright sample
            # holder ("base"), which does not turn with the spindle - is left out
            # of the measurements (the segmentation, the centre estimate and the
            # reconstruction). It is still loaded and still shown: only the
            # numbers ignore it.
            depth_row = QtWidgets.QHBoxLayout()
            depth_row.addWidget(QtWidgets.QLabel("Depth rows", self))
            self.depth_start_spin = QtWidgets.QSpinBox(self)
            self.depth_start_spin.setRange(0, 4096)
            self.depth_start_spin.setValue(int(self.pending_depth_window[0]))
            self.depth_start_spin.setToolTip(
                "First depth row that is measured. Rows above it are still loaded "
                "and shown, but left out of the segmentation."
            )
            depth_row.addWidget(self.depth_start_spin)
            depth_row.addWidget(QtWidgets.QLabel("to", self))
            self.depth_stop_spin = QtWidgets.QSpinBox(self)
            self.depth_stop_spin.setRange(1, 4096)
            self.depth_stop_spin.setValue(int(self.pending_depth_window[1]))
            self.depth_stop_spin.setToolTip(
                "First depth row that is NOT measured (half-open range). Rows below "
                "it are still loaded and shown, but left out of the segmentation - "
                "this is what keeps the sample holder out of the numbers."
            )
            depth_row.addWidget(self.depth_stop_spin)
            self.depth_hint = QtWidgets.QLabel(
                f"measured only, never cropped away - every row is loaded and shown "
                f"(default {DEFAULT_DEPTH_WINDOW[0]}-{DEFAULT_DEPTH_WINDOW[1]})", self
            )
            depth_row.addWidget(self.depth_hint)
            depth_row.addStretch(1)
            layout.addLayout(depth_row, 2, 0, 1, 5)
            self.depth_start_spin.valueChanged.connect(self._on_depth_window_edited)
            self.depth_stop_spin.valueChanged.connect(self._on_depth_window_edited)
            return group

        def _build_stage2_group(self):
            """The door to stage 2: the 3D reconstruction panel."""
            group = QtWidgets.QGroupBox("3D reconstruction (stage 2)", self)
            layout = QtWidgets.QHBoxLayout(group)
            self.stage2_button = QtWidgets.QPushButton("Open the volume panel...",
                                                       self)
            self.stage2_button.setToolTip(
                "Hand the loaded scan to the 3D panel: it aligns and averages the "
                "turns, calibrates the rotation axis (which does not have to lie in "
                "the B-line) and reconstructs a volume by splatting the pages.")
            self.stage2_button.clicked.connect(self.open_stage2_panel)
            self.stage2_button.setEnabled(False)
            layout.addWidget(self.stage2_button)
            hint = QtWidgets.QLabel(
                "uses every full turn, averages them into one, and reconstructs "
                "with the tilted-axis geometry - load a scan first", self)
            hint.setWordWrap(True)
            layout.addWidget(hint, 1)
            return group

        def open_stage2_panel(self):
            """Open (or refresh) the stage 2 panel with the loaded stack."""
            if self.data is None:
                QtWidgets.QMessageBox.information(
                    self, "3D reconstruction",
                    "Load a C-scan first - the panel works on the stack that is "
                    "already loaded.")
                return None
            try:
                from RotationalVolumePanel import VolumeWindow
            except Exception as error:  # noqa: BLE001 - reported to the user
                QtWidgets.QMessageBox.critical(
                    self, "3D reconstruction",
                    f"The stage 2 panel could not be imported:\n{error}")
                return None
            plan = self.segment_plan if isinstance(self.segment_plan, dict) else {}
            # The depth rows the segmentation actually used: the panel reports them
            # and reconstructs every row of the file regardless.
            start, stop = self._depth_window()
            self.meta["depth_window"] = [int(start), int(stop)]
            if self.stage2_panel is None:
                self.stage2_panel = VolumeWindow(self.data, self.meta, plan)
            else:
                self.stage2_panel.adopt_scan(self.data, self.meta, plan)
            self.stage2_panel.set_depth_window(start, stop)
            self.stage2_panel.show()
            self.stage2_panel.raise_()
            self.stage2_panel.activateWindow()
            return self.stage2_panel

        def _build_result_group(self):
            """What stage 1 found, in one line: the turn length and the segments.
            There is no parameter to set here. The segmentation runs by itself when
            a scan is read, and the only input it needs - the nominal number of
            B-lines per 180 deg, ``self.half_turn_blines`` - is the centre of the
            detector's search window rather than something to tune. The detailed
            report goes to stdout.
            """
            group = QtWidgets.QGroupBox("Segmentation result", self)
            layout = QtWidgets.QGridLayout(group)

            # STAGE 3: the rotation centre is not known yet. It must not even be
            # inside the B-line, and the segmentation below does not use it, so the
            # centre controls are commented out with the rest of stage 2.
            # layout.addWidget(QtWidgets.QLabel("Centre pixel Xc", self), 1, 0)
            # self.center_slider = QtWidgets.QSlider(Qt.Horizontal, self)
            # self.center_slider.setRange(0, 0)
            # layout.addWidget(self.center_slider, 1, 1)
            # self.center_spin = QtWidgets.QDoubleSpinBox(self)
            # self.center_spin.setDecimals(1)
            # self.center_spin.setSingleStep(0.1)
            # self.center_spin.setRange(0.0, 0.0)
            # layout.addWidget(self.center_spin, 1, 2)
            # self.center_spin.valueChanged.connect(
            #     lambda value: self.center_slider.setValue(int(round(value * 10.0)))
            # )
            # self.center_slider.valueChanged.connect(
            #     lambda value: self.center_spin.setValue(value / 10.0)
            # )

            # Segmentation is automatic: a scan is segmented as soon as it has
            # been read, so there is nothing to press. The detector reports the
            # turn length, the implied B-lines per 180 deg and every segment.

            # ------------------------------------------------------------------ #
            # STAGE 2 - the reconstruction controls stay commented out while only
            # the segmentation is developed. They all need the rotation centre,
            # which is a later step; stage 1 never builds a volume.
            # ------------------------------------------------------------------ #
            # options = QtWidgets.QHBoxLayout()
            # self.log_check = QtWidgets.QCheckBox("Log / dB display", self)
            # options.addWidget(self.log_check)
            # self.median_check = QtWidgets.QCheckBox("Median filter", self)
            # options.addWidget(self.median_check)
            # self.log_check.stateChanged.connect(self.refresh_views)
            # self.median_check.stateChanged.connect(self.refresh_views)
            # options.addWidget(QtWidgets.QLabel("Decimate", self))
            # self.decimate_combo = QtWidgets.QComboBox(self)
            # for factor in (1, 2, 4):
            #     self.decimate_combo.addItem(str(factor), factor)
            # options.addWidget(self.decimate_combo)
            # options.addStretch(1)
            #
            # self.apply_button = QtWidgets.QPushButton("Apply", self)
            # self.apply_button.setEnabled(False)
            # self.apply_button.clicked.connect(self.apply_parameters)
            # options.addWidget(self.apply_button)
            #
            # self.export_button = QtWidgets.QPushButton("Export volume...", self)
            # self.export_button.setEnabled(False)
            # self.export_button.clicked.connect(self.export_volume_dialog)
            # options.addWidget(self.export_button)
            # layout.addLayout(options, 3, 0, 1, 3)
            #
            # contrast = QtWidgets.QHBoxLayout()
            # contrast.addWidget(QtWidgets.QLabel("Contrast", self))
            # self.vmin_spin = QtWidgets.QDoubleSpinBox(self)
            # self.vmin_spin.setDecimals(4)
            # self.vmin_spin.setRange(-1e9, 1e9)
            # contrast.addWidget(self.vmin_spin)
            # self.vmax_spin = QtWidgets.QDoubleSpinBox(self)
            # self.vmax_spin.setDecimals(4)
            # self.vmax_spin.setRange(-1e9, 1e9)
            # self.vmax_spin.setValue(1.0)
            # contrast.addWidget(self.vmax_spin)
            # self.auto_button = QtWidgets.QPushButton("Auto", self)
            # self.auto_button.clicked.connect(self.auto_contrast)
            # contrast.addWidget(self.auto_button)
            # contrast.addStretch(1)
            # layout.addLayout(contrast, 4, 0, 1, 3)
            # self.vmin_spin.valueChanged.connect(self.refresh_views)
            # self.vmax_spin.valueChanged.connect(self.refresh_views)

            self.info_label = QtWidgets.QLabel(
                "Load a scan; it is segmented as soon as it has been read.", self
            )
            self.info_label.setWordWrap(True)
            layout.addWidget(self.info_label, 0, 0, 1, 3)

            # The full segmentation report is not shown here any more: it is
            # printed to stdout, and the window keeps the single line above.
            return group

        def _make_canvas(self):
            try:
                from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
            except ImportError:
                from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
            from matplotlib.figure import Figure

            figure = Figure(facecolor="black")
            canvas = FigureCanvasQTAgg(figure)
            canvas.setMinimumSize(320, 320)
            axes = figure.add_subplot(111)
            axes.set_facecolor("black")
            return figure, canvas, axes

        # ------------------------------------------------------------------ #
        # Segment view: the stage-1 output
        # ------------------------------------------------------------------ #

        def _build_segment_view(self):
            """The sum of the detected turns, at one angle and over the whole turn.

            The full turns should be exact repeats of each other: summing them puts
            the same angle of every turn on top of each other, so the result is
            sharp only while the segmentation is right. Lining them up is done
            automatically (``auto_align_all_segments``) as soon as the segmentation
            is known, so the only control left is the page row - the angle the upper
            panel shows.
            """
            try:
                from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
            except ImportError:
                from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
            from matplotlib.figure import Figure

            group = QtWidgets.QGroupBox("Segment overlap (stage 1 check)", self)
            layout = QtWidgets.QVBoxLayout(group)

            self.segment_figure = Figure(facecolor="white", layout="constrained")
            self.segment_canvas = FigureCanvasQTAgg(self.segment_figure)
            self.segment_canvas.setMinimumSize(520, 340)
            grid = self.segment_figure.add_gridspec(2, 1, height_ratios=[3, 2])
            # Top: the pages of the summed segments at one angle, added directly -
            # the speckle only adds up while they really are the same view.
            self.overlap_page_axes = self.segment_figure.add_subplot(grid[0])
            # Bottom: the same sum over the whole turn, page row on Y (drag here).
            self.overlap_turn_axes = self.segment_figure.add_subplot(
                grid[1], sharex=self.overlap_page_axes
            )
            self.segment_canvas.mpl_connect("motion_notify_event", self._on_segment_motion)
            layout.addWidget(self.segment_canvas, 1)

            # One control only: the page row, the angle the upper panel shows. The
            # turns are lined up automatically, so there is nothing to slide by
            # hand and no text to read - the numbers go to stdout.
            row = QtWidgets.QHBoxLayout()
            row.addWidget(QtWidgets.QLabel("page row", self))
            self.row_slider = QtWidgets.QSlider(Qt.Horizontal, self)
            self.row_slider.setRange(0, 0)
            self.row_slider.setToolTip(
                "Which page of the turn the upper panel shows. Dragging on the "
                "image moves it as well."
            )
            self.row_slider.valueChanged.connect(self._on_row_changed)
            row.addWidget(self.row_slider, 1)
            layout.addLayout(row)
            return group

        def draw_segment_view(self, plan, data):
            """Sum the detected turns on top of each other (the stage-1 check)."""
            page_axes = self.overlap_page_axes
            turn_axes = self.overlap_turn_axes
            page_axes.clear()
            turn_axes.clear()
            if data is None or not plan:
                page_axes.set_title("no segmentation yet", fontsize=9)
                turn_axes.set_title("load a scan: it is segmented at once", fontsize=8)
                self.segment_canvas.draw_idle()
                return

            summary = self._summed_image(plan, data)
            turn = int(plan.get("turn_pages", 0))
            if summary is None:
                page_axes.set_title("no segment long enough to sum", fontsize=9)
                self.segment_canvas.draw_idle()
                return

            self._sync_row_range(summary["low"], summary["high"])
            row = int(self.row_cursor)
            # The pages are added as raw amplitudes and shown in dB. That is what
            # the "Log / dB" box used to switch; the box went with the other
            # controls and the display stays as it was.
            log_display = True

            image = summary["image"]
            shown = image
            vmin = vmax = None
            if log_display:
                ceiling = float(np.percentile(image, 99.9))
                shown = np.clip(20.0 * np.log10(np.clip(image, 1e-9, None)
                                                / max(1e-9, ceiling)), -40.0, 0.0)
                vmin, vmax = -40.0, 0.0
            turn_axes.imshow(shown, cmap="gray", aspect="auto", origin="lower",
                             extent=(0, int(data.shape[1]), summary["low"],
                                     summary["high"]),
                             vmin=vmin, vmax=vmax, interpolation="nearest")
            turn_axes.axhline(row, color="cyan", lw=0.9)
            names = ", ".join(f"#{index + 1}" for index in summary["included"])
            shifts = " / ".join(str(int(summary["shifts"][index]))
                                for index in summary["included"])
            turn_axes.set_title(
                f"sum of segment(s) {names} over the whole turn  |  Y shift {shifts} "
                f"pages  |  turn {turn} pages = {turn / 2.0:.1f} B-lines per 180 deg",
                fontsize=8,
            )
            turn_axes.set_ylabel("page in the turn (Y)", fontsize=7)
            turn_axes.set_xlabel("B-line column (X)", fontsize=7)
            turn_axes.tick_params(labelsize=6)

            self._draw_summed_page(page_axes, plan, data, summary, row, log_display)
            self.segment_canvas.draw_idle()

        def redraw_segment_view(self):
            """Redraw from the last plan (new data and the row slider land here)."""
            self.draw_segment_view(self.segment_plan, self.data)

        # ------------------------------------------------------------------ #
        # Segment overlap: preparing the data
        # ------------------------------------------------------------------ #

        def _plan_key(self, plan):
            """Identity of a plan, so the silhouettes survive an ordinary redraw.

            The depth rows are part of the key: changing them changes what the
            silhouettes have to be taken from, so the cache must not survive it.
            """
            if not plan:
                return None
            return (tuple(plan.get("offsets", [])), tuple(plan.get("lengths", [])),
                    int(plan.get("turn_pages", 0)), int(plan.get("pages_total", 0)),
                    self._segmentation_rows())

        def _segment_kymographs(self, plan, data):
            """Lateral silhouette (max over depth) of every segment.

            Shape ``(pages of that segment, B-line columns)``: row *n* is page
            ``offset + n`` reduced to one profile, which is what the summed image and
            the lower panel compare against each other. The maximum is taken over the
            depth rows the segmentation uses, so the (static) sample holder cannot
            flatten every profile to the same shape.
            """
            key = self._plan_key(plan)
            if self.segment_kymographs is not None and self.segment_kymograph_key == key:
                return self.segment_kymographs
            offsets = [int(value) for value in plan.get("offsets", [])]
            lengths = [int(value) for value in plan.get("lengths", [])]
            kymographs = []
            for index, offset in enumerate(offsets):
                length = lengths[index] if index < len(lengths) else 0
                start = max(0, min(int(offset), int(data.shape[0])))
                stop = max(start, min(int(data.shape[0]), int(offset) + max(0, length)))
                if stop <= start:
                    kymographs.append(np.zeros((0, int(data.shape[1])), dtype=np.float32))
                    continue
                pages = np.asarray(data[start:stop], dtype=np.float32)
                rows = self._segmentation_rows()
                if rows is not None:
                    pages = pages[:, :, int(rows[0]):int(rows[1])]
                kymographs.append(pages.max(axis=2))
            self.segment_kymographs = kymographs
            self.segment_kymograph_key = key
            return kymographs

        def _included_segments(self, plan):
            """Indices of the segments that are summed on top of each other."""
            count = len(plan.get("offsets", []))
            is_full = [bool(value) for value in plan.get("is_full", [])]
            included = [index for index in range(count)
                        if index < len(is_full) and is_full[index]]
            if len(included) < 2:
                # Only partial segments (or a single one): sum whatever there is.
                included = list(range(count))
            return included

        def _ensure_segment_shifts(self, plan):
            """One Y shift per segment, in pages; created once per plan."""
            count = max(1, len(plan.get("offsets", [])))
            if len(self.segment_shifts) != count:
                self.segment_shifts = [0] * count

        def _summed_image(self, plan, data):
            """Sum of the included segments with the current Y shifts."""
            kymographs = self._segment_kymographs(plan, data)
            self._ensure_segment_shifts(plan)
            shifts = list(self.segment_shifts)
            included = self._included_segments(plan)
            usable = [index for index in included
                      if kymographs[index].size
                      and kymographs[index].shape[1] == int(data.shape[1])
                      and kymographs[index].shape[0] + shifts[index] > 0]
            if not usable:
                return None
            low = max(0, max(-shifts[index] for index in usable))
            high = min(kymographs[index].shape[0] - shifts[index] for index in usable)
            if high - low < 8:
                return None
            total = None
            for index in usable:
                block = kymographs[index][low + shifts[index]:high + shifts[index]]
                total = (block.astype(np.float32, copy=True) if total is None
                         else total + block)
            return {"image": total, "low": int(low), "high": int(high),
                    "included": usable, "kymographs": kymographs, "shifts": shifts}

        def _draw_summed_page(self, axis, plan, data, summary, row, log_display):
            """One B-scan per turn at the current page row, added on top of each other.

            This is the direct answer to "are the turns really repeats": the same
            view adds up coherently (and the speckle stays contrasty) only while the
            pages of the segments really show the same angle. The whole B-line is
            drawn - the sample holder belongs to the picture - and dashed lines mark
            the depth rows the measurements were taken from.
            """
            offsets = [int(value) for value in plan.get("offsets", [])]
            total = None
            used = []
            for index in summary["included"]:
                page_index = ((offsets[index] if index < len(offsets) else 0)
                              + row + int(summary["shifts"][index]))
                if not (0 <= page_index < int(data.shape[0])):
                    continue
                block = np.asarray(data[page_index], dtype=np.float32)
                total = block.copy() if total is None else total + block
                used.append(page_index)
            if total is None:
                axis.set_title("no page at this row", fontsize=8)
                return
            shown = total
            vmin = vmax = None
            if log_display:
                ceiling = float(np.percentile(total, 99.9))
                shown = np.clip(20.0 * np.log10(np.clip(total, 1e-9, None)
                                                / max(1e-9, ceiling)), -40.0, 0.0)
                vmin, vmax = -40.0, 0.0
            axis.imshow(np.transpose(shown), cmap="gray", aspect="auto", origin="upper",
                        vmin=vmin, vmax=vmax, interpolation="nearest")
            # The whole B-line is drawn; the dashed lines are the depth rows the
            # numbers were taken from (the sample, not the holder).
            depth_rows = self._segmentation_rows()
            if depth_rows is None:
                rows_text = "all depth rows measured"
            else:
                rows_text = (f"rows {depth_rows[0]}-{depth_rows[1] - 1} measured "
                             f"(dashed)")
                for edge in depth_rows:
                    axis.axhline(int(edge) - 0.5, color="#ff8c00", lw=0.7,
                                 linestyle="--")
            axis.set_title(
                f"page {row} of every turn, added: B-lines "
                f"{', '.join(str(value) for value in used)} - all "
                f"{int(total.shape[1])} depth rows shown, {rows_text}", fontsize=7
            )
            axis.set_ylabel("depth", fontsize=7)
            axis.tick_params(labelsize=6)

        def _sync_row_range(self, low, high):
            """Keep the page-row slider inside the rows that the sum covers."""
            last = max(int(low), int(high) - 1)
            self.row_slider.blockSignals(True)
            self.row_slider.setRange(int(low), last)
            self.row_slider.setValue(max(int(low), min(int(self.row_cursor), last)))
            self.row_slider.blockSignals(False)
            self.row_cursor = int(self.row_slider.value())

        def _overlap_text(self, plan, data, summary, row):
            """How the summed segments are shifted, and how well their pages match."""
            offsets = [int(value) for value in plan.get("offsets", [])]
            turn = int(plan.get("turn_pages", 0))
            parts = [f"rows {summary['low']}-{summary['high'] - 1} of the turn summed "
                     f"({summary['high'] - summary['low']} pages); page row {row} is "
                     f"the image above"]
            reference = summary["included"][0] if summary["included"] else 0
            for index in summary["included"]:
                if index == reference:
                    continue
                shift = int(summary["shifts"][index])
                aligned = (offsets[index] if index < len(offsets) else 0) + shift
                score = self._page_match_score(
                    plan, data, reference, int(summary["shifts"][reference]),
                    index, shift, sample=12, x_step=2,
                )
                text = (f"#{index + 1}: starts {aligned} pages after #{reference + 1}"
                        f", page match {score:.2f}")
                if turn:
                    residual = ((aligned + turn // 2) % turn) - turn // 2
                    text += f" ({residual:+d} pages off a {turn}-page turn)"
                parts.append(text)
            parts.append("the Y shifts were found by maximising this page match")
            return "  |  ".join(parts)

        def _page_match_score(self, plan, data, reference, reference_shift,
                              index, index_shift, sample=8, x_step=4, z_step=2):
            """NCC of the raw pages of two segments at the given page shifts.

            Pages are sampled along the turn, which keeps this cheap while it uses the
            speckle (and not only the envelope), so it reacts to a shift of a single
            page - unlike a silhouette correlation, which is far too flat. Only the
            depth rows the segmentation uses are compared: the sample holder is the
            same in every turn, so it would only add the same offset to both sides.
            """
            offsets = [int(value) for value in plan.get("offsets", [])]
            lengths = [int(value) for value in plan.get("lengths", [])]
            depth_rows = self._segmentation_rows()
            if depth_rows is None:
                z_span = slice(0, int(data.shape[2]), int(z_step))
            else:
                z_span = slice(int(depth_rows[0]), int(depth_rows[1]), int(z_step))

            def limits(segment, shift):
                length = lengths[segment] if segment < len(lengths) else 0
                low = max(0, -shift, -offsets[segment])
                high = min(length - shift,
                           int(data.shape[0]) - offsets[segment] - shift)
                return low, high

            low = max(limits(reference, reference_shift)[0], limits(index, index_shift)[0])
            high = min(limits(reference, reference_shift)[1], limits(index, index_shift)[1])
            if high - low < 8:
                return float("nan")
            step = max(1, (high - low) // max(1, int(sample)))
            rows = np.arange(low, high, step)[:int(sample)]
            first = np.asarray(data[offsets[reference] + rows + reference_shift,
                                    ::x_step, z_span], dtype=np.float64)
            second = np.asarray(data[offsets[index] + rows + index_shift,
                                     ::x_step, z_span], dtype=np.float64)
            left = first.reshape(-1) - first.mean()
            right = second.reshape(-1) - second.mean()
            denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
            return float(left @ right / denominator) if denominator else float("nan")

        def auto_align_segment(self, selected, span=150):
            """Best Y shift of one segment against another one, in pages.

            The score is the speckle NCC of sampled pages (``_page_match_score``),
            searched coarsely over the whole span and then refined around the peak.
            Returns ``(shift, score)``; the score is nan when nothing could be
            scored. No widget is touched: the caller decides what to do with it.
            """
            plan, data = self.segment_plan, self.data
            if not plan or data is None:
                return 0, float("nan")
            kymographs = self._segment_kymographs(plan, data)
            included = self._included_segments(plan)
            if selected not in included:
                return 0, float("nan")
            reference = next((index for index in included
                              if index != selected and kymographs[index].size), None)
            if reference is None:
                return 0, float("nan")
            self._ensure_segment_shifts(plan)
            reference_shift = int(self.segment_shifts[reference])
            best_shift, best_score = 0, float("nan")
            for shift in range(-span, span + 1):            # coarse, then fine
                score = self._page_match_score(plan, data, reference, reference_shift,
                                               selected, shift, sample=6, x_step=8,
                                               z_step=4)
                if math.isnan(score):
                    continue
                if math.isnan(best_score) or score > best_score:
                    best_shift, best_score = shift, score
            for shift in range(best_shift - 4, best_shift + 5):
                if abs(shift) > span:
                    continue
                score = self._page_match_score(plan, data, reference, reference_shift,
                                               selected, shift, sample=12, x_step=2)
                if math.isnan(score):
                    continue
                if math.isnan(best_score) or score > best_score:
                    best_shift, best_score = shift, score
            return int(best_shift), float(best_score)

        def auto_align_all_segments(self, span=150):
            """Line every full segment up with the first one, and report the shifts.

            The turns can only be compared once they lie on top of each other, so
            this runs as soon as the segmentation is known - there is no button for
            it in the window. The page match it maximises is the same number that
            used to be shown below the image; it goes to stdout instead now, and
            the Y shifts it found stay in the figure title.
            """
            plan, data = self.segment_plan, self.data
            if not plan or data is None:
                return
            included = self._included_segments(plan)
            self._ensure_segment_shifts(plan)
            self.segment_shifts = [0] * len(self.segment_shifts)
            if len(included) < 2:
                return
            reference = included[0]
            for index in included[1:]:
                shift, score = self.auto_align_segment(index, span=span)
                self.segment_shifts[index] = int(shift)
                score_text = "nan" if math.isnan(score) else f"{score:.3f}"
                print(f"[RotationalReconstruct] Y shift: segment #{index + 1} best "
                      f"matches #{reference + 1} at {int(shift):+d} pages "
                      f"(page match {score_text}, searched +/-{int(span)})")

        def _on_row_changed(self, value):
            """The row cursor moved: redraw the summed page and the cursor only."""
            self.row_cursor = int(value)
            plan, data = self.segment_plan, self.data
            if plan is None or data is None or self.segment_kymographs is None:
                return
            summary = self._summed_image(plan, data)
            if summary is None:
                return
            page_axes = self.overlap_page_axes
            page_axes.clear()
            self._draw_summed_page(page_axes, plan, data, summary, int(value), True)
            for line in list(self.overlap_turn_axes.lines):
                line.remove()
            self.overlap_turn_axes.axhline(int(value), color="cyan", lw=0.9)
            self.segment_canvas.draw_idle()

        def _on_segment_motion(self, event):
            """Dragging on the turn map moves the page row (the Y axis)."""
            if event.button != 1 or event.inaxes is not self.overlap_turn_axes:
                return
            if event.ydata is None:
                return
            self.row_slider.setValue(int(round(float(event.ydata))))

        # ===== STAGE 2 - commented out while only the segmentation is developed =====
#         def _build_xy_view(self):
#             layout = QtWidgets.QVBoxLayout()
#             title = QtWidgets.QLabel("XY view (depth slice, origin lower-left)", self)
#             layout.addWidget(title)
#
#             self.xy_figure, self.xy_canvas, self.xy_axes = self._make_canvas()
#             layout.addWidget(self.xy_canvas, 1)
#
#             row = QtWidgets.QHBoxLayout()
#             row.addWidget(QtWidgets.QLabel("Depth", self))
#             self.depth_slider = QtWidgets.QSlider(Qt.Horizontal, self)
#             self.depth_slider.setRange(0, 0)
#             self.depth_slider.valueChanged.connect(self.update_xy_view)
#             row.addWidget(self.depth_slider, 1)
#             self.depth_label = QtWidgets.QLabel("-", self)
#             row.addWidget(self.depth_label)
#             layout.addLayout(row)
#             return layout
#
#         def _build_xz_view(self):
#             layout = QtWidgets.QVBoxLayout()
#             title = QtWidgets.QLabel("XZ view (averaged B-line at angle)", self)
#             layout.addWidget(title)
#
#             self.xz_figure, self.xz_canvas, self.xz_axes = self._make_canvas()
#             layout.addWidget(self.xz_canvas, 1)
#
#             row = QtWidgets.QHBoxLayout()
#             row.addWidget(QtWidgets.QLabel("Angle", self))
#             self.angle_slider = QtWidgets.QSlider(Qt.Horizontal, self)
#             self.angle_slider.setRange(0, 0)
#             self.angle_slider.valueChanged.connect(self.update_xz_view)
#             row.addWidget(self.angle_slider, 1)
#             self.angle_label = QtWidgets.QLabel("-", self)
#             row.addWidget(self.angle_label)
#             layout.addLayout(row)
#             return layout
#
        # ------------------------------------------------------------------ #
        # File handling
        # ------------------------------------------------------------------ #

        def set_selected_paths(self, paths):
            self.selected_paths = [str(path) for path in paths]
            if self.selected_paths:
                self.path_edit.setText("; ".join(Path(p).name for p in self.selected_paths))
                if len(self.selected_paths) == 1:
                    self.current_source = self.selected_paths[0]

        def browse(self):
            paths, _filter = QtWidgets.QFileDialog.getOpenFileNames(
                self, "Select C-scan TIFF file(s)", "", "TIFF files (*.tif *.tiff);;All files (*)"
            )
            if paths:
                self.set_selected_paths(paths)
                self.load_files()

        def load_files(self):
            if not self.selected_paths:
                self.status_label.setText("Select scan file(s) first.")
                return
            self._set_busy(True, "Loading...", busy_apply=True)
            worker = LoadWorker(self.selected_paths, self._depth_window(), self)
            worker.progressed.connect(self._on_progress)
            worker.loaded.connect(self._on_loaded)
            worker.failed.connect(self._on_worker_failed)
            self.load_worker = worker
            worker.start()

        def _on_loaded(self, data, meta):
            self.data = data
            self.meta = meta
            self.volume = None
            self.slot_pages = []
            self.slot_angles = np.zeros(0)
            self.info = {}
            if hasattr(self, "stage2_button"):
                # The 3D panel works on the stack that is already in memory.
                self.stage2_button.setEnabled(True)

            x_pixels = int(meta["x_pixels"])
            z_pixels = int(meta["z_pixels"])
            # STAGE 3: the centre controls are commented out together with stage 2.
            # The rotation axis is not known yet - and it does not have to lie
            # inside the B-line - while the segmentation needs no centre at all.
            # self.center_spin.setRange(0.0, float(x_pixels - 1))
            # self.center_slider.setRange(0, (x_pixels - 1) * 10)
            # self.center_spin.setValue((x_pixels - 1) / 2.0)
            # if self.pending_center is not None:
            #     self.center_spin.setValue(float(self.pending_center))
            #     self.pending_center = None
            # The number of B-lines per 180 deg cannot be read from the file; the
            # nominal value (``self.half_turn_blines``) is the centre of the
            # detector's search window, and the detector widens the search by
            # itself when the peak is weak, so a file holding several turns still
            # yields the true period.

            # The depth slider of the XY view belongs to stage 2.
            # self.depth_slider.setRange(0, max(0, z_pixels - 1))
            # self.depth_slider.setValue(0)

            # Show the depth window the measurements will use: it may have been
            # clamped to the rows the file really has. Nothing was cropped while
            # reading - the spins only say which rows the numbers may look at.
            window = meta.get("depth_window") or [0, z_pixels]
            full_z = int(meta.get("z_pixels_full", z_pixels))
            for spin in (self.depth_start_spin, self.depth_stop_spin):
                spin.blockSignals(True)
            self.depth_start_spin.setRange(0, max(0, full_z - 1))
            self.depth_stop_spin.setRange(1, max(1, full_z))
            self.depth_start_spin.setValue(int(window[0]))
            self.depth_stop_spin.setValue(int(window[1]))
            for spin in (self.depth_start_spin, self.depth_stop_spin):
                spin.blockSignals(False)
            self.depth_hint.setText(
                f"measured rows {int(window[0])}-{int(window[1]) - 1} of {full_z}"
                f" - all {z_pixels} rows are loaded and shown"
            )

            self.status_label.setText(
                f"{Path(meta['files'][0]).name if len(meta['files']) == 1 else str(len(meta['files'])) + ' files'}"
                f"  |  {meta['pages']} B-lines  |  X={x_pixels}  Z={z_pixels} "
                f"(all rows loaded)"
                f"  |  dx={meta['x_step_um']:.1f} um/px"
                f"  |  segmentation rows {int(window[0])}-{int(window[1]) - 1}"
            )
            self.info_label.setText("Loaded. Segmenting the recording...")
            self.segment_plan = None
            # New data: the per-segment silhouettes of the previous file are stale.
            self.segment_kymographs = None
            self.segment_kymograph_key = None
            self.segment_shifts = []
            self.row_cursor = 0
            self._set_busy(False, "Loaded.", busy_apply=False)
            self.redraw_segment_view()
            # A scan is segmented as soon as it has been read: that is the whole of
            # stage 1, and there is no button to press for it.
            self.detect_turns_now()
            # self.apply_button.setEnabled(True)

        def _on_progress(self, value, total, message):
            percent = 0 if total <= 0 else int(100 * value / total)
            self.progress.setValue(max(0, min(100, percent)))
            if message:
                self.info_label.setText(message)

        def _on_worker_failed(self, message):
            self._set_busy(False, "Failed.", busy_apply=False)
            if message == "cancelled":
                self.info_label.setText("Cancelled.")
                return
            self.info_label.setText(f"Error: {message}")
            QtWidgets.QMessageBox.critical(self, "Reconstruction error", str(message))

        def _set_busy(self, busy, message, busy_apply=True):
            """Enable/disable the controls that stage 1 owns."""
            # ``busy_apply`` belongs to the stage-2 call sites that are commented
            # out; stage 1 has no Apply/Export button of its own.
            _ = busy_apply
            self.load_button.setEnabled(not busy)
            self.browse_button.setEnabled(not busy)
            if message:
                self.status_label.setText(message)
            if not busy:
                self.progress.setValue(0)

        # ------------------------------------------------------------------ #
        # Depth window
        # ------------------------------------------------------------------ #

        def _depth_window(self):
            """Depth window in the spin boxes, as a ``(start, stop)`` pixel pair."""
            start = int(self.depth_start_spin.value())
            stop = int(self.depth_stop_spin.value())
            if stop <= start:
                stop = start + 1
            return start, stop

        def _segmentation_rows(self):
            """Depth rows the measurements may use, or ``None`` for every row.

            The stack itself always holds every depth row (that is what the views
            show); this is the half-open row range that the segmentation
            signatures, the silhouettes and the page match are taken from, so the
            (non-rotating) sample holder stays out of the numbers.
            """
            start, stop = self._depth_window()
            if self.data is None:
                return (start, stop)
            z_pixels = int(self.data.shape[2])
            start = max(0, min(start, z_pixels - 1))
            stop = max(start + 1, min(stop, z_pixels))
            if start <= 0 and stop >= z_pixels:
                return None                 # every row: nothing to leave out
            return (start, stop)

        def _on_depth_window_edited(self, _value):
            """The window only steers the measurements, so segment again at once."""
            if self.data is None:
                return
            rows = self._segmentation_rows() or (0, int(self.data.shape[2]))
            # Anything that reads the window downstream (the stage 2 panel, the JSON
            # sidecar) has to see the rows the measurements really used.
            self.meta["depth_window"] = [int(rows[0]), int(rows[1])]
            self.depth_hint.setText(
                f"measured rows {rows[0]}-{rows[1] - 1} of {int(self.data.shape[2])}"
                f" - all {int(self.data.shape[2])} rows are loaded and shown"
            )
            if self.detect_worker is not None and self.detect_worker.isRunning():
                self.info_label.setText(
                    "Depth rows changed - the running search finishes first; change "
                    "them again afterwards to segment from the new rows."
                )
                return
            self.info_label.setText("Depth rows changed - segmenting from the new rows.")
            self.detect_turns_now()

        # ------------------------------------------------------------------ #
        # Reconstruction
        # ------------------------------------------------------------------ #

        # ===== STAGE 2 - commented out while only the segmentation is developed =====
#         def apply_parameters(self):
#             if self.data is None:
#                 return
#             if self.reconstruct_worker is not None and self.reconstruct_worker.isRunning():
#                 self.reconstruct_worker.cancel()
#                 self.reconstruct_worker.wait(2000)
#
#             self._set_busy(True, "Reconstructing...", busy_apply=True)
#             worker = ReconstructWorker(
#                 self.data,
#                 float(self.center_spin.value()),
#                 float(self.rotation_spin.value()),
#                 int(self.decimate_combo.currentData()),
#                 detect=bool(self.segment_check.isChecked()),
#                 segment_index=max(0, int(self.segment_spin.value()) - 1),
#                 parent=self,
#             )
#             worker.progressed.connect(self._on_progress)
#             worker.detected.connect(self._on_plan_ready)
#             worker.done.connect(self._on_reconstructed)
#             worker.failed.connect(self._on_worker_failed)
#             self.reconstruct_worker = worker
#             worker.start()
#
        def detect_turns_now(self):
            """Search for the repeated turns of the loaded scan (stage 1)."""
            if self.data is None:
                return
            if self.detect_worker is not None and self.detect_worker.isRunning():
                return
            self._set_busy(True, "Looking for the rotation repeats...", busy_apply=True)
            worker = DetectWorker(
                self.data, float(self.half_turn_blines),
                depth_window=self._segmentation_rows(), parent=self,
            )
            worker.progressed.connect(self._on_progress)
            worker.detected.connect(self._on_plan_ready)
            worker.failed.connect(self._on_worker_failed)
            worker.finished.connect(
                lambda: self._set_busy(False, None, busy_apply=False)
            )
            self.detect_worker = worker
            worker.start()

        def _on_plan_ready(self, plan):
            """Show the detected turns as soon as they are known, not at the end."""
            self.segment_plan = plan
            self._update_segment_report({"segment_plan": plan, "segment_index": 0})
            # The turns can only be summed meaningfully once they lie on top of
            # each other, so they are lined up here, before anything is drawn.
            self.auto_align_all_segments()
            self.redraw_segment_view()
            summary = (self._summed_image(plan, self.data)
                       if plan and self.data is not None else None)
            if summary is not None:
                print("[RotationalReconstruct] "
                      + self._overlap_text(plan, self.data, summary,
                                           int(self.row_cursor)))
            turn = int(plan.get("turn_pages", 0)) if plan else 0
            if turn:
                nominal = " (nominal fallback, not measured)" if plan.get("nominal") else ""
                count = len(plan.get("offsets", []))
                self.info_label.setText(
                    f"Detected {count} segment(s): {turn} pages per turn = "
                    f"{turn / 2.0:.1f} B-lines per 180 deg{nominal}. The turns are "
                    f"added in the overlap view below."
                )

        # "Use detected N" is gone with the parameter field it wrote into: the
        # detector reports the measured turn itself, so there is nothing to set.

        # ===== STAGE 2 - commented out while only the segmentation is developed =====
#         def _on_reconstructed(self, volume, slot_pages, slot_angles, info):
#             self.volume = volume
#             self.slot_pages = list(slot_pages)
#             self.slot_angles = np.asarray(slot_angles)
#             self.info = dict(info)
#             self.applied_xc = float(self.center_spin.value())
#             self._update_segment_report(info)
#
#             last_slot = max(0, len(self.slot_pages) - 1)
#             self.angle_slider.setRange(0, last_slot)
#             self.angle_slider.setValue(min(self.angle_slider.value(), last_slot))
#             self.depth_slider.setRange(0, max(0, volume.shape[2] - 1))
#
#             if info.get("segment_plan"):
#                 pages_text = (
#                     f"used {info['pages_used']} pages from segment "
#                     f"#{int(info.get('segment_index', 0)) + 1}"
#                 )
#             else:
#                 pages_text = (
#                     f"used {info['pages_used']} of {info['pages_total']} pages "
#                     f"({info['pages_discarded']} discarded)"
#                 )
#             summary = [
#                 f"volume {volume.shape[0]}x{volume.shape[1]}x{volume.shape[2]}",
#                 f"half-turn {info['half_turn_blines']} B-lines "
#                 f"({info['degrees_per_bline']:.4f} deg/B-line)",
#                 f"{info['angular_samples']} directions over 360 deg",
#                 pages_text,
#                 f"grid {info['n_grid']} px ({info['grid_um']:.1f} um/px)",
#                 f"radius {info['radius_um']:.2f} mm",
#             ]
#             window = info.get("page_window")
#             if info.get("segment_plan") and window:
#                 summary.append(f"pages {window[0]}..{window[1]} of the recording")
#             missing = int(info.get("missing_slots", 0))
#             if missing:
#                 summary.append(
#                     f"{missing} direction(s) had no B-line: the file has fewer "
#                     f"pages than requested"
#                 )
#             summary.extend(info.get("notes", []))
#             self.info_label.setText("  |  ".join(summary))
#
#             self._set_busy(False, None, busy_apply=False)
#             self.export_button.setEnabled(True)
#             self.auto_contrast()
#
        def _segment_report_text(self, plan, used_index=0):
            """Readable report: turn length, implied N, boundaries and phases."""
            if not plan:
                return (
                    "No segmentation was run: reconstructing from the first pages "
                    "of the file."
                )
            turn = int(plan.get("turn_pages", 0))
            score = float(plan.get("turn_score", float("nan")))
            score_text = "--" if math.isnan(score) else f"{score:.2f}"
            total = int(plan.get("pages_total", 0))
            holds = (total / turn) if turn else 0.0
            lines = [
                f"turn {turn} pages = {turn / 2.0:.1f} B-lines per 180 deg "
                f"(ncc {score_text})"
                + ("  [nominal fallback]" if plan.get("nominal") else "")
                + (f"  |  file holds {holds:.2f} turns" if turn else ""),
                describe_segment_plan(plan),
                "phases "
                + ", ".join(f"{value:.1f}" for value in plan.get("phases_deg", []))
                + f" deg  |  highlighted segment #{used_index + 1}",
            ]
            notes = [str(note) for note in plan.get("notes", [])]
            if notes:
                lines.append("note: " + "  ".join(notes))
            return "\n".join(lines)

        def _update_segment_report(self, info):
            """Record the detected segmentation and print its report.

            The report used to fill a text box in the window; the window keeps only
            one line of it now, so the details go to stdout, where they are still
            available when a number is needed.
            """
            plan = info.get("segment_plan")
            self.segment_plan = plan
            used_index = int(info.get("segment_index", 0))
            print("[RotationalReconstruct] segmentation report:")
            for line in self._segment_report_text(plan, used_index).splitlines():
                print("[RotationalReconstruct]   " + line)

        def _segment_snapshot(self):
            """Segment plan in a JSON-friendly shape (empty until detected)."""
            plan = self.info.get("segment_plan")
            if not plan:
                return {}
            return {
                "offsets": [int(value) for value in plan.get("offsets", [])],
                "lengths": [int(value) for value in plan.get("lengths", [])],
                "is_full": [bool(value) for value in plan.get("is_full", [])],
                "phases_deg": [float(value) for value in plan.get("phases_deg", [])],
                "scores": [None if math.isnan(float(value)) else float(value)
                           for value in plan.get("scores", [])],
                "turn_pages": int(plan.get("turn_pages", 0)),
                "turn_score": (None if math.isnan(float(plan.get("turn_score", float("nan"))))
                               else float(plan.get("turn_score", 0.0))),
                "nominal": bool(plan.get("nominal", False)),
                "pages_used": int(plan.get("pages_used", 0)),
                "pages_total": int(plan.get("pages_total", 0)),
                "pages_discarded": int(plan.get("pages_discarded", 0)),
                "notes": [str(note) for note in plan.get("notes", [])],
            }

        # ===== STAGE 2 - commented out while only the segmentation is developed =====
#         def _parameter_snapshot(self):
#             return {
#                 "half_turn_blines": float(self.rotation_spin.value()),
#                 "xc": float(self.center_spin.value()),
#                 "degrees_per_bline": float(self.info.get("degrees_per_bline", 0.0)),
#                 "angular_samples": int(self.info.get("angular_samples", 0)),
#                 "pages_used": int(self.info.get("pages_used", 0)),
#                 "pages_total": int(self.info.get("pages_total", 0)),
#                 "pages_discarded": int(self.info.get("pages_discarded", 0)),
#                 "spacing": float(self.info.get("spacing", 1.0)),
#                 "page_offset": int(self.info.get("page_offset", 0)),
#                 "page_count": int(self.info.get("page_count", 0)),
#                 "segment_index": int(self.info.get("segment_index", 0)),
#                 "segments": self._segment_snapshot(),
#             }
#
#         # ------------------------------------------------------------------ #
#         # Views
#         # ------------------------------------------------------------------ #
#
#         def _prepare_image(self, image):
#             prepared = np.asarray(image, dtype=np.float32)
#             if self.median_check.isChecked() and NDIMAGE is not None:
#                 prepared = NDIMAGE.median_filter(prepared, size=3)
#             if self.log_check.isChecked():
#                 prepared = np.log10(1.0 + np.maximum(prepared, 0.0))
#             return prepared
#
#         def _contrast_limits(self):
#             low = float(self.vmin_spin.value())
#             high = float(self.vmax_spin.value())
#             if high <= low:
#                 high = low + 1.0
#             return low, high
#
#         def _current_xy_image(self):
#             if self.volume is None:
#                 return None
#             index = max(0, min(int(self.depth_slider.value()), self.volume.shape[2] - 1))
#             return self.volume[:, :, index], index
#
#         def update_xy_view(self):
#             current = self._current_xy_image()
#             if current is None:
#                 return
#             image, index = current
#             display = self._prepare_image(image)
#             low, high = self._contrast_limits()
#             axes = self.xy_axes
#             axes.clear()
#             axes.set_facecolor("black")
#             axes.imshow(display, cmap="gray", vmin=low, vmax=high,
#                         origin="lower", aspect="equal", interpolation="nearest")
#             axes.set_title(f"XY  z={index}", color="white", fontsize=9)
#             axes.tick_params(colors="white", labelsize=7)
#             self.depth_label.setText(f"{index} / {self.volume.shape[2] - 1}")
#             self.xy_canvas.draw_idle()
#
#         def update_xz_view(self):
#             if self.data is None or not self.slot_pages:
#                 return
#             slot = max(0, min(int(self.angle_slider.value()), len(self.slot_pages) - 1))
#             pages = self.slot_pages[slot]
#             if pages:
#                 profile = self.data[pages[0]]
#                 caption = f"B-line {pages[0]}"
#             else:
#                 # No B-line behind this direction: show it as empty instead of
#                 # leaving the previous angle on screen.
#                 profile = np.zeros(self.data.shape[1:], dtype=np.float32)
#                 caption = "no B-line"
#             display = self._prepare_image(np.transpose(profile))
#             low, high = self._contrast_limits()
#             axes = self.xz_axes
#             axes.clear()
#             axes.set_facecolor("black")
#             axes.imshow(display, cmap="gray", vmin=low, vmax=high,
#                         origin="upper", aspect="auto", interpolation="nearest")
#             if getattr(self, "applied_xc", None) is not None:
#                 axes.axvline(float(self.applied_xc), color="#ff4040",
#                              linewidth=0.8, alpha=0.8)
#             angle = float(self.slot_angles[slot]) if self.slot_angles.size else 0.0
#             axes.set_title(f"XZ  angle={angle:.2f} deg  ({caption})",
#                            color="white", fontsize=9)
#             axes.tick_params(colors="white", labelsize=7)
#             self.angle_label.setText(f"{slot} / {len(self.slot_pages) - 1}")
#             self.xz_canvas.draw_idle()
#
#         def refresh_views(self):
#             self.update_xy_view()
#             self.update_xz_view()
#
#         def auto_contrast(self):
#             current = self._current_xy_image()
#             if current is None:
#                 return
#             image = self._prepare_image(current[0])
#             positive = image[image > 0]
#             if positive.size == 0:
#                 low, high = 0.0, 1.0
#             else:
#                 low = float(np.percentile(positive, 1.0))
#                 high = float(np.percentile(positive, 99.5))
#                 if high <= low:
#                     high = low + 1.0
#             self.vmin_spin.setValue(low)
#             self.vmax_spin.setValue(high)
#             self.refresh_views()
#
#         # ------------------------------------------------------------------ #
#         # Export
#         # ------------------------------------------------------------------ #
#
#         def export_volume_dialog(self):
#             if self.volume is None:
#                 return
#             source = self.current_source or (self.meta.get("files") or ["RotVol"])[0]
#             suggested = default_volume_filename(
#                 source, self.volume.shape[0], self.volume.shape[2]
#             )
#             path, _filter = QtWidgets.QFileDialog.getSaveFileName(
#                 self, "Export reconstructed volume", str(suggested),
#                 "TIFF files (*.tif *.tiff)",
#             )
#             if not path:
#                 return
#             QtWidgets.QApplication.setOverrideCursor(Qt.WaitCursor)
#             try:
#                 tif_path, json_path = export_volume(
#                     self.volume, path, self.meta, self._parameter_snapshot()
#                 )
#             except Exception as error:  # noqa: BLE001 - reported to the user
#                 QtWidgets.QMessageBox.critical(self, "Export failed", str(error))
#             else:
#                 self.info_label.setText(
#                     f"Exported {tif_path.name} and {json_path.name} to {tif_path.parent}"
#                 )
#             finally:
#                 QtWidgets.QApplication.restoreOverrideCursor()
#
#
def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Rotational C-scan reconstruction into a 3D volume"
    )
    parser.add_argument("paths", nargs="*", help="Cscan-*.tif files to open")
    parser.add_argument("--self-test", action="store_true",
                        help="run the synthetic round-trip validation and exit")
    parser.add_argument("--half-turn-blines", type=float, default=None,
                        help="nominal number of B-lines spanning 180 degrees; the "
                             "window has no field for it, the detector only uses it "
                             "as the centre of its search window and reports the "
                             "measured turn length instead")
    parser.add_argument("--center", type=float, default=None,
                        help="initial rotation centre pixel inside the B-line")
    parser.add_argument("--depth-rows", type=str, default=None, metavar="START,STOP",
                        help="depth pixel window the segmentation may measure "
                             f"(default {DEFAULT_DEPTH_WINDOW[0]},{DEFAULT_DEPTH_WINDOW[1]}); "
                             "every depth row is loaded and displayed either way - "
                             "the window only keeps the sample holder out of the "
                             "numbers - and \"all\" measures over every row")
    args = parser.parse_args(argv)

    if args.self_test:
        return run_self_test()

    depth_window = DEFAULT_DEPTH_WINDOW
    if args.depth_rows is not None:
        text = str(args.depth_rows).strip().lower()
        if text in ("all", "full", "none"):
            depth_window = None
        else:
            parts = [part for part in text.replace(":", ",").split(",") if part.strip()]
            if len(parts) != 2:
                raise SystemExit("--depth-rows wants \"START,STOP\" (e.g. 100,350) or \"all\"")
            try:
                depth_window = (int(round(float(parts[0]))), int(round(float(parts[1]))))
            except ValueError:
                raise SystemExit("--depth-rows wants \"START,STOP\" (e.g. 100,350) or \"all\"")

    # The window has no field for the nominal rotation speed any more, so this is
    # the only way to move the centre of the detector's search window.
    nominal_n = (float(args.half_turn_blines) if args.half_turn_blines is not None
                 else float(DEFAULT_HALF_TURN_BLINES))
    if not (MIN_HALF_TURN_BLINES <= nominal_n <= MAX_HALF_TURN_BLINES):
        raise SystemExit(
            f"--half-turn-blines wants {MIN_HALF_TURN_BLINES}..{MAX_HALF_TURN_BLINES}")

    if not QT_AVAILABLE:
        raise SystemExit(
            "PyQt5 is required for the interactive window. "
            f"Import failed with: {QT_IMPORT_ERROR}"
        )

    print(f"[RotationalReconstruct] running {SCRIPT_PATH}")
    print(f"[RotationalReconstruct] build: {BUILD_TAG}")
    print("[RotationalReconstruct] STAGE 1: segmentation only - the polar assembly, "
          "the reconstruction and the export are commented out in this file")
    print("[RotationalReconstruct] window: the scan import and the segmentation "
          "result (the summed pages, the summed turn and the per-segment numbers)")
    window_text = ("all rows" if depth_window is None
                   else f"{depth_window[0]}-{depth_window[1]} px")
    print(f"[RotationalReconstruct] depth window for the measurements: {window_text} "
          "(the whole depth is loaded and displayed; the window only keeps the "
          "sample holder out of the numbers)")
    print(f"[RotationalReconstruct] nominal B-lines per 180 deg: {nominal_n:g} "
          "(only the centre of the detector's search window; the measured turn is "
          "reported in the window)")
    print(f"[RotationalReconstruct] self-test (must print 14 checks): "
          f"python \"{SCRIPT_PATH}\" --self-test")

    application = QtWidgets.QApplication(sys.argv[:1])
    window = ReconstructionWindow(
        args.paths or None,
        initial_center=args.center,
        initial_half_turn_blines=args.half_turn_blines,
        initial_depth_window=depth_window,
    )
    window.show()
    return application.exec_()


if __name__ == "__main__":
    sys.exit(main())
