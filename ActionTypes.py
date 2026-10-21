# -*- coding: utf-8 -*-
"""Central action and acquisition type names shared across worker threads."""

EXIT_ACTION = "exit"


class AcqTypes:
    CONTINUOUS_ALINE = "ContinuousAline"
    FINITE_ALINE = "FiniteAline"
    FINITE_BSCAN = "FiniteBscan"
    CONTINUOUS_BLINE = "ContinuousBline"
    CONTINUOUS_KX = "ContinuousKX"
    # Diagnostic mode: one B-line acquisition, displayed as the log-scale intensity
    # spectrum obtained by FFT along the X (galvo) direction.
    X_FREQUENCY = "Xfrequency"

    FINITE_BLINE = "FiniteBline"
    CONTINUOUS_CSCAN = "ContinuousCscan"
    FINITE_CSCAN = "FiniteCscan"
    LOCATION_CAMERA_LIVE = "LocationCameraLive"


class WeaverActions:
    ZSTAGE_REPEATIBILITY = "ZstageRepeatibility"
    GET_BACKGROUND = "get_background"
    GET_SURFACE = "get_surface"
    GOTO_ZERO = "Gotozero"


class GPUActions:
    GPU = "GPU"
    CPU = "CPU"
    CLEAR = "Clear"
    UPDATE_DISPERSION = "update_Dispersion"
    UPDATE_BACKGROUND = "update_background"
    DISPLAY_FFT_ACTIONS = "display_FFT_actions"
    DISPLAY_COUNTS = "display_counts"
    # 整卷 Y 向陷波：DnS 把整卷交给 GPU 线程，结果从 VolumeResultQueue 取回
    Y_NOTCH_VOLUME = "y_notch_volume"


class DnSActions:
    CLEAR = "Clear"
    DISPLAY_COUNTS = "display_counts"
    AGAR_TILE = "agarTile"
    WRITE_AGAR = "WriteAgar"


