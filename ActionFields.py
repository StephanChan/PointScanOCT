# -*- coding: utf-8 -*-
"""Queue action field classes shared across worker threads."""

from ActionTypes import EXIT_ACTION


class AODOActionField:
    def __init__(self, action, direction=1):
        super().__init__()
        self.action = action
        self.direction = direction


class WeaverActionField:
    def __init__(self, action, acq_mode=None, context=None):
        super().__init__()
        self.action = action
        self.acq_mode = action if acq_mode is None else acq_mode
        self.context = [] if context is None else context


class DnSActionField:
    def __init__(
        self,
        action,
        acq_mode=None,
        data=None,
        raw=False,
        context=None,
        gpu_avg_count=1,
        filename_bundle=None,
        skip_save=False,
        package_start=0,
        package_frame_count=None,
        package_total_frames=None,
        is_final=True,
        profile=None,
        output_buffer_token=None,
    ):
        super().__init__()
        self.action = action
        self.acq_mode = action if acq_mode is None else acq_mode
        self.data = [] if data is None else data
        self.raw = raw
        self.context = [] if context is None else context
        self.gpu_avg_count = gpu_avg_count
        self.filename_bundle = {} if filename_bundle is None else filename_bundle
        self.skip_save = bool(skip_save)
        self.package_start = int(package_start or 0)
        self.package_frame_count = package_frame_count
        self.package_total_frames = package_total_frames
        self.is_final = bool(is_final)
        self.profile = {} if profile is None else dict(profile)
        self.output_buffer_token = output_buffer_token


class GPUActionField:
    def __init__(
        self,
        action,
        DnS_action='',
        acq_mode=None,
        memory_slot=0,
        context=None,
        filename_bundle=None,
        skip_save=False,
        package_start=0,
        package_frame_count=None,
        package_total_frames=None,        is_final=True,
        profile=None,
    ):
        super().__init__()
        self.action = action
        self.DnS_action = DnS_action
        self.acq_mode = DnS_action if acq_mode is None else acq_mode
        self.memory_slot = memory_slot
        self.context = [] if context is None else context
        self.filename_bundle = {} if filename_bundle is None else filename_bundle
        self.skip_save = bool(skip_save)
        self.package_start = int(package_start or 0)
        self.package_frame_count = package_frame_count
        self.package_total_frames = package_total_frames
        self.is_final = bool(is_final)
        self.profile = {} if profile is None else dict(profile)


class DActionField:
    def __init__(self, action):
        super().__init__()
        self.action = action


class DbackActionField:
    def __init__(
        self,
        memory_slot,
        error=None,
        package_start=0,
        package_frame_count=None,
        package_total_frames=None,        is_final=True,
        profile=None,
    ):
        super().__init__()
        self.memory_slot = memory_slot
        self.error = error
        self.package_start = int(package_start or 0)
        self.package_frame_count = package_frame_count
        self.package_total_frames = package_total_frames
        self.is_final = bool(is_final)
        self.profile = {} if profile is None else dict(profile)


class EXITField:
    def __init__(self):
        super().__init__()
        self.action = EXIT_ACTION
