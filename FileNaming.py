import os


def _shape3(shape):
    return int(shape[0]), int(shape[1]), int(shape[2])


def cscan_filename(cscan_num, shape):
    ypix, xpix, zpix = _shape3(shape)
    return f"Cscan-{cscan_num}-Y{ypix}-X{xpix}-Z{zpix}.tif"


def cscan_mean_volume_filename(cscan_num, shape):
    ypix, xpix, zpix = _shape3(shape)
    return f"CscanMean-{cscan_num}-Y{ypix}-X{xpix}-Z{zpix}.tif"


def bline_filename(bline_num, shape):
    yrpt, xpix, zpix = _shape3(shape)
    return f"Bline-{bline_num}-Yrpt{yrpt}-X{xpix}-Z{zpix}.tif"


def aline_filename(aline_num, shape):
    yrpt, xrpt, zpix = _shape3(shape)
    return f"Aline-{aline_num}-Yrpt{yrpt}-Xrpt{xrpt}-Z{zpix}.tif"


class FileNaming:
    def __init__(self, ui):
        self.ui = ui
        self.aline_num = 1
        self.bline_num = 1
        self.cscan_num = 1

    def _base_dir(self):
        return self.ui.DIR.toPlainText()

    def _current_time_number(self):
        if hasattr(self.ui, "timeReader"):
            return int(self.ui.timeReader.value())
        if hasattr(self.ui, "CuSlice"):
            return int(self.ui.CuSlice.value())
        return 1

    def save_dir(self, acq_mode):
        return self._base_dir()

    def reset_all_counters(self):
        self.aline_num = 1
        self.bline_num = 1
        self.cscan_num = 1

    def increment_aline(self):
        self.aline_num += 1

    def increment_bline(self):
        self.bline_num += 1

    def increment_cscan(self):
        self.cscan_num += 1

    def get_filename(self, kind, acq_mode, shape, ypixels=None):
        base_dir = self.save_dir(acq_mode)

        if kind == "aline":
            return os.path.join(base_dir, aline_filename(self.aline_num, shape))
        if kind == "bline":
            return os.path.join(base_dir, bline_filename(self.bline_num, shape))
        if kind == "cscan":
            return os.path.join(base_dir, cscan_filename(self.cscan_num, shape))
        if kind == "cscan_mean":
            return os.path.join(base_dir, cscan_mean_volume_filename(self.cscan_num, shape))
        raise ValueError(f"Unknown filename kind: {kind}")
