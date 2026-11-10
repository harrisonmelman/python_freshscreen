"""Per-contrast display defaults."""
from .naming import get_contrast_from_filename

DEFAULT_THRESHOLDS = {
    "dwi": (0, 25000),
    "fa": (0, 1),
    "nqa": (0, 1),
    "ad": (0, 0.8),
    "rd": (0, 0.8),
    "md": (0, 0.8),
    "qa": (0, 1),
    "gfa": (0, 1),
    "color": (0, 128),
    "NeuN": (0, 1500),
    "MBP": (0, 1300),
    "MOG": (0, 2500),
    "PDE10A": (0, 2000),
    "autof": (0, 1200),
    "SST": (0, 1000),
    "mHTT": (0, 6000),
    "NFH": (0, 2000),
    "NPY": (0, 1200),
    "Thy1": (0, 4000),
    "IBA1": (0, 1300),
    "Syto16": (0, 12000),
    "tdi": (0, 20),
    "tdi3": (0, 20),
    "tdi5": (0, 20),
    "m0": (0, 70000),
    "m1": (0, 45000),
    "m2": (0, 25000),
    "m3": (0, 15000),
    "b0": (0, 30000),
    "annotation": (0, 100),
    "average-template": (0, 100),
    "ara-nissl": (0, 100),
    "nqa-color": (0, 100),
    "gqi-color": (0, 100),
    "tdi-color": (0, 100),
    "tdi5um-color": (0, 100),
    "tdi3-color": (0, 100),
}

# thresholds for the 17gaj40 project, kept for reference. not currently used
D17GAJ40_THRESHOLDS = {
    "dwi": (0, 5000),
    "fa": (0, 1),
    "lc": (0, 1),
    "ad": (0, 0.0009),
    "rd": (0, 0.0005),
    "md": (0, 0.0006),
    "color": (0, 128),
    "tdi": (0, 20),
    "tdi3": (0, 20),
    "tdi5": (0, 20),
    "b0": (0, 15000),
    "nqa-color": (0, 100),
    "gqi-color": (0, 100),
    "tdi-color": (0, 100),
    "tdi3-color": (0, 100),
}

DEFAULT_OVERLAY_COLORS = {
    "NeuN": "green",
    "MBP": "red",
    "PDE10A": "cyan",
    "mHTT": "yellow",
}


def get_default_threshold(filename: str):
    """returns the predefined (min, max) display threshold for the file's contrast, or (0, 1) if unknown

    Depends on current (11/7/2022) freshscreen filenames. Scenes now compute the range
    from the data (see display_range); this table is the fallback reference."""
    contrast = get_contrast_from_filename(filename)
    if contrast in DEFAULT_THRESHOLDS:
        return DEFAULT_THRESHOLDS[contrast]
    print("WARNING: contrast unknown. using default contrast boundaries of (0,1)")
    return (0, 1)


def get_default_color_overlay(filename):
    contrast = get_contrast_from_filename(filename)
    if contrast in DEFAULT_OVERLAY_COLORS:
        return DEFAULT_OVERLAY_COLORS[contrast]
    print("WARNING: contrast unknown. using default color of grayscale")
    return "white"
