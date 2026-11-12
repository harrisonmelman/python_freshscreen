from dataclasses import dataclass
@dataclass
class Dataset:
    """Class for keeping track of an n5 file on freshscreen and all of the relevant metadata
    display_range is calculated from a lowres n5 copy
    space parameters are read from the nhdr file used to create the n5"""
    filename: str            # S3 basename, e.g. 15mo-WT-M_N02_dwi.n5 or .precomputed for labels
    runno: str               # N12345 or group name 15mo-WT-M
    contrast: str            # dwi, fa, gqi-color, ...
    kind: str                # "image" | "color" | "labels"
    url: str                 # get_s3_url_from_file_basename(filename)
    voxel_size: list[float]  # meters, x/y/z
    space_directions: list[list[float]]  # from the nhdr, mm
    space_origin: list[float]            # from the nhdr, mm
    default_color: str | None = "white"
    display_range: dict | None = None    # {"range": [...], "window": [...]}; None for labels
    age: str | None = None   # "15mo", parsed from the group prefix

    def to_dict(self): ...
    @classmethod
    def from_dict(cls, d): ...

