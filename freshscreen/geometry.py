"""Voxel sizes and Neuroglancer source transforms, derived from nhdr headers.

Neuroglancer works in meters; nhdr headers are in millimeters.
"""
import logging
import re
from typing import OrderedDict

import numpy as np

# Offset (mm) from the volume origin to the center of the anterior commissure in DMBA space.
# Subtracting it puts the anterior commissure, rather than Bregma, at the scene center.
DMBA_OFFSET_CORRECTION = [0.005199037941243297, 0.004563725401255936, -4.169626074596984]


def get_voxel_size_from_nhdr_dict(nhdr: OrderedDict):
    """voxel size (in meters) of each spatial dimension, from a header read by nrrd.read_header

    Does not assume isotropic volumes. Only the first 3 dimensions are used, which drops the channel axis of color files."""
    sizes = []
    for i in range(3):
        size = max(abs(nhdr["space directions"][i]))
        # conversion from mm to m
        size = size / 1000
        sizes.append(size)
    return sizes


def get_voxel_size_from_color_nhdr(color_nhdr_file):
    """voxel size (mm) of the 3 spatial axes, read straight from the header text.
    pynrrd can't read LIST color headers, and the 'none' channel axis can be first or last."""
    with open(color_nhdr_file) as f:
        for line in f:
            if line.startswith('space directions:'):
                vecs = re.findall(r'\(([^)]*)\)', line)
                return [max(abs(float(v)) for v in vec.split(',')) for vec in vecs]
    raise ValueError("no space directions in {}".format(color_nhdr_file))


def make_empty_transform():
    """returns a neuroglancer source transform for update_dimensions and update_matrix_transform to fill in"""
    return {
        "matrix": [[0, 0, 0, 0] for _ in range(3)],
        "inputDimensions": {axis: [0, "m"] for axis in "xyz"},
        "outputDimensions": {axis: [0, "m"] for axis in "xyz"},
    }


def update_dimensions(data_layer: dict, voxel_sizes: list, dimension_field: str, job_id: str = ""):
    if dimension_field not in ["inputDimensions", "outputDimensions"]:
        logging.warning("unrecognized dimension field: {}\nCan not write. Only allowed options are: 'inputDimensions' and 'outputDimensions'".format(dimension_field))
        return

    i = 0
    for key in data_layer["source"]["transform"][dimension_field].keys():
        data_layer["source"]["transform"][dimension_field][key][0] = voxel_sizes[i]
        i += 1
    logging.info("updated {} {} to: {}".format(job_id, dimension_field, data_layer["source"]["transform"][dimension_field]))


def update_matrix_transform(xform_matrix: list, nhdr_dict: dict, label_voxel_sizes: list, is_stereotaxic=True):
    """fills in a 3x4 neuroglancer transform (3x3 rotation + translation column) from the nhdr, in place

    The rotation is the sign of each nhdr space direction. The translation is the
    nhdr origin in label voxels (so divide by the MRI voxel size even for lightsheet)."""
    data_nhdr_matrix = nhdr_dict["space directions"]
    temp_shape = data_nhdr_matrix.shape
    data_nhdr_matrix = list(map(lambda x: x / abs(x) if x != 0 else 0, data_nhdr_matrix.flatten()))
    data_nhdr_matrix = np.asarray(data_nhdr_matrix, dtype=int).reshape(temp_shape)

    for i in range(data_nhdr_matrix.shape[0]):
        for j in range(data_nhdr_matrix.shape[1]):
            xform_matrix[i][j] = data_nhdr_matrix[i][j]

    # label_voxel_sizes are in METERS. data_nhdr["space origin"] are in MM. so divide by an extra 1000
    for i in range(len(label_voxel_sizes)):
        temp = nhdr_dict["space origin"][i] / label_voxel_sizes[i] / 1000
        xform_matrix[i][-1] = temp

    # Without a correction the volume origin lands at the scene center, which is right for
    # symmetric15um-aligned data. For DMBA-aligned data that puts Bregma at the center, so
    # shift by the anterior commissure offset instead; this matches previous behavior.
    if is_stereotaxic:
        print(xform_matrix)
        for i in range(len(DMBA_OFFSET_CORRECTION)):
            xform_matrix[i][-1] = xform_matrix[i][-1] - (DMBA_OFFSET_CORRECTION[i] / label_voxel_sizes[i] / 1000)


def flip_xform_dimension(xform_matrix: list):
    """flips the row that maps to z, so the coronal view in neuroglancer is right side up. in place"""
    image_matrix_numpy_array = np.asarray(xform_matrix)
    flip_row_index = np.argmax(image_matrix_numpy_array[:, 2] != 0)
    for i in range(len(xform_matrix[flip_row_index])):
        xform_matrix[flip_row_index][i] = -1 * xform_matrix[flip_row_index][i]


def get_crossSectionScale_factor(voxel_size):
    """initial zoom of a scene. 2 works well for 15um data, 0.8 for 45um. voxel_size is in meters"""
    eps = 1e-15
    scale_factor = 2
    if abs(abs(voxel_size) - 0.000015) < eps:
        scale_factor = 2
    if abs(abs(voxel_size) - 0.000045) < eps:
        scale_factor = 0.8
    logging.info("setting scale factor to: {}".format(scale_factor))
    return scale_factor
