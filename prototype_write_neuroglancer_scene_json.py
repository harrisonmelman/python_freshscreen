# docstring convention example:
# like git commit rules. one line short description, skip a line, then more descriptive
#def complex(real=0.0, imag=0.0):
"""Form a complex number.

    Keyword arguments:
    real -- the real part (default 0.0)
    imag -- the imaginary part (default 0.0)
    """

# IMPORTS
from typing import OrderedDict
import urllib.parse
import json
import logging
import os
import re
import sys
import nrrd
import numpy as np
import functools
import glob
import subprocess
import csv
import re
import zarr

# GLOBAL VARIABLES -- caps-locking these to make it obvious they are global and BAD
# needed at the beginning of all aws s3 filepaths. holds the file protocol and s3 bucket information
# example usage:
    # n5_filepath="{}/{}/{}.format(N5_PREFIX, filename, N5_SUFFIX)"
    # precomputed_filepath="{}/{}.format(PRECOMPUTED_PREFIX, filename)"
#N5_PREFIX = "n5://https://d3mof5o.s3.amazonaws.com"
#PRECOMPUTED_PREFIX = "precomputed://https://d3mof5o.s3.amazonaws.com"
N5_PREFIX = "n5://https://fast-s3.oit.duke.edu/freshscreen/dh-civm-public"
PRECOMPUTED_PREFIX = "precomputed://https://fast-s3.oit.duke.edu/freshscreen/dh-civm-public"
# this is needed at the end of all N5 file paths, to tell neuroglancer where the data lives in the folder struct
N5_SUFFIX="setup0/timepoint0/"
LIGHTSHEET_CONTRASTS = ["NeuN", "autof", "Thy1", "MBP", "ChAT"]
RCLONE = "K:/DevApps/rclone-v1.58.1-windows-amd64/rclone.exe"

# a class to allow json to encode assorted numpy data types
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super(NpEncoder, self).default(obj)
def get_fsname_from_fsfile(filename: str):
    _, filename = os.path.split(filename)
    match = re.match(r'(.+?)(_M4D)?\.(.+)', filename)

# TODO: this needs to be refactored or reconsidered
def get_runno_from_filename(filename: str):
    _,filename = os.path.split(filename)
    f = filename.split("_")
    print(f)
    # run number should always be the second to last field in the freshscreen name
    guess = f[-2]
    # matching criteria are it starts with N or S AND is length 6 (N12345) or length 11 (N12345NLSAM)
    if guess is not None and guess[0] in "NS" and len(guess) in [6,11]:
        return guess
    # if for whatever reason this does not match as a runnumber, loop through the others to see if you can find it
    for guess in f:
        if guess is not None and guess[0] in "NS" and len(guess) in [6,11]:
            return guess
    logging.warning("unable to find specimen run number from the filename: {}\n\t Will attempt to treat it as an atlas dataset. Setting runno to: {}".format(filename, f[0]))
    return f[0]


def get_contrast_from_filename(filename: str):
    # in freshscren, all filenames look similar (${spec_id}_${number}_${runno}_${contrast}.n5)
    # split on "_" and take the last one
    # split on "." to remove the file extension
    return filename.split("_")[-1].split(".")[0]


def _pick_array(node, max_voxels):
    """Return an array to sample: the node itself, or the best pyramid level in a group."""
    if isinstance(node, zarr.Array):
        return node
    levels = sorted(
        (k for k in node.array_keys() if re.fullmatch(r"s\d+", k)),
        key=lambda k: int(k[1:]),
    )
    if not levels:
        raise ValueError(f"No array or s0/s1/... pyramid levels found in {node.name!r}")
    # Finest level that fits the voxel budget; otherwise the coarsest available
    for k in levels:
        if node[k].size <= max_voxels:
            return node[k]
    return node[levels[-1]]


def _sample_chunks(arr, max_voxels, rng):
    """Read a random subset of whole chunks (cheap: each chunk is read once)."""
    grid = [int(np.ceil(s / c)) for s, c in zip(arr.shape, arr.chunks)]
    n_chunks = int(np.prod(grid))
    voxels_per_chunk = int(np.prod(arr.chunks))
    n = max(1, min(n_chunks, max_voxels // voxels_per_chunk))
    picks = rng.choice(n_chunks, size=n, replace=False)

    samples = []
    for flat in picks:
        idx = np.unravel_index(flat, grid)
        sl = tuple(
            slice(i * c, min((i + 1) * c, s))
            for i, c, s in zip(idx, arr.chunks, arr.shape)
        )
        samples.append(np.asarray(arr[sl]).ravel())
    return np.concatenate(samples)



def compute_display_range(n5_path,dataset="setup0/timepoint0",max_voxels=2000000,lower_pct=0.5,upper_pct=99.5,ignore_value=0,seed=0):
    root = zarr.open(zarr.N5Store(n5_path), mode="r")
    node = root[dataset] if dataset else root
    arr = _pick_array(node, max_voxels)

    rng = np.random.default_rng(seed)
    if arr.size <= max_voxels:
        data = np.asarray(arr[:]).ravel()
    else:
        data = _sample_chunks(arr, max_voxels, rng)

    if np.issubdtype(data.dtype, np.floating):
        data = data[np.isfinite(data)]
    if ignore_value is not None:
        data = data[data != ignore_value]

    if data.size == 0:
        # Nothing usable sampled: fall back to the dtype's natural range
        if np.issubdtype(arr.dtype, np.integer):
            info = np.iinfo(arr.dtype)
            return {"range": [float(info.min), float(info.max)],
                    "window": [float(info.min), float(info.max)]}
        return {"range": [0.0, 1.0], "window": [0.0, 1.0]}

    lo, hi = np.percentile(data, [lower_pct, upper_pct])
    if hi <= lo:  # constant data
        hi = lo + 1
    return {
        "range": [float(lo), float(hi)],
        "window": [float(min(data.min(), lo)), float(max(data.max(), hi))],
    }


def get_default_threshold(filename: str):
    """from filename, figure out which type of contrast it is, and return the appropriate threshold value from the dictionary

    Completely dependendent on current (11/7/2022) freshscreen filenames"""
    DEFAULT_THRESHOLDS = {
        "dwi" : (0,25000),
        "fa" : (0,1),
        "nqa": (0,1),
        "ad" : (0,0.8),
        "rd" : (0,0.8),
        "md" : (0,0.8),
        "qa" : (0,1),
        "gfa" : (0,1),
        "color" : (0,128),
        "NeuN" : (0,1500),
        "MBP" : (0,1300),
        "MOG" : (0,2500),
        "PDE10A" : (0,2000),
        "autof" : (0,1200),
        "SST" : (0,1000),
        "mHTT" : (0,6000),
        "NFH" : (0,2000),
        "NPY" : (0,1200),
        "Thy1" : (0,4000),
        "IBA1" : (0,1300),
        "Syto16" : (0,12000),
        "tdi" : (0,20),
        "tdi3" : (0,20),
        "tdi5" : (0,20),
        "m0" : (0,70000),
        "m1" : (0,45000),
        "m2" : (0,25000),
        "m3" : (0,15000),
        "b0" : (0,30000),
        "annotation" : (0,100),
        "average-template" : (0,100),
        "ara-nissl" : (0,100),
        "nqa-color" : (0,100),
        "gqi-color" : (0,100),
        "tdi-color" : (0,100),
        "tdi5um-color" : (0,100),
        "tdi3-color" : (0,100)
    }
    # no idea if this will work as expected or not...
    d17gaj40_THRESHOLDS = {
        "dwi" : (0,5000),
        "fa" : (0,1),
        "lc" : (0,1),
        "ad" : (0,0.0009),
        "rd" : (0,0.0005),
        "md" : (0,0.0006),
        "color" : (0,128),
        "tdi" : (0,20),
        "tdi3" : (0,20),
        "tdi5" : (0,20),
        "b0" : (0,15000),
        "nqa-color" : (0,100),
        "gqi-color" : (0,100),
        "tdi-color" : (0,100),
        "tdi3-color" : (0,100)
    }
    contrast = get_contrast_from_filename(filename)
    if contrast in DEFAULT_THRESHOLDS:
        return DEFAULT_THRESHOLDS[contrast]
    else:
        print("WARNING: contrast unknown. using default contrast boundaries of (0,1)")
        return (0,1)
    #return d17gaj40_THRESHOLDS[contrast]



def get_default_color_overlay(filename):
    contrast = get_contrast_from_filename(filename)
    defaults = {
        "NeuN" : "green",
        "MBP" : "red",
        "PDE10A": "cyan",
        "mHTT": "yellow"
    }
    if contrast in defaults:
        return defaults[contrast]
    else:
        print("WARNING: contrast unknown. using default color of grayscale")
        return "white"


def convert_dict_to_string(data: dict):
    return json.dumps(data, ensure_ascii=False, cls=NpEncoder)


def write_dict_to_json_file(file: str, data: dict):
    """converts a dict to json format and writes it to the file path provided
    WILL OVERWRITE any existing file at that location"""
    output_dir = os.path.dirname(file)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    with open(file, 'w') as f:
        # i THINK that ensure_ascii=False is analogous to python2's encoding='utf-8'
        # json cannot handle numpy data types so i must define my own encoder
        # TODO: encoder stolen from stack overflow. ensure that it works
        # dump writes to file. dumps writes to a string.
        # any other differences? idk
        json.dump(data, f, ensure_ascii=False, cls=NpEncoder)


def convert_json_to_url(file: str):
    """file is the path to a json file that contains scene information for neuroglancer
    Returns a shareable neuroglancer link: "neuroglancer.freshscreen.com/#!{}".format(encoded_json)

    you can also pass a JSON in string format and it will handle that instead. this function can tell if it is a filepath or a json string"""

    encoded_text = ""
    # might be more accurate to check if the first character is {, but that leaves room for extra whitespace to break me
    if "{" in file:
        # then we have a JSON string, not a file to read from
        # below line breaks it bc it removes ALL spaces in the json
            # this is fine for most things (i.e. going from "name" : "fa" to "name":"fa")
            # but it breaks the internal code block that handles rendering. (i.e. "#uicontrol invlerp normalized(range=[0,1])" is NOT the same as "#uicontrolinvlerpnormalized(range=[0,1])" )
        #encoded_text = urllib.parse.quote("".join(file.split()))
        encoded_text = urllib.parse.quote(file)
    else:
        with open(file) as f:
            # TODO: check if the "no spaces" failure also happens when reading and encoding from a json file
            # the reason i did this originally was bc neuroglancer gave a "url too long" error
            encoded_text = urllib.parse.quote("".join(f.read().split()))
    # missing a "#!" at beggining
    url = "https://neuroglancer.freshscreen.com/#!{}".format(encoded_text)
    logging.info("\n\n{}".format(url))
    return url


def get_voxel_size_from_nhdr_dict(nhdr: OrderedDict):
    """Determines the voxel size (in meters) of an image given the nhdr file as an OrderedDict (already read into an object by nrrd.read_nhdr)
    does not assume isotropic volumes. returns the voxel size of each dimension as an array"""
    nhdr["space directions"]
    sizes = []
    n_dims = nhdr["space directions"].shape[0]
    # hard-coding in three here to "destroy" the 4th dimension (3) in color files
    for i in range(3):
    #for i in range(n_dims):
        size = max(abs(nhdr["space directions"][i]))
        # conversion from mm to m
        size = size / 1000
        sizes.append(size)
    return sizes


def get_s3_url_from_file_basename(filename: str, n5_suffix: str=N5_SUFFIX):
    """given a file basename (e.g. 200316-1-1_N10_N58204NLSAM_fa.n5), return the full url (in neuroglancer format) of that file on the d3mof5o s3 bucket

    This function determines if the file is of type n5 or precomputed, and builds the path accordingly"""
    # TODO: add an argument to make this work for arbitrary s3 buckets
    # TODO: test to ensure that os.path.join always does the right job here. PROBABLY fails on windows
    suffix = filename.split(".")[-1]
    if "n5" in suffix:
        return pathjoin(N5_PREFIX, filename, n5_suffix)
    elif "precomputed" in suffix:
        return pathjoin(PRECOMPUTED_PREFIX, filename)
    else:
        logging.warning("ERROR: file does not seem to be in n5 or precomputed format")
        return None


def update_matrix_transform(xform_matrix: dict, nhdr_dict: dict, label_voxel_sizes: list, is_stereotaxic=True):
    # update the transform matrix with what is in the nhdr -- 3x4 in neuroglancer. includes 3x3 rotation matrix and 3x1 translation vector
    # this SHOULD always be 3x3, but no point risking it
    # flatten array so it can be used inside of a lambda easily
    # divide by its own absolute value, so it preserves its sign but scales it to 1
    # convert back to a np array, convert it to int, and reshape it back to 3x3
    data_nhdr_matrix = nhdr_dict["space directions"]
    temp_shape = data_nhdr_matrix.shape
    data_nhdr_matrix = list(map(lambda x: x / abs(x) if x != 0 else 0, data_nhdr_matrix.flatten()))
    data_nhdr_matrix = np.asarray(data_nhdr_matrix, dtype=int).reshape(temp_shape)

    # TODO: vectorize?
    for i in range(data_nhdr_matrix.shape[0]):
        for j in range(data_nhdr_matrix.shape[1]):
            xform_matrix[i][j] = data_nhdr_matrix[i][j]

    # translation transform part
    # should retain the signs. just divide by the voxel size (of MRI -- so divide by 0.015 even for lightsheet)
    # label_voxel_sizes are in METERS. data_nhdr["space origin"] are in MM. so divide by an extra 1000
    for i in range(len(label_voxel_sizes)):
        # iterating over the last column
        temp = nhdr_dict["space origin"][i] / label_voxel_sizes[i] / 1000
        xform_matrix[i][-1] = temp

    # ABOVE NO LONGER CORRECT FOR DMBA DATA
    # this puts the volume origin at the center of the neuroglancer scene. appropriate for symmetric15um-aligned data
    # NOT appropriate for DMBA aligned data, because it puts Bregma at the center of the neyuroglancer scene
    # IF your data is aligned with DMBA, then you need to factor in a specific translation offset. This will instead put the center of the anterior Commissure at the center of the neuroglancer scene. this more or less matches previous functionality
    # 0.005199037941243297,0.004563725401255936,-4.169626074596984 (in mm)
    if is_stereotaxic:
        DMBA_offset_correction=[0.005199037941243297,0.004563725401255936,-4.169626074596984]
        print(xform_matrix);
        for i in range(len(DMBA_offset_correction)):
            xform_matrix[i][-1] = xform_matrix[i][-1] - ( DMBA_offset_correction[i] / label_voxel_sizes[i] / 1000)

# a "fix" to make sure the coronal view in neuroglacner is right side up
def flip_xform_dimension(xform_matrix: dict, dim: int=2):
    image_matrix_numpy_array = np.asarray(xform_matrix)
    flip_row_index = np.argmax(image_matrix_numpy_array[:,2] != 0)
    # flip sign of each value in the found row
    for i in range(len(xform_matrix[flip_row_index])):
        xform_matrix[flip_row_index][i] = -1 * xform_matrix[flip_row_index][i]


# helper function for convert_lut_file
def rgba_to_hex(r, g, b, a):
    r=int(float(r))
    g=int(float(g))
    b=int(float(b))
    a=int(float(a))
    # :02x means "2 digits, lowercase hex" (use :02X for uppercase)
    return f"#{r:02x}{g:02x}{b:02x}{a:02x}"


# reads a CIVM complex stats lookup table and converts it to a neuroglancer compatible lut
# returns a dict with key=ROI number as a string, value=color in hex
def convert_lut_file(label_lut_file):
    roi_dict = {}
    with open(label_lut_file, 'r') as file:
        # resolve header fields
        # Done this way to safely have 'ROI' as the key field instead of '# ROI'
        original_reader = csv.reader(file,delimiter='\t')
        original_headers = next(original_reader)

        # strip whitespace and replace "# ROI" with "ROI"
        cleaned_headers = [h.strip().replace('# ROI', 'ROI') for h in original_headers]

        # Skip the comment row
        next(file)

        # use dict reader with cleaned headers
        reader = csv.DictReader(file, fieldnames=cleaned_headers, delimiter='\t')

        # Now iterate as normal
        for row in reader:
            roi = row['ROI']
            if 'NaN' in roi:
                # TODO: i don't love this conditional. is it safe??
                continue
            hex_color = rgba_to_hex(row['c_r'],row['c_g'],row['c_b'],row['c_a'])
            roi_dict[roi] = hex_color
    return roi_dict


def load_segment_colors(label_type: str="RCCF"):
    """returns the dict of label ROI number -> hex color stored in data/segment_colors/{label_type}.json"""
    code_dir = os.path.dirname(os.path.realpath(__file__))
    with open(pathjoin(code_dir, "data", "segment_colors", "{}.json".format(label_type))) as f:
        return json.load(f)


def make_empty_transform():
    """returns a neuroglancer source transform for update_dimensions and update_matrix_transform to fill in"""
    return {
        "matrix": [[0, 0, 0, 0] for _ in range(3)],
        "inputDimensions": {axis: [0, "m"] for axis in "xyz"},
        "outputDimensions": {axis: [0, "m"] for axis in "xyz"},
    }


def make_scene(layers: list, voxel_sizes: list, cross_section_scale: float):
    """returns the dict for a full neuroglancer scene. layers[0] must be the image layer, which is what opens in the side panel"""
    return {
        "dimensions": {axis: [size, "m"] for axis, size in zip("xyz", voxel_sizes)},
        "position": [0, 0, 0],
        "crossSectionScale": cross_section_scale,
        "projectionScale": 2048,
        "layers": layers,
        "selectedLayer": {"size": 677, "visible": True, "layer": layers[0]["name"]},
        "layout": "4panel",
    }


def single_image_scene(data_file: str, label_precomputed_file: str, data_nhdr: dict, label_nhdr: dict, n5_dir: str, is_stereotaxic=True, overlay_layers=[], color_overlay: str="white"):
    """Function to build the scene for our most typical use case: one image volume and one labelset
        inputs:
        data_file -- name of the root folder of data file, as it sits on S3
        label_precomputed_file -- name of the root folder of label file, as it sits on S3
        data_nhdr -- path to nhdr file for the data. necessary to pull metadata from
        label_nhdr -- analogous to data_nhdr for label_precomputed_file
        n5_dir -- local folder holding data_file, used to compute the default display range
        is_stereotaxic -- boolean whether to offset the volume to be centered at Bregma
        overlay_layers -- optional list of lookup table file paths to color the labels with

        returns a dict that represents the neuroglancer scene. writing it is left to the caller


        assumptions:
        - the data is stored on the d3mof5o AWS S3 bucket and follows all freshscreen organization conventions
        - file (actually a directory) ends in .precomputed or .n5. will infer which one it is based on the extension

        - layer[0] is image volume
        - layer[1] is segmentation volume
        - any statistical overlay layers come after that"""

    """function to write json for our most typical use case: two layers
        1) the contrast of interest
            different for each contrast
            need to update:
                filepath
                contrast window/range
                transform matrix
                input/output dimensions (out_dims should always be MRI resolution)
                name
                opacity -- ?
                shader - volume only
                shader controls - volume only
        2) labelset
            one for each SPECIMEN. should be the same parameters for all (except origin transform)"""
    _,data_file = os.path.split(data_file)

    # these are dicts representing the local data
    """data_nhdr is an ORDERED DICT -- keys are identical to those in the nhdr file
        space directions = transform matrix (with embedded voxel size in mm)
        space origin = origin translation in mm
        type = data type
        dimension = 3 (or more for COLOR)
        space = "left-posterior-superior" -- will give us hints on where to throw negatives
        sizes = array of num voxels in each dimension
        kinds = "domain domain domain" (different for COLOR)
        endian = big or little (usually little)
        encoding = raw or gzip
        data file = not used here -- could start EVERYTHING from the nhdr, including naming the neuroglancer file...? hmmm"""

    # i calculate these also in setup_one_image_layer, but i also need them for other things...
    data_voxel_sizes = get_voxel_size_from_nhdr_dict(data_nhdr)
    label_voxel_sizes = get_voxel_size_from_nhdr_dict(label_nhdr)

    ###**************####
    # create image layer
    ###**************####"
    image_layer = setup_one_image_layer(data_file, data_nhdr, label_nhdr, is_stereotaxic=is_stereotaxic, color_overlay=color_overlay, n5file_dir=n5_dir)

    ###**************####
    # create rCCF label layer
    ###**************####
    # again for labels we do NOT want a single color overlay, they are being colored by RCCF colors (or by statistical overlay)
    rccf_label_layer = setup_rccf_label_layer(label_precomputed_file, label_nhdr, is_stereotaxic=is_stereotaxic)
    layers = [image_layer, rccf_label_layer]

    ###**************####
    # create statistical overlay layers
    ###**************####
    # take in a list of lookup tables
    # for each, copy RCCF labels layer into a new layer
    # update the lookup table
    # will use the same label precomputed file for all
    for lut_file in overlay_layers:
        lut_dict = convert_lut_file(lut_file)
        new_layer = rccf_label_layer.copy()
        # update the lut
        new_layer["name"] = os.path.basename(lut_file).split(".")[0]
        new_layer["segmentColors"] = lut_dict
        new_layer["visible"] = False
        new_layer["opacity"] = 0.5
        layers.append(new_layer)

    data = make_scene(layers, label_voxel_sizes, get_crossSectionScale_factor(data_voxel_sizes[0]))

    logging.debug("final form of data dict: \n{}".format(data))
    return data


# TODO: data_threshold_max by default currently looks at a dict of deafult values. eventually, do NOT allow this. force user to pass a decent value to this function
def write_grayscale_json(data_file: str, label_precomputed_file: str, data_nhdr: dict, label_nhdr: dict, output_file: str, data_threshold_max=None, is_stereotaxic=True, overlay_layers=[], color_overlay: str="white"):
    """builds the scene for one image volume and writes it to output_file as a freshscreen display json

    the n5 file is expected to be in the same folder as output_file. data_threshold_max is not used"""
    _,data_file = os.path.split(data_file)
    scene = single_image_scene(data_file, label_precomputed_file, data_nhdr, label_nhdr, os.path.dirname(output_file), is_stereotaxic=is_stereotaxic, overlay_layers=overlay_layers, color_overlay=color_overlay)
    write_freshscreen_display_json(scene, data_file, output_file)


def get_crossSectionScale_factor(voxel_size):
    # crossSectionScale is a "global" variable to each neuroglancer scene
    # determines how zoomed in it is
        # found 2 works well for 15um data
        # 0.6 works well for 45um
        # remember that neuroglancer assumes nanometers, so that is why the vox size comparison is so wild
    eps=1e-15
    scale_factor = 2
    if abs(abs(voxel_size) - 0.000015) < eps:
        scale_factor = 2
    if abs(abs(voxel_size) - 0.000045) < eps:
        scale_factor = 0.8
    logging.info("setting scale factor to: {}".format(scale_factor))
    return scale_factor


def update_dimensions(data_layer: dict, voxel_sizes: list, dimension_field: str, job_id: str=""):
    if dimension_field not in ["inputDimensions", "outputDimensions"]:
        logging.warning("unrecognized dimension field: {}\nCan not write. Only allowed options are: 'inputDimensions' and 'outputDimensions'".format(dimension_field))
        return

    # set inputDimensions to nhdr_voxel_size / 1000 (nhdr is in mm, json is in m)
    i = 0
    for key in data_layer["source"]["transform"][dimension_field].keys():
        data_layer["source"]["transform"][dimension_field][key][0] = voxel_sizes[i]
        i+=1
    logging.info("updated {} {} to: {}".format(job_id, dimension_field, data_layer["source"]["transform"][dimension_field]))


def setup_rccf_label_layer(label_precomputed_file: str, label_nhdr: str, is_stereotaxic=True):
    rccf_label_layer = {
        "type": "segmentation",
        "source": {
            "url": get_s3_url_from_file_basename(label_precomputed_file),
            "transform": make_empty_transform(),
            "subsources": {"default": True, "bounds": False, "properties": True, "mesh": True},
            "enableDefaultSubsources": False,
        },
        "tab": "source",
        "selectedAlpha": 0.6,
        "notSelectedAlpha": 0.1,
        "ignoreNullVisibleSet": False,
        "name": "{} {}".format(get_runno_from_filename(label_precomputed_file), get_contrast_from_filename(label_precomputed_file)),
        "segmentColors": load_segment_colors("RCCF"),
    }

    label_voxel_sizes = get_voxel_size_from_nhdr_dict(label_nhdr)

    update_dimensions(rccf_label_layer, label_voxel_sizes, "inputDimensions", job_id="RCCF label layer")
    update_dimensions(rccf_label_layer, label_voxel_sizes, "outputDimensions", job_id="RCCF label layer")

    update_matrix_transform(rccf_label_layer["source"]["transform"]["matrix"], label_nhdr, label_voxel_sizes, is_stereotaxic)
    flip_xform_dimension(rccf_label_layer["source"]["transform"]["matrix"], 2)
    return rccf_label_layer


def setup_one_image_layer(data_file: str, data_nhdr: dict, label_nhdr: dict, n5_suffix: str=N5_SUFFIX, is_stereotaxic=True, color_overlay: str="white", n5file_dir=None):
    """sets up one data layer in a neuroglancer scene
    call this one for a typical scene with one volume
    called three times for color images (1x per channel)
        color_overlay should be None for RGB color images (like TDI_color) as it is only meant for coloring lightsheet images
    """
    image_layer = {
        "type": "image",
        "source": {
            "url": get_s3_url_from_file_basename(data_file, n5_suffix),
            "transform": make_empty_transform(),
        },
        "tab": "rendering",
        "opacity": 1,
    }
    local_n5_file = pathjoin(n5file_dir, data_file)

    # edit the layer name
    # TODO: this solution will probably only work once, for DMBA, today (april 3 2025)
    if is_stereotaxic and "DMBA-LS" in data_file:  # use this as an indicator we are working with a DMBA atlas dataset
        # this is to solve an issue where lsfm datasets are getting named "DMBA-LS CD31" for example, omitting the individual's specimen ID number
        # ONLY RUN ON LSFM DATASETS
        spec_id = data_file.split("_")[1]
        image_layer["name"] = "{} {} {}".format(get_runno_from_filename(data_file), spec_id, get_contrast_from_filename(data_file))
    else:
        image_layer["name"] = "{} {}".format(get_runno_from_filename(data_file), get_contrast_from_filename(data_file))

    data_voxel_sizes = get_voxel_size_from_nhdr_dict(data_nhdr)
    label_voxel_sizes = get_voxel_size_from_nhdr_dict(label_nhdr)
    update_dimensions(image_layer, data_voxel_sizes, "inputDimensions", job_id="image layer")
    update_dimensions(image_layer, label_voxel_sizes, "outputDimensions", job_id="image layer")

    # update shader controls (default window and level)
    # range is range of data to be rendered on screen
    # window is range of the slider the user can use to edit the range in the GUI
    # data file is n5 file basename
    # assume it exists within freshscreen library -- the output dir (where we will write the output json)
    # TODO: i should also look at the s3 bucket to find the data. maybe i am updating a file that has already been deleted off the local bucket
    # useful for updating/reprocessing old data which may come to be necessary
    image_layer["shaderControls"] = {"normalized": compute_display_range(local_n5_file)}


    if color_overlay is None:
        color_overlay = get_default_color_overlay(data_file)
    if "color" not in data_file:
        image_layer["shader"] = (
            "#uicontrol invlerp normalized(range=[0, 0.8], clamp=true)\n"
            "#uicontrol float log2_gamma slider(min=-2, max=2, default=0, step=0.05)\n"
            "#uicontrol bool invert checkbox(default=false)\n"
            f"#uicontrol vec3 color color(default=\"{color_overlay}\")\n"
            "#uicontrol bool transparent_bg checkbox(default=false)\n\n"
            "void main() {\n"
            "  float x = clamp(normalized(), 0.0, 1.0);\n"
            "  if (invert) x = 1.0 - x;\n"
            "  x = pow(x, exp2(-log2_gamma));   // log2_gamma > 0 brightens, < 0 darkens\n"
            "  if (transparent_bg) {\n"
            "    emitRGBA(vec4(color * x, x));\n"
            "  } else {\n"
            "    emitRGB(color * x);\n"
            "  }\n"
            "}"
        )
    else:
        image_layer["shader"] = (
    """
    #uicontrol float black slider(min=0, max=1, default=0)
    #uicontrol float white slider(min=0, max=1, default=1)
    #uicontrol float gamma slider(min=0.2, max=3, default=1)
    #uicontrol float saturation slider(min=0, max=3, default=1)
    #uicontrol float opacity slider(min=0, max=1, default=1)
    void main() {
    vec3 rgb = vec3(toNormalized(getDataValue(0)),
                    toNormalized(getDataValue(1)),
                    toNormalized(getDataValue(2)));
    rgb = clamp((rgb - black) / max(white - black, 1e-6), 0.0, 1.0);
    rgb = pow(rgb, vec3(1.0 / gamma));
    float lum = dot(rgb, vec3(0.2126, 0.7152, 0.0722));
    rgb = clamp(mix(vec3(lum), rgb, saturation), 0.0, 1.0);
    emitRGBA(vec4(rgb, opacity));
    }"""
        )
    update_matrix_transform(image_layer["source"]["transform"]["matrix"], data_nhdr, label_voxel_sizes, is_stereotaxic)
    flip_xform_dimension(image_layer["source"]["transform"]["matrix"], 2)

    return image_layer


# also take in the data_file name because it is available to pass, and it is "obfuscated" within the data dict by the N5_PREFIX/PRECOMPUTED_PREFIX
def write_freshscreen_display_json(data: dict, data_file: str, output_file: str, display_name: str=None):
    """given the dict created earlier, generate the json file for freshscreen
    three keys:
        - filename in freshscreen s3
        - display name
        - neuroglancer url

    create a dict, convert to json, write to file"""
    organization_json = dict()
    organization_json["object_Name"] = data_file
    # if user does not specify, then look to see what I named this layer in neuroglancer and use that
    if display_name is None:
        organization_json["object_displayName"] = data["layers"][0]["name"]
    else:
        organization_json["object_displayName"] = display_name

    organization_json["view_neuroglancerURL"] = convert_json_to_url(convert_dict_to_string(data))
    write_dict_to_json_file(output_file, organization_json)


def pathjoin(*args):
    """takes a list of path components and joins them together with forward slashes as the separator
    this is easier than doing it inline because using *args automatically turns all of your input arguments into a single tuple (so you don't have to wrap your arguments to "/".join() into a list)"""
    return "/".join(args)


def find_nhdr_file(n5_file: str):
    """takes as input an n5 file found within s3 (as we are looping over them). tries to find the corresponding nhdr file on s3
        if it cannot find one on s3, then try to find one locally, with the optional argument nhdr_dir
        returns a pynrrd dict"""
    code_dir = os.path.dirname(os.path.realpath(__file__)).replace("\\","/")
    nhdr_file_name = n5_file.replace(".n5", "_M4D.nhdr")
    if "color" in nhdr_file_name:
        # then search my s3 file list for the dwi, and use that
        # BUT if your n5 is a tdix3 or x5, then you need to divide your voxel sizes
        cmd_str="grep dwi {}/temp/subprocess_output".format(code_dir)
        a=subprocess.run(cmd_str, capture_output=True)
        nhdr_file_name = a.stdout.decode("utf-8")
    out_dir = "{}/data/nhdr_cache".format(code_dir)
    cmd_str="{} copy freshscreen:d3mof5o/{} {}/".format(RCLONE, nhdr_file_name, out_dir)
    subprocess.run(cmd_str)

    # if you fail to find the nhdr on s3, then what?
    # here is how i was finding things locally. maybe do that. maybe just fail and yell at user
#        data_nhdr_file = glob.glob(pathjoin(nhdr_dir, "*{}*_{}.nhdr".format(runno, contrast)))
#        if len(data_nhdr_file) == 0:
#            # then maybe the files do not have a runno in the name (light lightsheet). search for spec_id instead
#            data_nhdr_file = glob.glob(pathjoin(nhdr_dir, "*{}*_{}*.nhdr".format(spec_id, contrast)))
#        if len(data_nhdr_file) != 1:
#            logging.error("found zero or multiple nhdr files for {} {}. do not know what to do.\n\t{}".format(spec_id, contrast, data_nhdr_file))
#            continue
#            #exit()
#        # WARNING: glob still throws backslashes in...
#            # this is because under the hood it still uses os.path.join() and os.path.sep WILL end up in your path outputted by glob
#        data_nhdr_file = data_nhdr_file[0].replace("\\","/")

    nhdr = nrrd.read_header("{}/{}".format(out_dir, nhdr_file_name))
    return nhdr


def find_nhdr_file(runno: str, contrast: str="dwi", local_search_dir: str=None):
    """search for an nhdr file matching both runno and contrast in the schema *${runno}*${contrast}*.nhdr VERY BROAD INCLUSIVE SEARCH (technically, order does not even matter, just runs grep callls in series)

    runno=a specimen identifier. Can be runno-like (N12345), specimen-id-lik (200316-1_1) or atlas-id-like (DMBA)
    contrast=an image contrast. by default, will return the DWI (a good default template for non-contrast-specific tasks)
    If provided, will first search through local_search_dir, then it will search within the Freshscreen S3 bucket. returns None if it fails to find any matching nhdr files
    """
    if local_search_dir is not None:
        data_nhdr_file = glob.glob(pathjoin(local_search_dir, "*{}*_{}*.nhdr".format(runno, contrast)))
        if len(data_nhdr_file) > 0:
            return data_nhdr_file[0]
        # also check one folder up, in case files were reorganized and the dwi is actually hiding one folder up
        data_nhdr_file = glob.glob(pathjoin(local_search_dir, "/../*{}*_{}*.nhdr".format(runno, contrast)))
        if len(data_nhdr_file) > 0:
            return data_nhdr_file[0]


    # ACTUALLY, second guess this. We will need more thought to read a nrrd using pynrrd FROM s3
    # all it actually takes is a subprocess.run call, using "rclone cat"
    # search s3
    # this next line is the only difference (lsf vs lsd) in searching for heirarchial vs nhdr files
    cmd_str="{} lsf freshscreen:d3mof5o".format(RCLONE)
    a=subprocess.run(cmd_str, shell=True, capture_output=True)
    temp_file_list = "{}/data/temp/subprocess_output_lsf".format(os.path.dirname(os.path.realpath(__file__)).replace("\\","/"))
    with open(temp_file_list, "w+b") as f:
        f.write(a.stdout)


    # filter for desired nhdr files only
    cmd_str = "grep {} {}".format(".nhdr", temp_file_list)
    a=subprocess.run(cmd_str, capture_output=True)
    with open(temp_file_list, "w+b") as f:
        f.write(a.stdout)

    # filter for desired runno
    cmd_str = "grep {} {}".format(runno, temp_file_list)
    a=subprocess.run(cmd_str, capture_output=True)
    with open(temp_file_list, "w+b") as f:
        f.write(a.stdout)

    # filter for desired contrast only
    cmd_str = "grep {} {}".format(contrast, temp_file_list)
    a=subprocess.run(cmd_str, capture_output=True)
    with open(temp_file_list, "w+b") as f:
        f.write(a.stdout)

    filelist = a.stdout.decode("utf-8").split("\n")
    # hopefully the first one in this list is good. (if everything works as expected, it will have length one)
    return filelist[0]


# !!! this one ONLY gets heirarchial data from freshscreen.
# lsd only gets folders. this will only get .n5 and .precomputed datasets
# will not edit right now, will handle nhdr find code solely in the find_nhdr_file function
def get_file_list_from_freshscreen(spec_id_fresh: str, contrast_list: list=[]):
    # on windows i had to split this up because pipes are playing funny (read: not working). do not fully understand why.
    # TODO: test this doesn't break everything on mac
    # i think that i need to do with open... several times because i am usig the command line to read from the file (with grep and awk). python having the file open locks it from other prying eyes. there MIGHT(??) be a work around to this
    # this is the full command as you would run it on the command line
    # TODO: also, subprocess.run takes stdin as an argument. i am supposed to be able to pass a file handle to it, but that isn't working. that might be mroe reliable then what i am currently doing+
    #cmd_str="{} lsd freshscreen:d3mof5o | grep {} | awk '{{print $5}}'".format(RCLONE, spec_id_fresh)
    cmd_str="{} lsd freshscreen:d3mof5o".format(RCLONE)
    a=subprocess.run(cmd_str, shell=True, capture_output=True)
    temp_file_list = "{}/data/temp/subprocess_output".format(os.path.dirname(os.path.realpath(__file__)).replace("\\","/"))
    with open(temp_file_list, "w+b") as f:
        f.write(a.stdout)

    # filter for desired specimen_id (DMBA is a spec id)
    cmd_str = "grep {} {}".format(spec_id_fresh, temp_file_list)
    a=subprocess.run(cmd_str, capture_output=True)
    with open(temp_file_list, "w+b") as f:
        f.write(a.stdout)

    # "parallel" grep calls on the spec_id_fresh filtered filelist. calling grep in series act as an AND filter
    if len(contrast_list) > 0:
        # ALWAYS want to have the label file in the returned list because that is used in any case. if we are filtering, add label
        contrast_list.append("label")
        captured_output = []
        for contrast in contrast_list:
            cmd_str = "grep {} {}".format(contrast, temp_file_list)
            a=subprocess.run(cmd_str, capture_output=True)
            captured_output.append(a)
        with open(temp_file_list, "w+b") as f:
            for a in captured_output:
                f.write(a.stdout)

    cmd_str = "awk '{{print $5}}' {}".format(temp_file_list)
    a=subprocess.run(cmd_str, capture_output=True)
    with open(temp_file_list, "w+b") as f:
        f.write(a.stdout)

    filelist = a.stdout.decode("utf-8").split("\n")
    return filelist


def get_voxel_size_from_color_nhdr(color_nhdr_file):
    """voxel size (mm) of the 3 spatial axes, read straight from the header text.
    pynrrd can't read LIST color headers, and the 'none' channel axis can be first or last."""
    with open(color_nhdr_file) as f:
        for line in f:
            if line.startswith('space directions:'):
                vecs = re.findall(r'\(([^)]*)\)', line)
                return [max(abs(float(v)) for v in vec.split(',')) for vec in vecs]
    raise ValueError("no space directions in {}".format(color_nhdr_file))


def process_one_image(filename: str, output_dir: str, data_nhdr_file: str, label_nhdr_file: str, label_precomputed_file: str=None, is_stereotaxic=True, overlay_layers=[], color_overlay: str="white"):
    """Setup Neuroglancer scene and write the Freshscreen display JSON for one n5 file

    filename = n5 filename as found on AWS S3. This file must already be uploaded to S3 to work
    output_dir = directory to write the resulting json file
    data_nhdr_file = INPUT data nhdr file with CIVM naming
    label_nhdr_file = INPUT label nhdr file with CIVM naming
    label_precomputed_file = filename of precomputed label file as found on AWS S3. This file must already be uploaded to s3 to work.
    """
    # these are the types of files we DO NOT want to process here
    # labels are included here because they will never be presented on their own, only as a layer with other files
    if "label" in filename.lower():
        return
    if "gre" in filename.lower():
        return
    if not filename or filename is None:
        return
    runno = get_runno_from_filename(filename)
    contrast = get_contrast_from_filename(filename)

    logging.info("data nhdr file = {}".format(data_nhdr_file))
    # stash the original name away just in case it is a color, will be needed later. will not be used in non-color case
    color_nhdr_file = data_nhdr_file
    if "color" in data_nhdr_file:
        logging.info("processing COLOR file. Will use the DWI nhdr file for metadata purposes. ")
        search_dir = os.path.dirname(data_nhdr_file)
        data_nhdr_file = find_nhdr_file(runno, "dwi", local_search_dir=search_dir)
    #breakpoint()
    data_nhdr = nrrd.read_header(data_nhdr_file)
    if "color" in filename.lower():
        color_vox = np.array(get_voxel_size_from_color_nhdr(color_nhdr_file))
        sd = data_nhdr["space directions"]
        data_nhdr["space directions"] = sd / np.abs(sd).max(axis=1, keepdims=True) * color_vox[:, None]
    # TODO: find label nhdr on freshscreen
        # do we ever upload label nhdr to freshscreen? how do we distribute labels to people?
    label_nhdr = nrrd.read_header(label_nhdr_file)

    output_file = pathjoin(output_dir, "{}.json".format(filename))
    #if os.getlogin() == 'hmm56':
    #    output_file = pathjoin(output_dir, 'hmm56_reprocessed_json', "{}.json".format(filename))
    #logging.info("\n\nRUNNING FOR SPECIMEN:\n\t filename = {}\n\t data_nhdr_file = {}\n\t label_precomputed_file = {}\n\t label_nhdr_file = {}\n\t output_file = {}\n\n".format(filename, data_nhdr_file, label_precomputed_file, label_nhdr_file, output_file))

    # check if label_precomputed_file is an absolute path or just a filename
    # sometimes it is passed to us as B:/Freshscreen_library/xyzx_RCCF_labels.precomputed
    # but we want to find it in s3, so all we need is the basename xyzx_RCCF_labels.precomputed
    # confirmed that os.path.basename does nothing when given only "xyzx_RCCF_labels.precomputed", so no checks are necessary
    print(label_precomputed_file)
    label_precomputed_file = os.path.basename(label_precomputed_file)

    logging.info("PROCESSING FOLLOWING JOB:")
    logging.info("\tfilename = {}".format(filename))
    logging.info("\tlabel_precomputed_file = {}".format(label_precomputed_file))
    logging.info("\tdata nhdr = {}".format(data_nhdr))
    logging.info("\tlabel nhdr = {}".format(label_nhdr))
    logging.info("\toutput_file = {}".format(output_file))
    write_grayscale_json(filename, label_precomputed_file, data_nhdr, label_nhdr, output_file, is_stereotaxic=is_stereotaxic, overlay_layers=overlay_layers, color_overlay=color_overlay)


def loop_through_specimen_in_freshscreen(spec_id: str, output_dir: str, nhdr_dir: str, label_precomputed_file: str=None, contrast_list=[], is_stereotaxic=True):
    """loops through all n5 or precomputed files in s3 connected to the provided specimen id.  Skips over color files"""

    spec_id_fresh = spec_id.replace("_", "-") # follows freshscreen specifications i.e. no underscores allowed
    filelist = get_file_list_from_freshscreen(spec_id_fresh, contrast_list)
    print(filelist)

    if label_precomputed_file is None:
        # if user does not provide label file, loop through s3 bucket to try and find it. returns if cannot find
        for f in filelist:
            if "label" in f.lower():
                label_precomputed_file = f
                break
        if label_precomputed_file is None:
            logging.warning("cannot find a label file in s3 for specimen {}".format(spec_id_fresh))
            return None
    for filename in filelist:
        runno = get_runno_from_filename(filename)
        contrast = get_contrast_from_filename(filename)
        color_overlay = get_default_color_overlay(filename)
        print("contrast == {}".format(contrast))
        if "color" in filename.lower():
            # then we take the dwi nhdr and use that
            # the transform matrix will still be valid but voxel sizes NOT NECESSARILY
            # later we read vox size directly from the color nhdr
            # this is done because the pynrrd library fails to read in color nhdr files with channels split into separate raw files
            # TODO: finish improving James' nrrd reader replacement and universally use that one
            contrast="dwi"
        data_nhdr_file = glob.glob(pathjoin(nhdr_dir, "*{}*_{}.nhdr".format(runno, contrast)))
        if len(data_nhdr_file) == 0:
            # then maybe the files do not have a runno in the name (light lightsheet). search for spec_id instead
            # 2/23/24 removed the * between contrast and .nhdr; was too generous and picks up things like DMBA_dwi_bregma_update_test.nhdr
            data_nhdr_file = glob.glob(pathjoin(nhdr_dir, "*{}*_{}.nhdr".format(spec_id, contrast)))
        if len(data_nhdr_file) != 1:
            logging.error("found zero or multiple nhdr files for {} {}. do not know what to do.\n\t{}".format(spec_id, contrast, data_nhdr_file))
            return
        # WARNING: glob still throws backslashes in...
            # this is because under the hood it still uses os.path.join() and os.path.sep WILL end up in your path outputted by glob
        data_nhdr_file = data_nhdr_file[0].replace("\\","/")
        if not os.path.isfile(data_nhdr_file):
            logging.error("cannot find data nhdr locally: {}".format(data_nhdr_file))
            exit()
        label_nhdr_file = glob.glob(pathjoin(nhdr_dir,"labels","*", "*{}*label*.nhdr".format(runno)))
        if len(label_nhdr_file) == 0:
            logging.error("cannot find a relevant label nhdr file for {}.".format(spec_id))
            exit()
        label_nhdr_file = label_nhdr_file[0].replace("\\","/")
        if not os.path.isfile(label_nhdr_file):
            logging.error
        process_one_image(filename, output_dir, data_nhdr_file, label_nhdr_file, label_precomputed_file, is_stereotaxic=is_stereotaxic,color_overlay=color_overlay)


#******!*!*!*!*!*!*!*!*!*!*!*!!*****************#
# MAIN with command line arguments
#******!*!*!*!*!*!*!*!*!*!*!*!!*****************#
# WARNING: main is not currently used in the pipeline. process_one_image is called straight from $WORKSTATION_CODE/shared/Freshscreen_codes/prepare1image.py (harry)
# the real-data test that used to live here is now tests/test_scenes.py
def main():
    is_stereotaxic=True
    #Set variable to None if you want to search S3
    label_precomputed_file="DMBA_RCCF_labels.precomputed"
    #label_precomputed_file=None
    logging.getLogger().setLevel(logging.INFO)
    if len(sys.argv) > 2:
        # i am not even using project_code
        project_code = sys.argv[1]
        specimen_id = sys.argv[2]
        if len(sys.argv) > 4:
            # then I do not want to use the typical R drive organization, instead pass an nhdr_dir(input) and out_dir
            nhdr_dir = sys.argv[3]
            output_dir = sys.argv[4]
        else:
            nhdr_dir = "Q:/{}/{}/Aligned-Data".format(project_code, specimen_id)
            output_dir = "S:/freshscreen_library/json_display_files/{}".format(specimen_id)
        # TODO: smartly determine what system we are on. windows or mac? -- should always be citrix really
        contrast_list=["dwi","gqi-color"]
        loop_through_specimen_in_freshscreen(specimen_id, output_dir, nhdr_dir, contrast_list=contrast_list, is_stereotaxic=is_stereotaxic, label_precomputed_file=label_precomputed_file)
    else:
        logging.warning("Not enough input arguments. Requires project_code, specimen_id, and nhdr_dir as positional arguments")


if __name__ == "__main__":
  main()
