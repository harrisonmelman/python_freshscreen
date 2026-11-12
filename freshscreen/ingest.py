import glob
import logging
import os
import subprocess
import nrrd
import numpy as np

from freshscreen.config import RCLONE, TEMP_DIR
from freshscreen.naming import get_age_from_group, pathjoin, get_runno_from_filename, get_contrast_from_filename, get_s3_url_from_file_basename
from freshscreen.geometry import get_voxel_size_from_color_nhdr, get_voxel_size_from_nhdr_dict
from freshscreen.display_range import compute_display_range
from freshscreen.dataset import Dataset
from freshscreen.contrasts import get_default_color_overlay


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
    temp_file_list = TEMP_DIR / "subprocess_output_lsf"
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



# function to do all of the reading work so we don't always need to open files
# User should run this before 
# Or it will be ran the first time, assuming the main code branch does not find a dataset.json file for this spceimen
def dataset_from_files(n5_path: str, data_nhdr_file: str) -> Dataset:
    logging.info("data nhdr file = {}".format(data_nhdr_file))
    # TODO: first we check to see if the spec file already exists
    #  

    # stash the original name away just in case it is a color, will be needed later. will not be used in non-color case
    n5_filename = os.path.basename(n5_path)
    runno = get_runno_from_filename(n5_filename)
    contrast = get_contrast_from_filename(n5_filename)
    url = get_s3_url_from_file_basename(n5_filename)
    display_range = compute_display_range(n5_path)
    default_color = get_default_color_overlay(n5_filename)

    # look at the n5 name instead of nhdr name just incase DWI gets sent here for metadata purposes
    kind = "color" if "color" in n5_filename.lower() else "image"
    color_nhdr_file = data_nhdr_file if "color" in os.path.basename(data_nhdr_file) else None

    if color_nhdr_file:
        # this is because pynrrd can't read nhdr files with LIST data paths
        logging.info("processing COLOR file. Will use the DWI nhdr file for metadata purposes. ")
        search_dir = os.path.dirname(data_nhdr_file)
        data_nhdr_file = find_nhdr_file(runno, "dwi", local_search_dir=search_dir)
    data_nhdr = nrrd.read_header(data_nhdr_file)

    # if color nhdr file does not exist, then we were passed a dwi nhdr for metadata
    # and we must (nothing else we can do) assume that this dataset has same resolution as the dwi
    # so no need to update space directions
    if kind == "color" and color_nhdr_file is not None:
        # this is done because if we used the DWI for nhdr metadata, there is a chance the voxel sizes are wrong
        # gqi color and tdi color should be fine, but supersampled datasets like tdi5um color would fail 
        color_vox = np.array(get_voxel_size_from_color_nhdr(color_nhdr_file))
        sd = data_nhdr["space directions"]
        data_nhdr["space directions"] = sd / np.abs(sd).max(axis=1, keepdims=True) * color_vox[:, None]
    voxel_size = get_voxel_size_from_nhdr_dict(data_nhdr)

    # for comparative scenes, we want to keep track of age
    # this is kind of specific to CHDI, but it only comes up when you have dashes in your runno
    # should be harmless in irrelevant projects? time will tell
    age = get_age_from_group(runno)

    # type conversions frmo numpy
    voxel_size = [float(x) for x in voxel_size]
    space_directions = data_nhdr["space directions"].tolist()
    space_origin = data_nhdr["space origin"].tolist()
    url = str(url)

    # before conversion these were voxel_size was ndarray[Unknown] and url was Path
    # it needed to be ndarray above to do the calculations in the if_color branch
    ds = Dataset(n5_filename, runno, contrast, kind, url, voxel_size, space_directions,\
                  space_origin, default_color, display_range, age)
    return ds
    


# takes in the label_nhdr and the precomputed filepath
# creates a dataset object that has the spatial metadata
# it also has the precomputed filename, much like how i should keep log of the n5 filename
# exact same concept, just less work needs to be done in the labels instance
# calling it labelset to match the above naming
def labelset_from_files(label_nhdr_file, precomputed_path):
    # TODO: find label nhdr on freshscreen
        # do we ever upload label nhdr to freshscreen? how do we distribute labels to people?
    label_nhdr = nrrd.read_header(label_nhdr_file)
    # in case this is passed to us as an absolute global path, we only need the basename
    precomputed_path = os.path.basename(precomputed_path)
    contrast = get_contrast_from_filename(precomputed_path)
    runno = get_runno_from_filename(precomputed_path)
    url = get_s3_url_from_file_basename(precomputed_path)
    voxel_size = get_voxel_size_from_nhdr_dict(label_nhdr)
    kind = "labels"
    age = get_age_from_group(runno)

    # type conversions
    space_directions = label_nhdr["space directions"].tolist()
    space_origin = label_nhdr["space origin"].tolist()
    url = str(url)
    voxel_size = [float(x) for x in voxel_size]

    ds = Dataset(precomputed_path, runno, contrast, kind, url, voxel_size,\
                 space_directions, space_origin, age=age)
    return ds
