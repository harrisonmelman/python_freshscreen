"""Scene building and S3 lookups that have not moved into the freshscreen package yet.

The helpers this module used to define now live in freshscreen/ and are re-exported
here, so existing callers (prepare1image.py, make_displayJSON.py, the tests) keep working.
"""
import glob
import logging
import os
import subprocess
import sys

import nrrd
import numpy as np

from freshscreen.config import (  # noqa: F401  (re-exported)
    LIGHTSHEET_CONTRASTS, N5_PREFIX, N5_SUFFIX, PRECOMPUTED_PREFIX, RCLONE,
)
from freshscreen.contrasts import get_default_color_overlay, get_default_threshold  # noqa: F401
from freshscreen.display_range import _pick_array, compute_display_range  # noqa: F401
from freshscreen.geometry import (  # noqa: F401
    flip_xform_dimension, get_crossSectionScale_factor, get_voxel_size_from_color_nhdr,
    get_voxel_size_from_nhdr_dict, make_empty_transform, update_dimensions, update_matrix_transform,
)
from freshscreen.luts import convert_lut_file, load_segment_colors, rgba_to_hex  # noqa: F401
from freshscreen.naming import (  # noqa: F401
    get_contrast_from_filename, get_runno_from_filename, get_s3_url_from_file_basename, pathjoin,
)
from freshscreen.output import (  # noqa: F401
    NpEncoder, convert_dict_to_string, convert_json_to_url, write_dict_to_json_file,
    write_freshscreen_display_json,
)
from freshscreen.shaders import RGB_SHADER, grayscale_shader


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


def write_grayscale_json(n5_filepath: str, label_precomputed_file: str, data_nhdr: dict, label_nhdr: dict, output_file: str, is_stereotaxic=True, overlay_layers=[], color_overlay: str="white"):
    """builds the scene for one image volume and writes it to output_file as a freshscreen display json

    the n5 file is expected to be in the same folder as output_file."""
    n5_dir,n5_filename = os.path.split(n5_filepath)
    scene = single_image_scene(n5_filename, label_precomputed_file, data_nhdr, label_nhdr, n5_dir, is_stereotaxic=is_stereotaxic, overlay_layers=overlay_layers, color_overlay=color_overlay)
    write_freshscreen_display_json(scene, n5_filename, output_file)


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
    flip_xform_dimension(rccf_label_layer["source"]["transform"]["matrix"])
    return rccf_label_layer


def setup_one_image_layer(data_file: str, data_nhdr: dict, label_nhdr: dict, is_stereotaxic=True, color_overlay: str="white", n5file_dir=None):
    """sets up one data layer in a neuroglancer scene
    call this one for a typical scene with one volume
    called three times for color images (1x per channel)
        color_overlay should be None for RGB color images (like TDI_color) as it is only meant for coloring lightsheet images
    """
    image_layer = {
        "type": "image",
        "source": {
            "url": get_s3_url_from_file_basename(data_file),
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
    display_range = compute_display_range(local_n5_file)

    if color_overlay is None:
        color_overlay = get_default_color_overlay(data_file)
    if "color" not in data_file:
        image_layer["shaderControls"] = {"normalized": display_range}
        image_layer["shader"] = grayscale_shader(color_overlay)
    else:
        image_layer["shaderControls"] = {"black": display_range["range"][0]/255}
        image_layer["shaderControls"] = {"white": display_range["range"][1]/255}
        image_layer["shader"] = RGB_SHADER
    update_matrix_transform(image_layer["source"]["transform"]["matrix"], data_nhdr, label_voxel_sizes, is_stereotaxic)
    flip_xform_dimension(image_layer["source"]["transform"]["matrix"])

    return image_layer


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


def process_one_image(n5_filepath: str, output_dir: str, data_nhdr_file: str, label_nhdr_file: str, label_precomputed_file: str=None, is_stereotaxic=True, overlay_layers=[], color_overlay: str="white"):
    """Setup Neuroglancer scene and write the Freshscreen display JSON for one n5 file

    n5_file = full file path to the n5 file in local storage (within B:/freshscreen_library probably)
    output_dir = directory to write the resulting json file
    data_nhdr_file = data nhdr file with freshscreen naming
    label_nhdr_file = label nhdr file with freshscreen naming
    label_precomputed_file = filename of precomputed label file as found on AWS S3. This file must already be uploaded to s3 to work.
    """
    # these are the types of files we DO NOT want to process here
    # labels are included here because they will never be presented on their own, only as a layer with other files
    if not n5_filepath or n5_filepath is None:
        return
    if "label" in os.path.basename(n5_filepath).lower():
        return
    if "gre" in os.path.basename(n5_filepath).lower():
        return
    runno = get_runno_from_filename(n5_filepath)
    contrast = get_contrast_from_filename(n5_filepath)

    logging.info("data nhdr file = {}".format(data_nhdr_file))
    # stash the original name away just in case it is a color, will be needed later. will not be used in non-color case
    color_nhdr_file = data_nhdr_file
    if "color" in data_nhdr_file:
        logging.info("processing COLOR file. Will use the DWI nhdr file for metadata purposes. ")
        search_dir = os.path.dirname(data_nhdr_file)
        data_nhdr_file = find_nhdr_file(runno, "dwi", local_search_dir=search_dir)
    data_nhdr = nrrd.read_header(data_nhdr_file)
    if "color" in os.path.basename(n5_filepath).lower():
        color_vox = np.array(get_voxel_size_from_color_nhdr(color_nhdr_file))
        sd = data_nhdr["space directions"]
        data_nhdr["space directions"] = sd / np.abs(sd).max(axis=1, keepdims=True) * color_vox[:, None]
    # TODO: find label nhdr on freshscreen
        # do we ever upload label nhdr to freshscreen? how do we distribute labels to people?
    label_nhdr = nrrd.read_header(label_nhdr_file)

    output_file = pathjoin(output_dir, "{}.json".format(os.path.basename(n5_filepath)))
    # in case this is passed to us as an absolute global path, we only need the basename
    label_precomputed_file = os.path.basename(label_precomputed_file)

    logging.info("PROCESSING FOLLOWING JOB:")
    logging.info("\tfilename = {}".format(n5_filepath))
    logging.info("\tlabel_precomputed_file = {}".format(label_precomputed_file))
    logging.info("\tdata nhdr = {}".format(data_nhdr))
    logging.info("\tlabel nhdr = {}".format(label_nhdr))
    logging.info("\toutput_file = {}".format(output_file))
    write_grayscale_json(n5_filepath, label_precomputed_file, data_nhdr, label_nhdr, output_file, is_stereotaxic=is_stereotaxic, overlay_layers=overlay_layers, color_overlay=color_overlay)


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
