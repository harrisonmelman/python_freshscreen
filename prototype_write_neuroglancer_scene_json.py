# docstring convention example:
# like git commit rules. one line short description, skip a line, then more descriptive

#def complex(real=0.0, imag=0.0):
"""Form a complex number.

    Keyword arguments:
    real -- the real part (default 0.0)
    imag -- the imaginary part (default 0.0)
    """


# create a dict that represents the json
# uise json library to convert it to json notation
# write that string to a file

# IMPORTS
from typing import OrderedDict
import urllib.parse
import json
import logging
import os
import sys
import nrrd
import numpy as np
import functools
# header = nrrd.read_header('output.nrrd') # possible to only read the nrrd header

# GLOBAL VARIABLES -- caps-locking these to make it obvious they are global and BAD
# needed at the beginning of all aws s3 filepaths. holds the file protocol and s3 bucket information
# example usage:
    # n5_filepath="{}/{}/{}.format(N5_PREFIX, filename, N5_SUFFIX)"
    # precomputed_filepath="{}/{}.format(PRECOMPUTED_PREFIX, filename)"
N5_PREFIX = "n5://https://d3mof5o.s3.amazonaws.com"
PRECOMPUTED_PREFIX = "precomputed://https://d3mof5o.s3.amazonaws.com"
# this is needed at the end of all N5 file paths, to tell neuroglancer where the data lives in the folder struct
N5_SUFFIX="setup0/timepoint0/"
LIGHTSHEET_CONTRASTS = ["NeuN", "autof", "Thy1", "MBP", "ChAT"]

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

def get_runno_from_filename(filename: str):
    f = filename.split("_")
    # run number should always be the second to last field in the freshscreen name
    guess = f[-2]
    # matching criteria are it starts with N or S AND is length 6 (N12345) or length 11 (N12345NLSAM)
    if guess is not None and guess[0] in "NS" and len(guess) in [6,11]:
        return guess
    # if for whatever reason this does not match as a runnumber, loop through the others to see if you can find it
    for guess in f:
        if guess is not None and guess[0] in "NS" and len(guess) in [6,11]:
            return guess
    logging.warning("unable to find specimen run number from the filename: {}".format(filename))
    return None

def get_contrast_from_filename(filename: str):
    #return filename.split("_")[-1].split(".")[0]
    # dirty fix for a couple of the lightsheet volumes endinging in -ls on freshscreen
    return filename.split("_")[-1].split(".")[0].split("-")[0]

def get_default_threshold(filename: str):
    """from filename, figure out which type of contrast it is, and return the appropriate threshold value from the dictionary

    Completely dependendent on current (11/7/2022) freshscreen filenames"""
    DEFAULT_THRESHOLDS = {
        "dwi" : 25000,
        "fa" : 1,
        "ad" : 0.8, #0.4, #0.02,
        "rd" : 0.8, #0.4, #0.02,
        "md" : 0.8, #0.4, #0.02,
        "qa" : 1,
        "gfa" : 1,
        "color" : 128,
        "NeuN" : 800,
        "MBP" : 1300,
        "autof" : 1200,
        "Thy1" : 1500,
        "IBA1" : 1300,
        "Syto16" : 1300,
        "tdi" : 20,
        "tdi3" : 20,
        "tdi5" : 20
    }

    # in freshscren, all filenames look similar (${spec_id}_${number}_${runno}_${contrast}.n5)
    # split on "_" and take the last one
    # split on "." to remove the file extension
    contrast = get_contrast_from_filename(filename)
    return DEFAULT_THRESHOLDS[contrast]

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
    print("\n\n{}".format(url))
    return url

def get_voxel_size_from_nhdr_dict(nhdr: OrderedDict):
    """Determines the voxel size (in meters) of an image given the nhdr file as an OrderedDict (already read into an object by nrrd.read_nhdr)
    does not assume isotropic volumes. returns the voxel size of each dimension as an array"""
    nhdr["space directions"]
    sizes = []
    n_dims = nhdr["space directions"].shape[0]
    for i in range(n_dims):
        size = max(abs(nhdr["space directions"][i]))
        # conversion from mm to m
        size = size / 1000
        sizes.append(size)
    return sizes

def get_s3_url_from_file_basename(filename: str):
    """given a file basename (e.g. 200316-1-1_N10_N58204NLSAM_fa.n5), return the full url (in neuroglancer format) of that file on the d3mof5o s3 bucket

    This function determines if the file is of type n5 or precomputed, and builds the path accordingly"""
    # TODO: add an argument to make this work for arbitrary s3 buckets
    # TODO: test to ensure that os.path.join always does the right job here. PROBABLY fails on windows
    suffix = filename.split(".")[-1]
    if "n5" in suffix:
        return os.path.join(N5_PREFIX, filename, N5_SUFFIX)
    elif "precomputed" in suffix:
        return os.path.join(PRECOMPUTED_PREFIX, filename)
    else:
        logging.warning("ERROR: file does not seem to be in n5 or precomputed format")
        return None

def update_matrix_transform(xform_matrix: dict, nhdr_dict: dict, label_voxel_sizes: list):
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

# a "fix" to make sure the coronal view in neuroglacner is right side up
def flip_xform_dimension(xform_matrix: dict, dim: int=2):
    image_matrix_numpy_array = np.asarray(xform_matrix)
    flip_row_index = np.argmax(image_matrix_numpy_array[:,2] != 0)
    # flip sign of each value in the found row
    for i in range(len(xform_matrix[flip_row_index])):
        xform_matrix[flip_row_index][i] = -1 * xform_matrix[flip_row_index][i]

# TODO: a function to setup a full image layer. this should follow pretty closely as the write_three_layer json function
# this would allow us to generalize up to N-dimensions, making color images trivial (??? probably still not trivial)
def setup_layer():
    pass

# TODO: data_threshold_max by default currently looks at a dict of deafult values. eventually, do NOT allow this. force user to pass a decent value to this function
# TODO: handle the orientation label layer data["layers"][0] -- currently keeping this one hidden
# TODO: color images are currently ignored
def write_three_layer_json(data_file: str, label_file: str, data_nhdr_file: str, label_nhdr_file: str, output_file: str,  json_template: str="data/neuroglancer_json_templates/N58204NLSAM_dwi_template.json", data_threshold_max=None):
    """Function to write a json file for our most typical use case: one image volume, one labelset, and one orientation label layer

    inputs:
    data_file -- name of the root folder of data file, as it sits on S3
    label_file -- name of the root folder of label file, as it sits on S3
    data_nhdr -- path to nhdr file for the data. necessary to pull metadata from
    label_nhdr -- analogous to data_nhdr for label_file
    output_file -- the freshscreen display json file to eventually write to.
    json_template -- a local file to act as json template. this script will add and edit what it needs to, but will not delete anything it does not have to. this could be useful for adding future functionality (just by chanign default template)
        - current template was copied from N58204NLSAM_dwi
    data_threshold_max -- upper bound for data rendering on freshscreen. default is None [TESTING ONLY]

    returns a dict that represents the json file. DON'T (???) have this one also write it to a file. make another function decide where to put it
    # actually it is not really necessary to save this as a json file. instead it could return the url encoded string? unsure. does not matter right now


    assumptions:
    - the data is stored on the d3mof5o AWS S3 bucket and follows all freshscreen organization conventions
    - file (actually a directory) ends in .precomputed or .n5. will infer which one it is based on the extension

    - layer[0] is orientation label
    - layer[1] is image volume
    - layer[2] is segmentation volume
    """
    """function to write json for our most typical use case: three layers
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
            one for each SPECIMEN. should be the same parameters for all (except origin transform)
        3) orientation label
            will always be the same file. placement might be a pain"""

    # check if json_template is relative or abspath and handle it accordingly
    if not os.path.isabs(json_template):
        dirname = os.path.dirname(os.path.realpath(__file__))
        json_template = os.path.join(dirname, json_template)
        print(dirname)
    print(json_template)
    exit()

    # .lower() converts string to all lowercase
    if "color" in get_contrast_from_filename(data_file).lower():
        logging.warning("Color files are not currently supported. Make this one manually.")
        return None

    if data_threshold_max is None:
        pass

    data = None
    with open(json_template) as template:
        data = json.load(template)
    if data is None:
        logging.warning("Unable to read template file: {}".format(json_template))
        return None

    ###**************####
    # edit image layer
    ###**************####
    # data["layers"][1] is the image volume
    image_layer = data["layers"][1]
    image_layer["source"]["url"] = get_s3_url_from_file_basename(data_file)

    # edit the layer name
    image_layer["name"] = "{} {}".format(get_runno_from_filename(data_file), get_contrast_from_filename(data_file))

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
    data_nhdr = nrrd.read_header(data_nhdr_file)

    # set inputDimensions to nhdr_voxel_size / 1000 (nhdr is in mm, json is in m)
    data_voxel_sizes = get_voxel_size_from_nhdr_dict(data_nhdr)
    i = 0
    for key in image_layer["source"]["transform"]["inputDimensions"].keys():
        image_layer["source"]["transform"]["inputDimensions"][key][0] = data_voxel_sizes[i]
        i+=1
    logging.info("updated inputDimensions to: {}".format(image_layer["source"]["transform"]["inputDimensions"]))

    # set outputDimensions. this should be inferred from the LABEL nhdr (because this will always be at MRI resolution, even if the image is a lightsheet)
    # TODO: this is almost identical to above 6 lines. functionize?
        # i also repeat these two chunks (nearly) verbatim below for the rccf label layer
    label_nhdr = nrrd.read_header(label_nhdr_file)
    label_voxel_sizes = get_voxel_size_from_nhdr_dict(label_nhdr)
    i = 0
    for key in image_layer["source"]["transform"]["outputDimensions"].keys():
        image_layer["source"]["transform"]["outputDimensions"][key][0] = label_voxel_sizes[i]
        i+=1
    logging.info("updated outputDimensions to: {}".format(image_layer["source"]["transform"]["outputDimensions"]))

    # update shader controls (default window and level)
    # range is range of data to be rendered on screen
    # window is range of the slider the user can use to edit the range in the GUI
    image_layer["shaderControls"]["normalized"]["range"][0] = 0
    image_layer["shaderControls"]["normalized"]["window"][0] = 0
    max_range = get_default_threshold(data_file)
    image_layer["shaderControls"]["normalized"]["range"][1] = max_range
    image_layer["shaderControls"]["normalized"]["window"][1] = max_range*1.2

    update_matrix_transform(image_layer["source"]["transform"]["matrix"], data_nhdr, label_voxel_sizes)
    flip_xform_dimension(image_layer["source"]["transform"]["matrix"], 2)

    ###**************####
    # edit rCCF label layer
    ###**************####
    # TODO: add more functions. this looks almost identical to code above
    # transform matrix should be identical to MRI volumes -- will be slightly different from lightsheet. hoping that the nhdr will solve ls problems (it did)
    rccf_label_layer = data["layers"][2]
    rccf_label_layer["source"]["url"] = get_s3_url_from_file_basename(label_file)
    rccf_label_layer["name"] = "{} {}".format(get_runno_from_filename(label_file), get_contrast_from_filename(label_file))

    # set inputDimensions to nhdr_voxel_size / 1000 (nhdr is in mm, json is in m)
    i = 0
    for key in rccf_label_layer["source"]["transform"]["inputDimensions"].keys():
        rccf_label_layer["source"]["transform"]["inputDimensions"][key][0] = label_voxel_sizes[i]
        i+=1
    logging.info("updated RCCF label inputDimensions to: {}".format(rccf_label_layer["source"]["transform"]["inputDimensions"]))

    label_nhdr = nrrd.read_header(label_nhdr_file)
    label_voxel_sizes = get_voxel_size_from_nhdr_dict(label_nhdr)
    i = 0
    for key in rccf_label_layer["source"]["transform"]["outputDimensions"].keys():
        rccf_label_layer["source"]["transform"]["outputDimensions"][key][0] = label_voxel_sizes[i]
        i+=1
    logging.info("updated RCCF label outputDimensions to: {}".format(rccf_label_layer["source"]["transform"]["outputDimensions"]))

    update_matrix_transform(rccf_label_layer["source"]["transform"]["matrix"], label_nhdr, label_voxel_sizes)
    flip_xform_dimension(rccf_label_layer["source"]["transform"]["matrix"], 2)

    ###*******####
    # edit other (things that are not within data["layers"])
    ###*******####
    # make the image layer be the default control panel that shows
    # TODO: this does not work. really don't understand why
    data["selectedLayer"]["layer"] = image_layer["name"]
    data["selectedLayer"]["visible"] = True

    # also change output voxel size for the orientation label -- maybe this is where my scene is getting confused
    # these could really all be done in the same loop, because the keys will always be [x,y,z] (TODO: confirm this)
    """i = 0
    for key in data["layers"][0]["source"]["transform"]["outputDimensions"].keys():
        data["layers"][0]["source"]["transform"]["outputDimensions"][key][0] = label_voxel_sizes[i]
        i+=1"""
    i = 0
    for key in data["dimensions"].keys():
        data["dimensions"][key][0] = label_voxel_sizes[i]
        i+=1

    logging.info("final form of data dict: \n{}".format(data))

    # TODO: output file name? -- will be exacly the n5 filename plus .json at the end
    # who should be responsible for choosing that name?
    write_freshscreen_display_json(data, data_file, output_file)
    exit()
    #url = convert_json_to_url(convert_dict_to_string(data))

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
        # if the neuroglancer name is the default "timepointN", then don't use that and just use the full filename
        # TODO: this might? wreck havoc on color files, but i am not handling those right now anyways. so it's ok..?
        if "timepoint" in data["layers"][1]["name"].lower():
            organization_json["object_displayName"] = data_file
        else:
            organization_json["object_displayName"] = data["layers"][1]["name"]
    else:
        organization_json["object_displayName"] = display_name

    organization_json["view_neuroglancerURL"] = convert_json_to_url(convert_dict_to_string(data))
    write_dict_to_json_file(output_file, organization_json)

# TODO: boto3 is an aws sdk for python. but it is confusing
def loop_through_specimen_in_freshscreen(spec_id: str, output_dir: str, label_file: str=None):
    """loops through all n5 or precomputed files in s3 connected to the provided specimen id.  Skips over color files"""
    import subprocess
    rclone = "K:/DevApps/rclone-v1.58.1-windows-amd64/rclone.exe"

    spec_id_fresh = spec_id.replace("_", "-") # follows freshscreen specifications i.e. no underscores allowed
    # on windows i had to split this up because pipes are playing funny (read: not working). do not fully understand why.
    # TODO: test this doesn't break everything on mac
    # i think that i need to do with open... several times because i am usig the command line to read from the file (with grep and awk). python having the file open locks it from other prying eyes. there MIGHT(??) be a work around to this
    # this is the full command as you would run it on the command line
    # TODO: also, subprocess.run takes stdin as an argument. i am supposed to be able to pass a file handle to it, but that isn't working. that might be mroe reliable then what i am currently doing+
    #cmd_str="{} lsd freshscreen:d3mof5o | grep {} | awk '{{print $5}}'".format(rclone, spec_id_fresh)

    cmd_str="{} lsd freshscreen:d3mof5o".format(rclone)
    a=subprocess.run(cmd_str, shell=True, capture_output=True)
    temp_file_list = "K:/workstation/code/display/python_freshscreen/data/temp/subprocess_output"
    with open(temp_file_list, "w+b") as f:
        f.write(a.stdout)

    cmd_str = "grep {} {}".format(spec_id_fresh, temp_file_list)
    a=subprocess.run(cmd_str, capture_output=True)
    with open(temp_file_list, "w+b") as f:
        f.write(a.stdout)

    cmd_str = "awk '{{print $5}}' {}".format(temp_file_list)
    a=subprocess.run(cmd_str, capture_output=True)
    with open(temp_file_list, "w+b") as f:
        f.write(a.stdout)

    filelist = a.stdout.decode("utf-8").split("\n")
    if label_file is None:
        # if user does not provide label file, loop through s3 bucket to try and find it. returns if cannot find
        for f in filelist:
            if "label" in f.lower():
                label_file = f
                break
        if label_file is None:
            logging.warning("cannot find a label file in s3 for specimen {}".format(spec_id_fresh))
            return None
    for f in filelist:
        if "color" in f.lower():
            logging.warning("color images not currently supported")
            continue
        if "label" in f.lower():
            continue
        if "gre" in f.lower():
            continue
        if not f or f is None:
            continue
        # extract run number and contrast to find the nhdr
        runno = f.split("_")[2]
        contrast = get_contrast_from_filename(f)

        data_file = f
        data_nhdr = os.path.join(NHDR_DIR, "{}_{}.nhdr".format(runno, contrast))
        if not os.path.isfile(data_nhdr):
            data_nhdr = os.path.join(NHDR_DIR, "{}_{}.nhdr".format(spec_id, contrast))
        # TODO: make next line more extensible. it needs a "find" feature to handle cases when filename looks liie $runno_RCCF_MC_labels instead of the canonical
        label_nhdr = os.path.join(NHDR_DIR,"labels","RCCF", "{}_RCCF_labels.nhdr".format(runno))
        if not os.path.isfile(label_nhdr):
            label_nhdr = os.path.join(NHDR_DIR,"labels","RCCF", "{}_RCCF_MC_labels.nhdr".format(runno))
        if not os.path.isfile(label_nhdr):
            logging.error("cannot find label file nhdr locally")
            exit()

        if contrast in LIGHTSHEET_CONTRASTS:
            # lightsheet volumes are saved locally with spec id instead of runno in the filename
            # TODO: clarify between freshscreen and CIVM specimen id's (i.e. difference between 190415-2_1 and 190415-2-1)
            print("contrast is: {}".format(contrast))
            if contrast in "mGRE":
                contrast = "mAVG"
            data_nhdr = os.path.join(NHDR_DIR, "{}_{}.nhdr".format(spec_id, contrast))

        output_file = os.path.join(output_dir, "{}.json".format(f))
        print("RUNNING FOR SPECIMEN:\n\t data_file = {}\n\t data_nhdr = {}\n\t label_file = {}\n\t label_nhdr = {}\n\t output_file = {}\n\n".format(data_file, data_nhdr, label_file, label_nhdr, output_file))
        write_three_layer_json(data_file, label_file, data_nhdr, label_nhdr, output_file)


#******!*!*!*!*!*!*!*!*!*!*!*!!*****************#
# MAIIN with command line arguments
#******!*!*!*!*!*!*!*!*!*!*!*!!*****************#
logging.getLogger().setLevel(logging.INFO)
if len(sys.argv) > 1:
    #specimen_id = "200302-1_1"
    project_code = sys.argv[1]
    specimen_id = sys.argv[2]
    output_dir = "/Users/harry/scratch/neuroglancer_python_prototype/data/other/freshscreen_json_display_files/{}".format(specimen_id)
    # TODO: smartly determine what system we are on. windows or mac? -- should always be citrix really
    #NHDR_DIR = "/Volumes/PWP-CIVM-CTX01/{}/{}/Aligned-Data".format(project_code, specimen_id)
    NHDR_DIR = "Q:/{}/{}/Aligned-Data".format(project_code, specimen_id)
    loop_through_specimen_in_freshscreen(specimen_id, output_dir)
else:
    logging.warning("Not enough input arguments. Requires project_code and specimen_id as positional arguments")


exit()
#**********************************************************************************************#
# MAIIN
data_file = "200316-1-1_N10_N58204NLSAM_fa.n5"
label_file = "200316-1-1_N58204NLSAM_RCCF_labels.precomputed"
data_nhdr = os.path.join(NHDR_DIR,"N58204NLSAM_fa.nhdr")
label_nhdr = os.path.join(NHDR_DIR,"labels","RCCF", "N58204NLSAM_RCCF_labels.nhdr")

# now test witha  lightsheet
data_nhdr = os.path.join(NHDR_DIR,"fullres_2003NeuN.nhdr")
data_file = "200316-1-1_N12_N58204NLSAM_NeuN-ls.n5"

# test with 190415-1_1
project_code = "19.gaj.43"
spec_id = "190415-2_1"
NHDR_DIR = "/Volumes/PWP-CIVM-CTX01/{}/{}/Aligned-Data".format(project_code, spec_id)
data_file = "190415-2-1_N09_N57205NLSAM_fa.n5"
label_file = "190415-2-1_N57205NLSAM_labels.precomputed"
data_nhdr = os.path.join(NHDR_DIR,"N57205NLSAM_fa.nhdr")
label_nhdr = os.path.join(NHDR_DIR,"labels","RCCF", "N57205NLSAM_RCCF_labels.nhdr")


# for now, min/max should be defined by a  dict of default values
# make default be user's input, if none are given, yell at user and use default dict
#write_three_layer_json(data_file,label_file, data_nhdr, label_nhdr, data_threshold_max=None)
specimen_id = "190415-2_1"
output_dir_default = "/Users/harry/scratch/neuroglancer_python_prototype/data/other/freshscreen_json_display_files/{}".format(specimen_id)
if not os.path.isdir(output_dir_default):
    os.makedirs(output_dir_default)
loop_through_specimen_in_freshscreen(specimen_id, output_dir_default)
#**********************************************************************************************#
