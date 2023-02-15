# test script to make a precomputed file
# cloud-volume and tensorstore are two tools to test
# use pynrrd to read a nrrd file into numpy arrays
# then write numpy arrays to precomputed format using cloudvolume

# TODO: i cant get this to import on a new pythn env, but it did fine on base...whatever just use base
from cloudvolume import CloudVolume
from taskqueue import LocalTaskQueue
import igneous.task_creation as tc
import numpy as np
import nrrd
import sys
from subprocess import run
import shutil
import os
import json
import logging

# TODO: create a generalized function for this that takes as input the unit to return as
def get_voxel_size_from_nhdr_dict_in_nanometers(nhdr: dict):
    """Determines the voxel size (in meters) of an image given the nhdr file as an OrderedDict (already read into an object by nrrd.read_nhdr)
    does not assume isotropic volumes. returns the voxel size of each dimension as an array"""
    nhdr["space directions"]
    sizes = []
    n_dims = nhdr["space directions"].shape[0]
    for i in range(n_dims):
        size = max(abs(nhdr["space directions"][i]))
        # conversion from mm to nanometers
        size = int(size * 1000 * 1000)
        sizes.append(size)
    return sizes

def make_precomputed(nhdr_file: str, out_file: str):
	readdata, header = nrrd.read(nhdr_file)
	print(type(readdata[0,0,0]))
	# this datatype only(?) appropriate for segmentatoin files
	readdata=readdata.astype(np.uint64)
	print(readdata.shape)
	print(header)

	# sizes: 813 1317 613
	info = CloudVolume.create_new_info(
		num_channels = 1,
		layer_type = 'segmentation', # 'image' or 'segmentation'
		data_type = 'uint64', # can pick any popular uint
		encoding = 'raw', # see: https://github.com/seung-lab/cloud-volume/wiki/Compression-Choices
		resolution = get_voxel_size_from_nhdr_dict_in_nanometers(header),
		voxel_offset = [ 0, 0, 0 ], # values X,Y,Z values in voxels
		chunk_size = [ 64, 64, 64 ], # rechunk of image X,Y,Z in voxels
		volume_size = header["sizes"]
	)

	"""info = CloudVolume.create_new_info(
		num_channels = 1,
		layer_type = 'image', # 'image' or 'segmentation'
		data_type = 'float32', # can pick any popular uint
		encoding = 'gzip', # see: https://github.com/seung-lab/cloud-volume/wiki/Compression-Choices
		# set to 15000nm which is 15um
		resolution = [ 15000, 15000, 15000 ], # X,Y,Z values in nanometers
		voxel_offset = [ 0, 0, 0 ], # values X,Y,Z values in voxels
		chunk_size = [ 64, 64, 64 ], # rechunk of image X,Y,Z in voxels
		volume_size = [ 813, 1317, 613 ], # X,Y,Z size in voxels
	)"""

	# If you're using amazon or the local file system, you can replace 'gs' with 's3' or 'file'
	#vol = CloudVolume('s3://d3mof5o/TESTING_N58204NLSAM_RCCF_labels.precomputed', info=info)
	#vol = CloudVolume('file://{}/N57205NLSAM_RCCF_labels.precomputed'.format(data_dir), info=info)
	vol = CloudVolume('file://{}'.format(out_file), info=info)


	# info (dict, rw) - Python dict representation of Neuroglancer info JSON file. You must call vol.commit_info() to save your changes to storage.
	vol.commit_info()
	#vol.provenance.description = "Description of Data"
	#vol.provenance.owners = ['email_address_for_uploader/imager'] # list of contact email addresses
	vol[:,:,:] = readdata

def downsample(precomputed_file: str, tq):
	layer_path = 'file://{}'.format(precomputed_file)
	tasks = tc.create_downsampling_tasks(layer_path, mip=0, axis="z", num_mips=4, preserve_chunk_size=True, encoding="raw", background_color=0, compress=None)
	tq.insert(tasks)
	tq.execute()

def mesh(precomputed_file: str, tq):
	layer_path = 'file://{}'.format(precomputed_file)
	tasks = tc.create_meshing_tasks(layer_path, mip=2, progress=True, compress=None)
	tq.insert(tasks)
	tq.execute()

def gunzip_all_chunks(precomputed_file: str):
	run("gunzip {}/*/*gz ".format(precomputed_file), shell=True)

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

def add_lookup_table(precomputed_file: str, lookup_table_template: str):
	# lookup table will always be the same.
	src = lookup_table_template
	dst_dir = pathjoin(precomputed_file, "segment_properties")
	dst = pathjoin(dst_dir, "info")
	if not os.path.isdir(dst_dir):
		os.makedirs(dst_dir)
	shutil.copyfile(src, dst)

	# add line in the info json to tell where to look for the lookup table
	info_json_file = pathjoin(precomputed_file, "info")
	info_json = None
	with open(info_json_file) as f:
		info_json = json.load(f)
		info_json["segment_properties"] = "segment_properties"
	if info_json is None:
		logging.warning("unable to write segment_properties to file: {}".format(info_json_file))
		return None
	with open(info_json_file, "w") as f:
		print(info_json)
		print("\n\n\n")
		print(f)
		json.dump(info_json, f, ensure_ascii=False, cls=NpEncoder)

# TODO: put this in a cenrtal location
# this is not needed on mac and this script ONLY WORKS ON MAC (or the cluster probably)
def pathjoin(*args):
    """takes a list of path components and joins them together with forward slashes as the separator
    this is easier than doing it inline because using *args automatically turns all of your input arguments into a single tuple (so you don't have to wrap your arguments to "/".join() into a list)"""
    return "/".join(args)


"""project_code = "19.gaj.43"
spec_id = "190415-2_1"
spec_id_fresh = spec_id.replace("_", "-")
runno = "N57205NLSAM"

data_dir = "/Users/harry/scratch/neuroglancer_python_prototype/data"
nhdr_file="{}/{}_RCCF_labels.nhdr".format(data_dir, runno)
#precomputed_file = "{}_subprocess_run_gunzip.precomputed".format(nhdr_file.split(".")[0])
precomputed_file = "{}/{}_{}_RCCF_labels.precomputed".format(data_dir, spec_id_fresh, runno)
# need to copy this folder into the precomputed file, and then also write the one line in the baselevel precomputed info.json
"""
##****#*#*#*#*#*#
## MAIN

if len(sys.argv) > 2:
	nhdr_file = sys.argv[1]
	precomputed_file = sys.argv[2]
	label_type = sys.argv[3]

# THIS IS FOR RCCF ONLY
# MAKE NEW ONE FOR WHS
lookup_table_template = "/Users/harry/scratch/neuroglancer_python_prototype/data/segment_properties/{}/info".format(label_type)

print(nhdr_file)
print(precomputed_file)

make_precomputed(nhdr_file, precomputed_file)

tq = LocalTaskQueue(parallel=8)
downsample(precomputed_file, tq)
mesh(precomputed_file, tq)
gunzip_all_chunks(precomputed_file)
add_lookup_table(precomputed_file, lookup_table_template)
