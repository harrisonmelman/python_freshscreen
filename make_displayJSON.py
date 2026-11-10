# given a neuroglancer JSON (what you paste into the {} dialogue box within neuroglancer) and a displayName, create a freshscreen display JSON file
from freshscreen.output import write_freshscreen_display_json
import json
import sys
import os


def encode_to_freshscreen_json(input_dict_file, display_name, output_file):
    with open(input_dict_file) as f:
        data = json.load(f)
    write_freshscreen_display_json(data, "", output_file, display_name)



#in_dir = "data/prototype_advanced_scenes"


in_file = sys.argv[1]
display_name = sys.argv[2]
out_dir = os.path.dirname(in_file)
out_file = os.path.join(out_dir,'freshscreen_jsons',f"{display_name}.json")

encode_to_freshscreen_json(in_file, display_name, out_file)
