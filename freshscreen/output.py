"""Encode scenes as Neuroglancer urls and write Freshscreen display JSONs."""
import json
import logging
import os
import urllib.parse

import numpy as np

from .config import NEUROGLANCER_URL


class NpEncoder(json.JSONEncoder):
    """lets json encode numpy scalars and arrays"""
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


def convert_dict_to_string(data: dict):
    return json.dumps(data, ensure_ascii=False, cls=NpEncoder)


def write_dict_to_json_file(file: str, data: dict):
    """writes data as json to file, creating the folder if needed. WILL OVERWRITE any existing file"""
    output_dir = os.path.dirname(file)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    with open(file, 'w') as f:
        json.dump(data, f, ensure_ascii=False, cls=NpEncoder)


def convert_json_to_url(file: str):
    """returns a shareable neuroglancer link for a scene, given either a json string or the path to a json file"""
    if "{" in file:
        # a JSON string. Do not strip whitespace: shader code like "#uicontrol invlerp" needs its spaces
        encoded_text = urllib.parse.quote(file)
    else:
        with open(file) as f:
            # TODO: whitespace stripping breaks shaders here too. originally done to avoid a "url too long" error
            encoded_text = urllib.parse.quote("".join(f.read().split()))
    url = "{}/#!{}".format(NEUROGLANCER_URL, encoded_text)
    logging.info("\n\n{}".format(url))
    return url


def write_freshscreen_display_json(data: dict, data_file: str, output_file: str, display_name: str = None):
    """writes the Freshscreen display JSON for a scene: the S3 object name, a display name and the neuroglancer url

    data_file is the dataset's S3 basename (use "" for scenes that span several datasets).
    If display_name is not given, the name of the first layer is used."""
    organization_json = dict()
    organization_json["object_Name"] = data_file
    if display_name is None:
        organization_json["object_displayName"] = data["layers"][0]["name"]
    else:
        organization_json["object_displayName"] = display_name

    organization_json["view_neuroglancerURL"] = convert_json_to_url(convert_dict_to_string(data))
    write_dict_to_json_file(output_file, organization_json)
    print(organization_json)
    print(output_file)
