"""Regression tests for scene generation, using real data.

Each case runs process_one_image and compares the result against a saved
display json in tests/baseline. The comparison decodes the neuroglancer url,
so a change in key order does not count as a failure.

These tests rely on the n5 files already existing in the freshscreen library
folder, so they only cover JSON creation. To test the full pipeline, including
n5 and precomputed generation, use prepare1image. Nothing is written to the
library folder.

Run from anywhere:
    python tests/test_scenes.py

After a deliberate change to the output, regenerate the baselines and review
the diff before committing it:
    UPDATE_BASELINE=1 python tests/test_scenes.py
"""
import json
import os
import sys
import tempfile
import unittest
import urllib.parse
from unittest import mock

TESTS_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.dirname(TESTS_DIR))
import prototype_write_neuroglancer_scene_json as scene_writer

# 24.chdi.01 is the test set. relatively lowres so it runs fast
IS_CITRIX = False
GROUP = "15mo-WT-M"
if IS_CITRIX:
    FS_LIBRARY = "B:/Freshscreen_Library/hmm56_test"
    INPUT_DIR = f"B:/24.chdi.01-PHASE2/AVG/{GROUP}/Aligned-Data"
else:
    FS_LIBRARY = "/Users/harry/Documents/work/example_chdi_data/freshscreen_library"
    INPUT_DIR = f"/Users/harry/Documents/work/example_chdi_data/{GROUP}/Aligned-Data"

LABEL_NHDR = f"{INPUT_DIR}/labels/RCCF/{GROUP}_RCCF_labels.nhdr"
LABEL_PRECOMPUTED = f"{GROUP}_N90_RCCF_labels.precomputed"
DWI_N5 = f"{GROUP}_N02_dwi.n5"
COLOR_N5 = f"{GROUP}_N07_gqi-color.n5"
DWI_NHDR = f"{INPUT_DIR}/{GROUP}_dwi.nhdr"
COLOR_NHDR = f"{INPUT_DIR}/{GROUP}_gqi_color.nhdr"

CASES = {
    "dwi": dict(n5_filepath=f"{FS_LIBRARY}/{DWI_N5}", data_nhdr_file=DWI_NHDR, is_stereotaxic=True),
    "gqi_color": dict(n5_filepath=f"{FS_LIBRARY}/{COLOR_N5}", data_nhdr_file=COLOR_NHDR, is_stereotaxic=True),
    "dwi_with_overlay": dict(
        n5_filepath=f"{FS_LIBRARY}/{DWI_N5}",
        data_nhdr_file=DWI_NHDR,
        is_stereotaxic=True,
        color_overlay="green",
        overlay_layers=[f"{TESTS_DIR}/fixtures/AD_change_lookup.txt"],
    ),
}


def read_display_json(path):
    """returns the display json as a dict, with the url replaced by the decoded scene"""
    with open(path) as f:
        display = json.load(f)
    encoded_scene = display.pop("view_neuroglancerURL").split("#!")[1]
    display["scene"] = json.loads(urllib.parse.unquote(encoded_scene))
    return display


def run_case(case, output_file):
    """runs process_one_image for one case and writes its display json to output_file

    The real write is intercepted, because the output folder has to be the
    library folder for the n5 lookup to work."""
    with mock.patch.object(scene_writer, "write_freshscreen_display_json") as write:
        scene_writer.process_one_image(
            output_dir=FS_LIBRARY,
            label_nhdr_file=LABEL_NHDR,
            label_precomputed_file=LABEL_PRECOMPUTED,
            **case,
        )
    write.assert_called_once()
    scene, data_file, _ = write.call_args.args
    scene_writer.write_freshscreen_display_json(scene, data_file, output_file)


@unittest.skipUnless(os.path.isdir(FS_LIBRARY), f"test data not found: {FS_LIBRARY}")
class SceneBaselineTests(unittest.TestCase):
    maxDiff = None

    def test_scenes_match_baselines(self):
        for name, case in CASES.items():
            with self.subTest(name), tempfile.TemporaryDirectory() as temp_dir:
                baseline_file = f"{TESTS_DIR}/baseline/{name}.json"
                if os.environ.get("UPDATE_BASELINE"):
                    run_case(case, baseline_file)
                    continue
                output_file = f"{temp_dir}/{name}.json"
                run_case(case, output_file)
                self.assertEqual(read_display_json(output_file), read_display_json(baseline_file))


if __name__ == "__main__":
    unittest.main()
