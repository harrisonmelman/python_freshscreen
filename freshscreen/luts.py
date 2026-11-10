"""Segment colors for label layers."""
import csv
import json
import os

# repo root: freshscreen/ sits directly under it
REPO_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
SEGMENT_COLORS_DIR = os.path.join(REPO_DIR, "data", "segment_colors")


def rgba_to_hex(r, g, b, a):
    r = int(float(r))
    g = int(float(g))
    b = int(float(b))
    a = int(float(a))
    return f"#{r:02x}{g:02x}{b:02x}{a:02x}"


def convert_lut_file(label_lut_file):
    """reads a CIVM complex stats lookup table and returns a neuroglancer segmentColors dict (ROI number as a string -> hex color)"""
    roi_dict = {}
    with open(label_lut_file, 'r') as file:
        # read the header separately so the key can be 'ROI' instead of '# ROI'
        original_reader = csv.reader(file, delimiter='\t')
        original_headers = next(original_reader)
        cleaned_headers = [h.strip().replace('# ROI', 'ROI') for h in original_headers]

        # Skip the comment row
        next(file)

        reader = csv.DictReader(file, fieldnames=cleaned_headers, delimiter='\t')
        for row in reader:
            roi = row['ROI']
            if 'NaN' in roi:
                # TODO: i don't love this conditional. is it safe??
                continue
            roi_dict[roi] = rgba_to_hex(row['c_r'], row['c_g'], row['c_b'], row['c_a'])
    return roi_dict


def load_segment_colors(label_type: str = "RCCF"):
    """returns the dict of label ROI number -> hex color stored in data/segment_colors/{label_type}.json"""
    with open(os.path.join(SEGMENT_COLORS_DIR, "{}.json".format(label_type))) as f:
        return json.load(f)
