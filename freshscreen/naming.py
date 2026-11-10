"""Parse Freshscreen filenames and turn them into S3 urls.

Freshscreen filenames look like ${spec_id}_${number}_${runno}_${contrast}.n5
Atlas and group-average datasets have no runno, e.g. 15mo-WT-M_N02_dwi.n5
"""
import logging
import os

from .config import N5_PREFIX, N5_SUFFIX, PRECOMPUTED_PREFIX


def pathjoin(*args):
    """joins path components with forward slashes, whatever the OS. used for urls and S3 paths"""
    return "/".join(args)


# TODO: this needs to be refactored or reconsidered
def get_runno_from_filename(filename: str):
    _, filename = os.path.split(filename)
    f = filename.split("_")
    print(f)
    # run number should always be the second to last field in the freshscreen name
    guess = f[-2]
    # matching criteria are it starts with N or S AND is length 6 (N12345) or length 11 (N12345NLSAM)
    if guess is not None and guess[0] in "NS" and len(guess) in [6, 11]:
        return guess
    # if for whatever reason this does not match as a runnumber, loop through the others to see if you can find it
    for guess in f:
        if guess is not None and guess[0] in "NS" and len(guess) in [6, 11]:
            return guess
    logging.warning("unable to find specimen run number from the filename: {}\n\t Will attempt to treat it as an atlas dataset. Setting runno to: {}".format(filename, f[0]))
    return f[0]


def get_contrast_from_filename(filename: str):
    """the contrast is the last "_" field, without the file extension"""
    _, filename = os.path.split(filename)
    return filename.split("_")[-1].split(".")[0]


def get_s3_url_from_file_basename(filename: str):
    """given a file basename (e.g. 200316-1-1_N10_N58204NLSAM_fa.n5), return the full Neuroglancer url of that file on S3

    The format (n5 or precomputed) is taken from the extension."""
    # TODO: add an argument to make this work for arbitrary s3 buckets
    suffix = filename.split(".")[-1]
    if "n5" in suffix:
        return pathjoin(N5_PREFIX, filename, N5_SUFFIX)
    elif "precomputed" in suffix:
        return pathjoin(PRECOMPUTED_PREFIX, filename)
    else:
        logging.warning("ERROR: file does not seem to be in n5 or precomputed format")
        return None
