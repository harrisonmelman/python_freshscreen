"""Site constants: where data lives and which tools to call."""

# Prepended to every S3 path. Example usage:
#   n5 url          = pathjoin(N5_PREFIX, filename, N5_SUFFIX)
#   precomputed url = pathjoin(PRECOMPUTED_PREFIX, filename)
# Older scenes point at the retired bucket:
#   N5_PREFIX = "n5://https://d3mof5o.s3.amazonaws.com"
#   PRECOMPUTED_PREFIX = "precomputed://https://d3mof5o.s3.amazonaws.com"
N5_PREFIX = "n5://https://fast-s3.oit.duke.edu/freshscreen/dh-civm-public"
PRECOMPUTED_PREFIX = "precomputed://https://fast-s3.oit.duke.edu/freshscreen/dh-civm-public"

# Appended to every n5 path, to tell Neuroglancer where the data lives in the folder structure
N5_SUFFIX = "setup0/timepoint0/"

NEUROGLANCER_URL = "https://neuroglancer.freshscreen.com"

LIGHTSHEET_CONTRASTS = ["NeuN", "autof", "Thy1", "MBP", "ChAT"]

RCLONE = "K:/DevApps/rclone-v1.58.1-windows-amd64/rclone.exe"
