import json
import os
import sys

with open(sys.argv[1]) as f:
    param = json.load(f)

envs = ["IMAGE_PATH", "IMAGE_MIP", "AFF_PATH", "AFF_MIP", "EXPAND_MARGIN_SIZE", "PATCH_NUM"]
for e in envs:
    if e in param:
        print('export {}="{}"'.format(e, param[e]))

if param.get("FILL_MISSING", False):
    print('export FILL_MISSING="--fill-missing"')

if "PYTORCH_MODEL_PATH" in param:
    print('export PYTORCH_MODEL_PKG="{}"'.format(os.path.join(param["PYTORCH_MODEL_PATH"], "pytorch-model.tgz")))

if "IMAGE_HISTOGRAM_PATH" in param:
    upper_threshold = param.get("CONTRAST_NORMALIZATION_UPPER_THRESHOLD", 0.01)
    lower_threshold = param.get("CONTRAST_NORMALIZATION_LOWER_THRESHOLD", 0.01)
    print('export CONSTRAST_NORMALIZATION="normalize-section-contrast -p {} -l {} -u {}"'.format(param["IMAGE_HISTOGRAM_PATH"], lower_threshold, upper_threshold))

if "IMAGE_MASK_PATH" in param:
    operator = "mask --name=mask_image --volume-path={} --mip {}".format(param["IMAGE_MASK_PATH"], param["IMAGE_MASK_MIP"])
    if "INVERT_IMAGE_MASK_PATH" in param:
        operator += " --inverse"
    if param.get("FILL_MISSING", False):
        operator += " --fill-missing"
    print('export MASK_IMAGE="{}"'.format(operator))

if "AFF_MASK_PATH" in param:
    operator = "mask --name=mask_aff --volume-path={} --mip {}".format(param["AFF_MASK_PATH"], param["AFF_MASK_MIP"])
    if "INVERT_AFF_MASK_PATH" in param:
        operator += " --inverse"
    if param.get("FILL_MISSING", False):
        operator += " --fill-missing"
    print('export MASK_AFF="{}"'.format(operator))
