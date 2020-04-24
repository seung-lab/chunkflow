import json
import os
import sys

with open(sys.argv[1]) as f:
    param = json.load(f)

bbox = param["BBOX"]

param["VOL_START"] = " ".join(str(x) for x in bbox[0:3][::-1])
param["VOL_STOP"] = " ".join(str(x) for x in bbox[3:6][::-1])
param["IMAGE_RESOLUTION"] = " ".join(str(x) for x in param["IMAGE_RESOLUTION"][::-1])

envs = ["VOL_START", "VOL_STOP", "AFF_PATH", "AFF_MIP", "IMAGE_RESOLUTION", "MAX_RAM", "MAX_MIP"]

for e in envs:
    print('export {}="{}"'.format(e, param[e]))
