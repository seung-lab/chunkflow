# -*- coding: utf-8 -*-
"""Operators to generate tasks or process chunks.
"""
from .cloud_watch import CloudWatchOperator
from .create_chunk import CreateChunkOperator
from .create_thumbnail import CreateThumbnailOperator
from .crop_margin import CropMarginOperator
from .cutout import CutoutOperator
from .inference import InferenceOperator
from .mask import MaskOperator
from .neuroglancer_view import NeuroglancerViewOperator
from .normalize_section_contrast import NormalizeSectionContrastOperator
from .read_file import ReadFileOperator
from .save import SaveOperator
from .upload_log import UploadLogOperator
from .view import ViewOperator
from .write_h5 import WriteH5Operator
