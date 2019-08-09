.. _api:

API Reference
##############

Unless otherwise stated below functions are available at top level import,
i.e. after ``import chunkflow``.

Chunk
*******
.. autoclass:: chunkflow.chunk.Chunk
   :members:

.. automodule:: chunkflow.chunk.validate
   :members:

Image
=======
.. automodule:: chunkflow.chunk.image.base
   :members:

.. autofunction:: chunkflow.chunk.image.adjust_grey.normalize_section_shang

Flow
******
.. automodule:: chunkflow.flow.main
   :members:

Operators
===========
.. automodule:: chunkflow.flow.operators
   :members:

.. autoclass:: chunkflow.flow.operators.cloud_watch.CloudWatchOperator
   :members:

.. autoclass:: chunkflow.flow.operators.create_chunk.CreateChunkOperator
   :members:

.. autoclass:: chunkflow.flow.operators.crop_margin.CropMarginOperator
   :members:

.. autoclass:: chunkflow.flow.operators.cutout.CutoutOperator
   :members:

.. autoclass:: chunkflow.flow.operators.downsample_upload.DownsampleUploadOperator
   :members:

.. autoclass:: chunkflow.flow.operators.inference.InferenceOperator
   :members:

.. autoclass:: chunkflow.flow.operators.mask.MaskOperator
   :members:

.. autoclass:: chunkflow.flow.operators.mesh.MeshOperator
   :members:

.. autoclass:: chunkflow.flow.operators.neuroglancer_view.NeuroglancerViewOperator
   :members:

.. autoclass:: chunkflow.flow.operators.normalize_section_contrast.NormalizeSectionContrastOperator
   :members:

.. autoclass:: chunkflow.flow.operators.normalize_section_shang.NormalizeSectionShangOperator
   :members:

.. autoclass:: chunkflow.flow.operators.read_file.ReadFileOperator
   :members:

.. autoclass:: chunkflow.flow.operators.save.SaveOperator
   :members:

.. autoclass:: chunkflow.flow.operators.save_images.SaveImagesOperator
   :members:

.. autoclass:: chunkflow.flow.operators.view.ViewOperator
   :members:

.. autoclass:: chunkflow.flow.operators.write_h5.WriteH5Operator
   :members:


Lib
*****

.. automodule:: chunkflow.lib
   :members:

AWS
===========
.. automodule:: chunkflow.lib.aws.cloud_watch
   :members:

.. automodule:: chunkflow.lib.aws.sqs_queue
   :members:
