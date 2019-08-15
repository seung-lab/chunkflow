.. _api:

API Reference
=============

Unless otherwise stated below functions are available at top level import,
i.e. after ``import chunkflow``.

Chunk
------------
.. autoclass:: chunkflow.chunk.Chunk
   :members:

Image
~~~~~~~
.. automodule:: chunkflow.chunk.image
   :members:

.. automodule:: chunkflow.chunk.validate
   :members:

Flow
------------
.. automodule:: chunkflow.flow.flow
   :members:

Operators
~~~~~~~~~~
.. automodule:: chunkflow.flow
   :members:

.. autoclass:: chunkflow.flow.cloud_watch.CloudWatchOperator
   :members:

.. autoclass:: chunkflow.flow.create_chunk.CreateChunkOperator
   :members:

.. autoclass:: chunkflow.flow.crop_margin.CropMarginOperator
   :members:

.. autoclass:: chunkflow.flow.cutout.CutoutOperator
   :members:

.. autoclass:: chunkflow.flow.downsample_upload.DownsampleUploadOperator
   :members:

.. autoclass:: chunkflow.flow.inference.InferenceOperator
   :members:

.. autoclass:: chunkflow.flow.mask.MaskOperator
   :members:

.. autoclass:: chunkflow.flow.mesh.MeshOperator
   :members:

.. autoclass:: chunkflow.flow.neuroglancer.NeuroglancerOperator
   :members:

.. autoclass:: chunkflow.flow.normalize_section_contrast.NormalizeSectionContrastOperator
   :members:

.. autoclass:: chunkflow.flow.normalize_section_shang.NormalizeSectionShangOperator
   :members:

.. autoclass:: chunkflow.flow.read_h5.ReadH5Operator
   :members:

.. autoclass:: chunkflow.flow.read_tif.ReadTIFOperator
   :members:

.. autoclass:: chunkflow.flow.save.SaveOperator
   :members:

.. autoclass:: chunkflow.flow.save_images.SaveImagesOperator
   :members:

.. autoclass:: chunkflow.flow.view.ViewOperator
   :members:

.. autoclass:: chunkflow.flow.write_h5.WriteH5Operator
   :members:


Lib
--------------------

.. automodule:: chunkflow.lib
   :members:

AWS
~~~~~~~~~~~~~~~~~~~
.. automodule:: chunkflow.lib.aws.cloud_watch
   :members:

.. automodule:: chunkflow.lib.aws.sqs_queue
   :members:
