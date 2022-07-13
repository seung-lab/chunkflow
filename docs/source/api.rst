.. _api:

API Reference
***************

Unless otherwise stated below functions are available at top level import,
i.e. after ``import chunkflow``.

Chunk
=======
.. autoclass:: chunkflow.chunk.Chunk
   :members:

Image
---------
.. automodule:: chunkflow.chunk.image
   :members:

.. automodule:: chunkflow.chunk.validate
   :members:

Flow
===========
.. automodule:: chunkflow.flow.flow
   :members:

Operators
------------
.. automodule:: chunkflow.flow
   :members:

.. autoclass:: chunkflow.flow.cloud_watch.CloudWatchOperator
   :members:

.. autoclass:: chunkflow.flow.load_precomputed.ReadPrecomputedOperator
   :members:

.. autoclass:: chunkflow.flow.downsample_upload.DownsampleUploadOperator
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

.. autoclass:: chunkflow.flow.save_precomputed.SavePrecomputedOperator
   :members:

.. autoclass:: chunkflow.flow.save_pngs.SavePNGsOperator
   :members:

.. autoclass:: chunkflow.flow.view.ViewOperator
   :members:


Lib
=========

.. automodule:: chunkflow.lib
   :members:

AWS
------------
.. automodule:: chunkflow.lib.aws.cloud_watch
   :members:

.. automodule:: chunkflow.lib.aws.sqs_queue
   :members:
