.. _tutorial:

Tutorial
##############


Composable Commandline Interface
*********************************

Visualization of a Chunk
==========================

CloudVolume Viewer
--------------------------------------
create a random image volume and visualize it in browser::

    chunkflow create-chunk view

open the link and you should see a image volume in browser:

|random_image_in_cloudvolume_viewer|

.. |random_image_in_cloudvolume_viewer| image:: _static/image/random_image_in_cloudvolume_viewer.png
    :width: 600

Neuroglancer
---------------------------------
create a random image volume and show it in neuroglancer::

    chunkflow create-chunk neuroglancer

open the link and you should see it:

|random_image_in_neuroglancer|

.. |random_image_in_neuroglancer| image:: _static/image/random_image_in_neuroglancer.png
    :width: 600

Note that the random image center is blacked out.

Now let's do some 

Distributed Computation
---------------------------

Deploy in Local Computers
~~~~~~~~~~~~~~~~~~~~~~~~~~

Deploy to Kubernetes Cluster in Cloud
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
