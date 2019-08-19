.. _tutorial:

Tutorial
##############


Composable Commandline Interface
*********************************
You can compose operators and create your own pipeline flexibly. The operators could be reused in different applications.

You can get a list of available operators by::
    
    chunkflow

You'll see a list like this:

|operator_list|

.. |operator_list| image:: _static/image/operator_list.png
    :width: 600

We keeps adding more and more operators, you might see more operators than this list. You can get help for each operator by typing `chunkflow operator --help`, such as::

    chunkflow create-chunk --help

Visualization of a Chunk
==========================

CloudVolume Viewer
--------------------------------------
Create a random image volume and visualize it in browser::

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

Input and Output
=================
I/O of Local Image
-------------------
Now let's play with some real data! You can download the data from `CREMI <https://cremi.org/>`_ challenge website. Let's use the `Dataset A <https://cremi.org/static/data/sample_A_20160501.hdf>`_ for now. 

Run this command and open the link in your browser::

   chunkflow read-h5 --file-name sample_A_20160501.hdf --dataset-path /volumes/raw neuroglancer --voxel-size 40 4 4 

Change the file path if you put your image in some other places. You should see the image in neuroglancer:

|cremi_image|

.. |cremi_image| image:: _static/image/cremi_image.png

Cutout/Save a Chunk from Large Scale Volume in Cloud Storage
-------------------------------------------------------------
We use CloudVolume_ to perform cutout/saving of chunks in a large scale volumemetric dataset. To be convenient, we use local file system in this tutorial. To use cloud storage, you can just setup the authentication following the documentation of CloudVolume_ and replace the path to cloud storage. You can create a volume and ingest the image stack to the volume::

   chunkflow read-h5 --file-name sample_A_20160501.hdf --dataset-path /volumes/raw save --volume-path file:///tmp/your/key/path 

Now you can cutout the image chunk from the volume::

   chunkflow cutout --volume-path file:///tmp/your/key/path --start 0 0 0 --stop 128 512 512 write-h5 --file-name /tmp/cutout_chunk.h5

.. _CloudVolume: https://github.com/seung-lab/cloud-volume


Evaluation of Segmentation
==========================
You can read two segmentation volumes and compare them::

   chunkflow read-tif --file-name groundtruth.tif -o gt read-tif --file-name /tmp/segmentation.tif -o seg evaluate-segmentation -g gt -s seg

The result will print out in terminal::

   Rand split: 0.958776
   Rand merge: 0.978602
   VOI split: 0.300484
   VOI merge: 0.143631

Of course, you can replace the `read-tif` operator to other reading operators, such as `read-h5` and `cutout`.


Convolutional Network Inference
================================
Given a trained convolution network model, it can process small patches of image and output a map, such as synapse cleft or boundary map. Due to the missing context around patch boundary, we normally need to reweight the patch. We trust the central region more and trust the marginal region less. The `inference` operator performs reweighting of patches and blend them together automatically, so the input chunk size can be arbitrary without patch alignment. The only restriction is the RAM size. After blending, the output chunk will looks like a single patch and could be used for further processing.

Synapse cleft detection
------------------------
With only one command, you can perform the inference to produce cleft map and visualize it::

   chunkflow read-tif -f "$IMAGE_PATH" -o image inference --convnet-model model.py --convnet-weight-path weight.chkpt --patch-size 18 192 192 --patch-overlap 4 64 64 --framework pytorch --batch-size 6 --bump wu --num-output-channels 1 --mask-output-chunk -i image -o cleft write-tif -i cleft -f cleft.tif neuroglancer -c image,cleft -p 33333 -v 30 6 6

You can see the image with output synapse cleft map:

|cleft|

.. |cleft| image:: _static/image/cleft.png


Distributed Computation
************************

Deploy in Local Computers
===========================

Deploy to Kubernetes Cluster in Cloud
======================================
