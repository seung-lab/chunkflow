from chunkflow.lib.bounding_boxes import Cartesian
from chunkflow.lib.region_of_interest import RegionOfInterest, ROITree


def test_roi():
    voxel_start = Cartesian(1,2,3)
    voxel_stop = Cartesian(40,50,60)
    voxel_size = Cartesian(128, 128, 128)
    roi = RegionOfInterest(voxel_start, voxel_stop, voxel_size)

    assert roi.voxel_shape == Cartesian(39,48,57)


    # test roitree
    factor = Cartesian(2,2,2)
    leaf_block_size = Cartesian(3,3,3)
    leaf_voxel_size = Cartesian(4,4,4)

    roitree = ROITree.from_root_roi(
        roi, factor, leaf_block_size, leaf_voxel_size)