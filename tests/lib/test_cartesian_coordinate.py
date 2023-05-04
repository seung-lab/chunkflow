import os

import numpy as np

from cloudvolume.lib import Bbox, Vec

from chunkflow.lib.cartesian_coordinate import BoundingBox, Cartesian, to_cartesian, BoundingBoxes, PhysicalBoudingBox


def test_cartesian():
    assert to_cartesian(None) == None
    ct = (1,2,3)
    assert to_cartesian(ct) == Cartesian(1,2,3)

    ct = Cartesian(1,2,3)
    ct += 2
    assert ct == Cartesian(3,4,5)

    ct -= 2
    assert ct == Cartesian(1,2,3)

    np.testing.assert_equal(ct.vec, Vec(1,2,3))

    ct = Cartesian(3,4,5)
    ct = ct // 2
    assert ct == Cartesian(1,2,2)

    # note that 2*ct will repeat the elements of ct!
    ct2 = ct*2
    assert ct2 > ct
    assert ct2 >= ct
    assert ct < ct2
    assert ct <= ct2

    ct3 = ct / 2
    assert ct3 == Cartesian(0.5, 1, 1)

    ct4 = Cartesian.from_collection((1,2,3))
    assert ct4 == Cartesian(1, 2, 3) 

    assert Cartesian(0, 0, 0)*Cartesian(1,2,3) == Cartesian(0, 0, 0)

    assert Cartesian(4,6,8) / Cartesian(2,3,2) == Cartesian(2,2,4)

    assert -Cartesian(1,-2,3) == Cartesian(-1, 2, -3)

    assert Cartesian(1,2,3).tuple == (1,2,3)
    assert Cartesian(1,2,3).vec is not None

def test_bounding_box():
    bbox = BoundingBox.from_string('3166-3766_7531-8131_2440-3040')
    bbox == BoundingBox(Cartesian(3166, 7531, 2440), Cartesian(3766, 8131, 3040))
    
    bbox = BoundingBox.from_string('Sp1,3166-3766_7531-8131_2440-3040.h5')
    bbox == BoundingBox(Cartesian(3166, 7531, 2440), Cartesian(3766, 8131, 3040))

    bbox = Bbox.from_delta((1,3,2), (64, 32, 8))
    bbox = BoundingBox.from_bbox(bbox)
    assert bbox.start == Cartesian(1,3,2)
    assert bbox.stop == Cartesian(65, 35, 10)

    bbox = bbox.clone()
    assert isinstance(bbox, BoundingBox)

    minpt = Cartesian(1,2,3)
    maxpt = Cartesian(2,3,4)
    bbox = BoundingBox(minpt, maxpt)

    bbox = BoundingBox.from_center(Cartesian(1,2,3), 3)
    assert bbox == BoundingBox.from_list([-2, -1, 0, 4, 5, 6])
    
    bbox = BoundingBox.from_center(Cartesian(1,2,3), 3, even_size=False)
    assert bbox == BoundingBox.from_list([-2, -1, 0, 5, 6, 7])

    bbox1 = BoundingBox.from_list([0,1,2, 2,3,4])
    bbox2 = BoundingBox.from_list([1,2,3, 3,4,5])
    assert bbox1.union(bbox2) == BoundingBox.from_list([0,1,2, 3,4,5])
    assert bbox1.intersection(bbox2) == BoundingBox.from_list([1,2,3, 2,3,4])


def test_bounding_boxes():
    fname = os.path.join(os.path.dirname(__file__), 'sp3_bboxes.txt')
    bboxes = BoundingBoxes.from_file(fname)
    fname = os.path.join(os.path.dirname(__file__), 'sp3_bboxes.npy')
    bboxes.to_file(fname)
    os.remove(fname)
    

def test_physical_bounding_box():
    start = Cartesian(0, 1, 2)
    stop  = Cartesian(2, 3, 4)
    voxel_size = Cartesian(2, 2, 2)
    pbbox = PhysicalBoudingBox(start, stop, voxel_size)

    pbbox2 = pbbox.to_other_voxel_size(Cartesian(1,1,1))
    assert pbbox2.start == Cartesian(0,2,4)