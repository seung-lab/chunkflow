import numpy as np

from cloudvolume.lib import Bbox, Vec

from chunkflow.lib.bounding_boxes import BoundingBox, Coordinate, to_cartesian


def test_cartesian():
    assert to_cartesian(None) == None
    ct = (1,2,3)
    assert to_cartesian(ct) == Coordinate(1,2,3)

    ct = Coordinate(1,2,3)
    ct += 2
    assert ct == Coordinate(3,4,5)

    ct -= 2
    assert ct == Coordinate(1,2,3)

    np.testing.assert_equal(ct.vec, Vec(1,2,3))

    ct = Coordinate(3,4,5)
    ct = ct // 2
    assert ct == Coordinate(1,2,2)

    # note that 2*ct will repeat the elements of ct!
    ct2 = ct*2
    assert ct2 > ct
    assert ct2 >= ct
    assert ct < ct2
    assert ct <= ct2

    ct3 = ct / 2
    assert ct3 == Coordinate(0.5, 1, 1)

    ct4 = Coordinate.from_collection((1,2,3))
    assert ct4 == Coordinate(1, 2, 3) 

    assert Coordinate(0, 0, 0)*Coordinate(1,2,3) == Coordinate(0, 0, 0)

    assert Coordinate(4,6,8) / Coordinate(2,3,2) == Coordinate(2,2,4)

    assert -Coordinate(1,-2,3) == Coordinate(-1, 2, -3)

def test_bounding_box():
    bbox = Bbox.from_delta((1,3,2), (64, 32, 8))
    bbox = BoundingBox.from_bbox(bbox)
    assert bbox.start == Coordinate(1,3,2)
    assert bbox.stop == Coordinate(65, 35, 10)

    bbox = bbox.clone()
    assert isinstance(bbox, BoundingBox)

    minpt = Coordinate(1,2,3)
    maxpt = Coordinate(2,3,4)
    bbox = BoundingBox(minpt, maxpt)

    bbox = BoundingBox.from_center(Coordinate(1,2,3), 3)
    assert bbox == BoundingBox.from_list([-2, -1, 0, 4, 5, 6])
    
    bbox = BoundingBox.from_center(Coordinate(1,2,3), 3, even_size=False)
    assert bbox == BoundingBox.from_list([-2, -1, 0, 5, 6, 7])


    