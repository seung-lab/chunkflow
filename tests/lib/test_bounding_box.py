from cloudvolume.lib import Bbox, Vec

from chunkflow.lib.bounding_boxes import BoundingBox, Cartesian


def test_cartesian():
    ct = Cartesian(1,2,3)
    ct += 2
    ct == Cartesian(3,4,5)

    ct -= 2
    ct == Cartesian(1,2,3)

    ct.vec == Vec(1,2,3)

def test_bounding_box():
    bbox = Bbox.from_delta((1,3,2), (64, 32, 8))
    bbox = BoundingBox.from_bbox(bbox)

    bbox = bbox.clone()
    assert isinstance(bbox, BoundingBox)

    minpt = Cartesian(1,2,3)
    maxpt = Cartesian(2,3,4)
    bbox = BoundingBox.from_corners(minpt, maxpt)

    bbox = BoundingBox.from_center(Cartesian(1,2,3), 3)
    bbox == BoundingBox.from_list([-2, -1, 0, 4, 5, 6])

    