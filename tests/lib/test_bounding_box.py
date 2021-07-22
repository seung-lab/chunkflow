from cloudvolume.lib import Bbox

from chunkflow.lib.bounding_boxes import BoundingBox


def test_bounding_box():
    bbox = Bbox.from_delta((1,3,2), (64, 32, 8))
    bbox = BoundingBox.from_bbox(bbox)

    bbox = bbox.clone()
    assert isinstance(bbox, BoundingBox)

    