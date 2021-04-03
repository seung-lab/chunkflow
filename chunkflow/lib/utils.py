from cloudvolume.lib import Bbox, Vec


def coordinates2bbox(start: tuple, size: tuple=None, stop: tuple=None):
    # use bounding box of volume
    start = Vec(*start)

    if size is None:
        assert stop is not None
        size = stop - start
    else:
        size = Vec(*size)
    return Bbox.from_delta(start, size)