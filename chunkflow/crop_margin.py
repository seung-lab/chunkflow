from .lib.offset_array import OffsetArray 


def crop_margin(chunk, output_bbox=None, margin_size=None, verbose=True):
    if margin_size:
        assert len(margin_size) == chunk.ndim
        global_offset = chunk.global_offset
        if chunk.ndim == 3:
            chunk = chunk[margin_size[0]:chunk.shape[0]-margin_size[0],
                          margin_size[1]:chunk.shape[1]-margin_size[1],
                          margin_size[2]:chunk.shape[2]-margin_size[2]]
        elif chunk.ndim == 4:
            chunk = chunk[:,
                          margin_size[1]:chunk.shape[1]-margin_size[1],
                          margin_size[2]:chunk.shape[2]-margin_size[2],
                          margin_size[3]:chunk.shape[3]-margin_size[3]]
        else:
            raise ValueError('the array dimension can only by 3 or 4.')
        global_offset = tuple(o+m for o, m in zip(global_offset, margin_size))
        return OffsetArray(chunk, global_offset=global_offset)
    else:
        if verbose:
            print('automatically crop the chunk to output bounding box.')
        return chunk.cutout(output_bbox.to_slices())


