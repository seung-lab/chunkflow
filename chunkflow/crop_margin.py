def crop_margin(chunk, output_bbox=None, margin_size=None, verbose=True):
    if margin_size:
        if chunk.ndim==3:
            return chunk[margin_size[0]:chunk.shape[1]-margin_size[0],
                         margin_size[1]:chunk.shape[2]-margin_size[1],
                         margin_size[2]:chunk.shape[3]-margin_size[2]]
        elif chunk.ndim==4:
            return chunk[:, 
                         margin_size[0]:chunk.shape[1]-margin_size[0],
                         margin_size[1]:chunk.shape[2]-margin_size[1],
                         margin_size[2]:chunk.shape[3]-margin_size[2]]
        else:
            raise ValueError('the array dimension can only by 3 or 4.')
    else:
        if verbose:
            print('automatically crop the chunk to output bounding box.')
        return chunk.cutout(output_bbox.to_slices())


