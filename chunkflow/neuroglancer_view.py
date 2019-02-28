import neuroglancer


def neuroglancer_view(chunks, voxel_size=(1,1,1)):
    """
    chunks: (list/tuple) multiple chunks 
    """
    viewer = neuroglancer.Viewer()

    with viewer.txn() as s:
        for idx, chunk in enumerate(chunks):
            s.layers.append(
                name='chunk-{}'.format(idx),
                layer=neuroglancer.LocalVolume(
                    data=chunk,
                    voxel_size=voxel_size,
                    # offset is in nm, not voxels
                    offset=list(o*v for o, v in zip(
                        chunk.global_offset[-3:], voxel_size)),
                ),
                shader=get_shader(chunk),
            )
    print('Open this url in browser: ')
    print(viewer)
    input('Press Enter to exit neuroglancer.')


def get_shader(chunk):
    if chunk.ndim==3:
        # this is a image
        return """void main() {
    emitGrayscale(toNormalized(getDataValue()));
}"""
    elif chunk.ndim==4 and chunk.shape[0]==3:
        # this is affinitymap
        return """void main() {
    emitRGB(vec3(toNormalized(getDataValue(0)),
                 toNormalized(getDataValue(1)),
                 toNormalized(getDataValue(2))));
}"""
    else:
        raise ValueError('only support image and affinitymap now.')

