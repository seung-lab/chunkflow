# Shang Mu, January 2019
# No stability consideration is accounted for in implementing this.
# TODO: Need unit tests
# TODO: use the out=None convention of numpy for in-place edits vs
# new copy of array

import numpy as np
from chunkflow.chunk import Chunk


def window_level(img, half_window, level):
    r'''
    Pixel values equal to level will become 0, level±window/2 will become ±1.
    Mathematically equivalent of rescale(img, level+np.array([-1,1])*half_window, [-1,1])
    '''
    if half_window <= 0:
        raise ValueError('invalid value.')
    #img = np.copy(img)
    img -= level
    img *= 1. / half_window
    return img


def rescale(img, old_range, new_range=[-1, 1]):
    r'''
    Linearly remap pixel values within old_range to the new_range.
    For example, from values between [0,1] to values between [-1,1].
    '''
    if np.array_equal(old_range, new_range):  # is this even fast in python?
        return img

    img -= old_range[0]
    img *= (new_range[1] - new_range[0]) / (old_range[1] - old_range[0])
    img += new_range[0]
    return img


def get_voxels_for_stats(img, min_max_invalid=[True, True], debug=False):

    min_invalid, max_invalid = min_max_invalid
    #TODO clip_percentile = [None,None]
    mask = True
    if min_invalid:
        mi = np.min(img)
        mask = img != mi
    if max_invalid:
        ma = np.max(img)
        mask = np.logical_and(mask, img != ma)

    #if quantile

    if mask is True:
        stat_img = img
    else:
        stat_img = img[mask]

    if debug and min_invalid and max_invalid:
        print('rawminmax =', mi, ma)

    return stat_img


def normalize(img,
              method,
              target_scale=[-1, 1],
              min_max_invalid=[True, True],
              invalid_values=[],
              clip_percentile=[None, None],
              do_clipping=False,
              make_copy=True,
              debug=False):
    r'''
    *Assuming floating point voxel values.*
    '''
    stat_img = get_voxels_for_stats(img,
                                    min_max_invalid=min_max_invalid,
                                    debug=debug)

    if make_copy:
        img = np.copy(img)

    if stat_img.size == 0:
        return img

    if method == 1 or method == 'meanstd':
        sd = np.std(stat_img)
        if sd > 0:
            img -= np.mean(stat_img)
            img /= sd
        if debug:
            print('sd=', sd, 'mean=', np.mean(stat_img))
            if do_clipping:
                img = np.clip(img, -2, 2, img)  # 2*std

    elif method == 2 or method == 'fill':

        mi = np.min(stat_img)
        ma = np.max(stat_img)
        if debug:
            print('minmax =', mi, ma)
        img = rescale(img, [mi, ma], new_range=target_scale)

        if do_clipping:
            img = np.clip(img, *target_scale, img)

    return img


def adjust_gamma(img, gamma, auto_rescale=False):
    r'''
    Floating point images only.
    Assuming pixel values in the range of [0,1]; values out of the range will be clipped.
    Otherwise setting auto_rescale=True will rescale image to [0,1] (or make no change
     if and only if the image has uniform value all across) before applying gamma adjustment.
    '''

    #img = np.copy(img)
    if auto_rescale:
        mi = np.min(img)
        ma = np.max(img)
        if mi != ma:
            img -= mi
            img /= ma - mi

    img = np.clip(img, 0, 1, img)
    img **= gamma

    return img


def grey_augment(img,
                 max_level_change=0.15,
                 max_window_change=0.15,
                 max_log2gamma_change=1.0,
                 level_prob=1.0,
                 window_prob=0.8,
                 gamma_prob=0.3,
                 value_range=[-1, 1],
                 make_copy=True):
    r"""
    Performs grey value (histogram) augmentation on img.
    Gamma adjustment is applied after adjusting window/level.
    value_range: (Expected) range of pixel values of the input image and this is also
         what the output image will conform to. The window/level of the pre-augmentation
         input image are inferred from this. Values outside the range in the output will be clipped.
    """
    if make_copy:
        img = np.copy(img)

    change_level = np.random.rand() < level_prob
    change_window = np.random.rand() < window_prob
    change_gamma = np.random.rand() < gamma_prob

    level = (value_range[0] + value_range[1]) / 2
    half_window = (value_range[1] - value_range[0]) / 2
    log2gamma = 0

    if change_level:
        level += 2 * (np.random.rand() - 0.5) * max_level_change
    if change_window:
        half_window += 2 * (np.random.rand() - 0.5) * max_window_change / 2
    if change_gamma:
        log2gamma += 2 * (np.random.rand() - 0.5) * max_log2gamma_change

    if change_level or change_window or change_gamma:
        if change_gamma:
            target_range = [0, 1]
            # gamma adjustment only makes sense in [0,1] range in the conventional sense
        else:
            target_range = value_range

        # level/window adjustement using rescale()
        img = rescale(img, level + np.array([-1, 1]) * half_window,
                      target_range)
        img = np.clip(img, *target_range, img)

        if change_gamma:  # perhaps gamma should always happen before window/level on the raw full range of values?
            img = adjust_gamma(img, 2**log2gamma)

            # rescale back to original/desired pixel value representation
            img = rescale(img, [0, 1], value_range)

    return img


def normalize_section_shang(image: np.ndarray, nominalmin: float,
                            nominalmax: float, clipvalues: bool):
    """
    Parameters
    ------------
    image:
        image volume.
    nominalmin:
        min threshold
    nominalmax:
        max threshold
    clipvalues:
        clip values or not.
    """
    assert nominalmin < nominalmax
    assert image.ndim == 3
    global_offset = image.global_offset
    originaltype = image.dtype
    arr = image.astype(np.float32)

    # number of bits per voxel
    nbits = np.dtype(originaltype).itemsize * 8
    default_nominalmax = float(2**nbits - 1)

    nominalmin = nominalmin if nominalmin is not None else 0.0
    nominalmax = nominalmax if nominalmax is not None else default_nominalmax

    normalization = 'fill'

    # stack/chunk-wise normalization first if necessary (for blank slices within a valid stack)
    #arr = normalize(arr, normalization, target_scale = [-1,1], min_max_invalid = [True]*2, make_copy=False)

    # slice-wise normalization
    # Note in chunkflow the first dim is z/slice
    for ii in range(arr.shape[0]):
        normalize(arr[ii, :, :],
                  normalization,
                  target_scale=[nominalmin, nominalmax],
                  min_max_invalid=[True, True],
                  do_clipping=clipvalues,
                  make_copy=False)

    # cast to original data type if necessary
    #arr = np.round(arr)
    #arr = arr.astype(originaltype)

    return Chunk(arr, global_offset=global_offset)


def test1_grey_augment():
    from numpy.random import randint
    x = np.random.rand(*randint(200, size=randint(5)))
    print(x.shape)
    y = grey_augment(x, value_range=[0, 1])
    ind = np.where(np.logical_and(y > 0, y < 1))
    return np.array_equal(np.argsort(x[ind]), np.argsort(y[ind]))


if __name__ == "__main__":
    import sys
    import glob
    import tifffile
    import ntpath
    import os

    if len(sys.argv) > 1:
        if sys.argv[1] == 'normalize':
            pattern, out = sys.argv[2:4]

            #all_tiffs = glob.glob("/omniData/TracerTasks/pinky/cell_bodies/BasilData2/RawData/*.tif")
            all_tiffs = sorted(glob.glob(pattern))
            #print(all_tiffs)

            for image_file in all_tiffs:
                imgs = tifffile.imread(image_file).astype(np.float32)
                print(image_file)
                #print(imgs.shape)
                if len(imgs.shape) == 2:
                    imgs = imgs[..., None]

                if 0:
                    normalization = 'meanstd'
                    old_range = (-2, 2)
                elif 1:
                    normalization = 'fill'
                    old_range = (-1, 1)
                if 1:  # stack-wise normalization
                    imgs = normalize(imgs,
                                     normalization,
                                     target_scale=[-1, 1],
                                     min_max_invalid=[True] * 2,
                                     make_copy=False,
                                     debug=True,
                                     do_clipping=True)
                else:  # slice-wise normalization
                    for ii in range(imgs.shape[2]):
                        normalize(imgs[..., ii],
                                  normalization,
                                  target_scale=[-1, 1],
                                  min_max_invalid=[True, True],
                                  make_copy=False,
                                  debug=True,
                                  do_clipping=True)

                imgs = rescale(imgs, old_range=old_range,
                               new_range=[0, 255]).astype(np.uint8)
                fname = out + '/' + ntpath.basename(image_file) + '_norm.tif'

                if os.path.exists(fname):
                    raise IOError('file already exist')
                else:
                    tifffile.imwrite(fname, data=imgs)
