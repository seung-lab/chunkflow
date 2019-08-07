import numpy as np
from warnings import warn

from skimage.feature import match_template


def validate_by_template_matching(img: np.ndarray, verbose: bool = True):
    """ Detect 3d black boxes by template matching.
    1. binarize the image. the voxels inside the black box will be false, and the outside will be true
    2. The template is 7x7x2 with one section true and the other false. 
    3. sliding the template through the array, and detect the matching regions. 
    4. rotate the template to be 7x2x7 and 2x7x7, do the same detection.
    5. if we can find multiple matchings in all the x,y,z direction, there is probably a black box. 
    Note that this is always effective. If the black box is large enough to reach both sides, 
    the detection will fail.

    Parameters
    -----------
    img:
        3D image volume.
    verbose:
        print out debuging info or not.
    """
    if verbose:
        print("validation by template matching...")

    if np.issubdtype(img.dtype, np.floating):
        warn(
            'do not support image with floating data type, will skip the validation.'
        )
        return True

    img = img.astype(dtype=np.bool)

    score_threshold = 0.9
    num_threshold = 100
    evidence_point = 0

    temp = np.zeros((7, 7, 2), dtype=np.bool)
    temp[:, :, 0] = True
    result = match_template(img, temp)
    if np.count_nonzero(result > score_threshold) > num_threshold:
        evidence_point += 1

    temp = np.zeros((7, 7, 2), dtype=np.bool)
    temp[:, :, 1] = True
    result = match_template(img, temp)
    if np.count_nonzero(result > score_threshold) > num_threshold:
        evidence_point += 1

    temp = np.zeros((2, 7, 7), dtype=np.bool)
    temp[0, :, :] = True
    result = match_template(img, temp)
    if np.count_nonzero(result > score_threshold) > num_threshold:
        evidence_point += 1

    temp = np.zeros((2, 7, 7), dtype=np.bool)
    temp[1, :, :] = True
    result = match_template(img, temp)
    if np.count_nonzero(result > score_threshold) > num_threshold:
        evidence_point += 1

    temp = np.zeros((7, 2, 7), dtype=np.bool)
    temp[:, 0, :] = True
    result = match_template(img, temp)
    if np.count_nonzero(result > score_threshold) > num_threshold:
        evidence_point += 1

    temp = np.zeros((7, 2, 7), dtype=np.bool)
    temp[:, 1, :] = True
    result = match_template(img, temp)
    if np.count_nonzero(result > score_threshold) > num_threshold:
        evidence_point += 1

    if evidence_point > 4:
        return False
    else:
        return True
