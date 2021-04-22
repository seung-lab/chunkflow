# coding=utf-8
"""
Note that this code is copied from [gala](https://github.com/janelia-flyem/gala)
The license is BSD-like [Janelia Farm license](http://janelia-flyem.github.io/janelia_farm_license.html)
"""

import numpy as np
import multiprocessing
import itertools as it
import collections as coll
from functools import partial
import logging
import h5py
import scipy.ndimage as nd
import scipy.sparse as sparse
from skimage.segmentation import relabel_sequential
from scipy.ndimage.measurements import label
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import precision_recall_curve


def nzcol(mat, row_idx):
    """Return the nonzero elements of given row in a CSR matrix.

    Parameters
    ----------
    mat : CSR matrix
        Input matrix.
    row_idx : int
        The index of the row (if `mat` is CSR) for which the nonzero
        elements are desired.

    Returns
    -------
    nz : array of int
        The location of nonzero elements of `mat[main_axis_idx]`.

    Examples
    --------
    >>> mat = sparse.csr_matrix(np.array([[0, 1, 0, 0], [0, 5, 8, 0]]))
    >>> nzcol(mat, 1)
    array([1, 2], dtype=int32)
    >>> mat[1, 2] = 0
    >>> nzcol(mat, 1)
    array([1], dtype=int32)
    """
    return mat[row_idx].nonzero()[1]


def pixel_wise_boundary_precision_recall(pred, gt):
    """Evaluate voxel prediction accuracy against a ground truth.

    Parameters
    ----------
    pred : np.ndarray of int or bool, arbitrary shape
        The voxel-wise discrete prediction. 1 for boundary, 0 for non-boundary.
    gt : np.ndarray of int or bool, same shape as `pred`
        The ground truth boundary voxels. 1 for boundary, 0 for non-boundary.

    Returns
    -------
    pr : float
    rec : float
        The precision and recall values associated with the prediction.

    Notes
    -----
    Precision is defined as "True Positives / Total Positive Calls", and
    Recall is defined as "True Positives / Total Positives in Ground Truth".

    This function only calculates this value for discretized predictions,
    i.e. it does not work with continuous prediction confidence values.
    """
    tp = float((gt * pred).sum())
    fp = (pred * (1-gt)).sum()
    fn = (gt * (1-pred)).sum()
    return tp/(tp+fp), tp/(tp+fn)


def wiggle_room_precision_recall(pred, boundary, margin=2, connectivity=1):
    """Voxel-wise, continuous value precision recall curve allowing drift.

    Voxel-wise precision recall evaluates predictions against a ground truth.
    Wiggle-room precision recall (WRPR, "warper") allows calls from nearby
    voxels to be counted as correct. Specifically, if a voxel is predicted to
    be a boundary within a dilation distance of `margin` (distance defined
    according to `connectivity`) of a true boundary voxel, it will be counted
    as a True Positive in the Precision, and vice-versa for the Recall.

    Parameters
    ----------
    pred : np.ndarray of float, arbitrary shape
        The prediction values, expressed as probability of observing a boundary
        (i.e. a voxel with label 1).
    boundary : np.ndarray of int, same shape as pred
        The true boundary map. 1 indicates boundary, 0 indicates non-boundary.
    margin : int, optional
        The number of dilations that define the margin. default: 2.
    connectivity : {1, ..., pred.ndim}, optional
        The morphological voxel connectivity (defined as in SciPy) for the
        dilation step.

    Returns
    -------
    ts, pred, rec : np.ndarray of float, shape `(len(np.unique(pred)+1),)`
        The prediction value thresholds corresponding to each precision and
        recall value, the precision values, and the recall values.
    """
    struct = nd.generate_binary_structure(boundary.ndim, connectivity)
    gtd = nd.binary_dilation(boundary, struct, margin)
    struct_m = nd.iterate_structure(struct, margin)
    pred_dil = nd.grey_dilation(pred, footprint=struct_m)
    missing = np.setdiff1d(np.unique(pred), np.unique(pred_dil))
    for m in missing:
        pred_dil.ravel()[np.flatnonzero(pred==m)[0]] = m
    prec, _, ts = precision_recall_curve(gtd.ravel(), pred.ravel())
    _, rec, _ = precision_recall_curve(boundary.ravel(), pred_dil.ravel())
    return list(zip(ts, prec, rec))


def get_stratified_sample(ar, n):
    """Get a regularly-spaced sample of the unique values of an array.

    Parameters
    ----------
    ar : np.ndarray, arbitrary shape and type
        The input array.
    n : int
        The desired sample size.

    Returns
    -------
    u : np.ndarray, shape approximately (n,)

    Notes
    -----
    If `len(np.unique(ar)) <= 2*n`, all the values of `ar` are returned. The
    requested sample size is taken as an approximate lower bound.

    Examples
    --------
    >>> ar = np.array([[0, 4, 1, 3],
    ...                [4, 1, 3, 5],
    ...                [3, 5, 2, 1]])
    >>> np.unique(ar)
    array([0, 1, 2, 3, 4, 5])
    >>> get_stratified_sample(ar, 3)
    array([0, 2, 4])
    """
    u = np.unique(ar)
    nu = len(u)
    if nu < 2*n:
        return u
    else:
        step = nu // n
        return u[0:nu:step]


def edit_distance(aseg, gt, size_threshold=1000, sp=None):
    """Find the number of splits and merges needed to convert `aseg` to `gt`.

    Parameters
    ----------
    aseg : np.ndarray, int type, arbitrary shape
        The candidate automatic segmentation being evaluated.
    gt : np.ndarray, int type, same shape as `aseg`
        The ground truth segmentation.
    size_threshold : int or float, optional
        Ignore splits or merges smaller than this number of voxels.
    sp : np.ndarray, int type, same shape as `aseg`, optional
        A superpixel map. If provided, compute the edit distance to the best
        possible agglomeration of `sp` to `gt`, rather than to `gt` itself.

    Returns
    -------
    (false_merges, false_splits) : float
        The number of splits and merges needed to convert aseg to gt.
    """
    if sp is None:
        return raw_edit_distance(aseg, gt, size_threshold)
    else:
        from . import agglo
        bps = agglo.best_possible_segmentation(sp, gt)
        return raw_edit_distance(aseg, bps, size_threshold)


def raw_edit_distance(aseg, gt, size_threshold=1000):
    """Compute the edit distance between two segmentations.

    Parameters
    ----------
    aseg : np.ndarray, int type, arbitrary shape
        The candidate automatic segmentation.
    gt : np.ndarray, int type, same shape as `aseg`
        The ground truth segmentation.
    size_threshold : int or float, optional
        Ignore splits or merges smaller than this number of voxels.

    Returns
    -------
    (false_merges, false_splits) : float
        The number of splits and merges required to convert aseg to gt.
    """
    aseg = relabel_sequential(aseg)[0]
    gt = relabel_sequential(gt)[0]
    r = contingency_table(aseg, gt, ignore_seg=[0], ignore_gt=[0], norm=False)
    r.data[r.data <= size_threshold] = 0
    # make each segment overlap count for 1, since it will be one
    # operation to fix (split or merge)
    r.data[r.data.nonzero()] /= r.data[r.data.nonzero()]
    false_splits = (r.sum(axis=0)-1)[1:].sum()
    false_merges = (r.sum(axis=1)-1)[1:].sum()
    return (false_merges, false_splits)


def contingency_table(seg, gt, *, ignore_seg=(), ignore_gt=(), norm=True):
    """Return the contingency table for all regions in matched segmentations.

    Parameters
    ----------
    seg : np.ndarray, int type, arbitrary shape
        A candidate segmentation.
    gt : np.ndarray, int type, same shape as `seg`
        The ground truth segmentation.
    ignore_seg : iterable of int, optional
        Values to ignore in `seg`. Voxels in `seg` having a value in this list
        will not contribute to the contingency table. (default: [0])
    ignore_gt : iterable of int, optional
        Values to ignore in `gt`. Voxels in `gt` having a value in this list
        will not contribute to the contingency table. (default: [0])
    norm : bool, optional
        Whether to normalize the table so that it sums to 1.

    Returns
    -------
    cont : scipy.sparse.csr_matrix
        A contingency table. `cont[i, j]` will equal the number of voxels
        labeled `i` in `seg` and `j` in `gt`. (Or the proportion of such voxels
        if `norm=True`.)
    """
    segr = seg.ravel()
    gtr = gt.ravel()
    ignored = np.zeros(segr.shape, np.bool)
    data = np.ones(gtr.shape)
    for i in ignore_seg:
        ignored[segr == i] = True
    for j in ignore_gt:
        ignored[gtr == j] = True
    data[ignored] = 0
    cont = sparse.coo_matrix((data, (segr, gtr))).tocsr()
    if norm:
        cont /= cont.sum()
    return cont


def assignment_table(seg_or_ctable, gt=None, *, dtype=np.bool_):
    """Create an assignment table of value in `seg` to `gt`.

    Parameters
    ----------
    seg_or_ctable : array of int, or 2D array of float
        The segmentation to assign. Every value in `seg` will be
        assigned to a single value in `gt`.
        Alternatively, pass a single, pre-computed contingency table
        to be converted to an assignment table.
    gt : array of int, same shape as seg
        The segmentation to assign to. Don't pass if `seg_or_cont` is
        a contingency matrix.
    dtype : numpy dtype specification
        The desired data type for the assignment matrix.

    Returns
    -------
    assignments : sparse matrix
        A matrix with `True` at position [i, j] if segment i in `seg`
        is assigned to segment j in `gt`.

    Examples
    --------
    >>> seg = np.array([0, 1, 1, 1, 2, 2])
    >>> gt = np.array([1, 1, 1, 2, 2, 2])
    >>> assignment_table(seg, gt).toarray()
    array([[False,  True, False],
           [False,  True, False],
           [False, False,  True]])
    >>> cont = contingency_table(seg, gt)
    >>> assignment_table(cont).toarray()
    array([[False,  True, False],
           [False,  True, False],
           [False, False,  True]])
    """
    if gt is None:
        ctable = seg_or_ctable.copy()
    else:
        ctable = contingency_table(seg_or_ctable, gt, norm=False)
    minval = _mindiff(ctable.data)
    ctable.data += np.random.randn(ctable.data.size) * 0.01 * minval
    maxes = ctable.max(axis=1).toarray()
    maxes_repeated = np.repeat(maxes, np.diff(ctable.indptr))
    assignments = sparse.csr_matrix((ctable.data == maxes_repeated,
                                     ctable.indices, ctable.indptr),
                                    dtype=dtype)
    assignments.eliminate_zeros()
    return assignments


def _mindiff(arr):
    """Compute the smallest nonzero difference between elements in arr

    Parameters
    ----------
    arr : array
        Array of *positive* numeric values.

    Returns
    -------
    mindiff : float
        The smallest nonzero difference between any two elements in arr.

    Examples
    --------
    >>> arr = np.array([5, 5, 2.5, 7, 9.2])
    >>> _mindiff(arr)
    2.0
    >>> arr = np.array([0.5, 0.5])
    >>> _mindiff(arr)
    0.5
    """
    arr = np.sort(arr)  # this *must* be a copy!
    diffs = np.diff(arr)
    diffs = diffs[diffs != 0]
    if arr[0] != 0:
        diffs = np.concatenate((diffs, [arr[0]]))
    mindiff = np.min(diffs)
    return mindiff



# note: subclassing scipy sparse matrices requires that the class name
# start with the same three letters as the given format. See:
# https://stackoverflow.com/questions/24508214/inherit-from-scipy-sparse-csr-matrix-class
# https://groups.google.com/d/msg/scipy-user/-1PIkEMFWd8/KX6idRoIqqkJ
class csrRowExpandableCSR(sparse.csr_matrix):
    """Like a scipy CSR matrix, but rows can be appended.

    Use `mat[i] = v` to append the row-vector v as row i to the matrix mat.
    Any rows between the current last row and i are filled with zeros.

    Parameters
    ----------
    arg1 :
        Any valid instantiation of a sparse.csr_matrix. This includes a
        dense matrix or 2D NumPy array, any SciPy sparse matrix, or a
        tuple of the three defining values of a scipy sparse matrix,
        (data, indices, indptr). See the documentation for
        sparse.csr_matrix for more information.
    dtype : numpy dtype specification, optional
        The data type contained in the matrix, e.g. 'float32', np.float64,
        np.complex128.
    shape : tuple of two ints, optional
        The number of rows and columns of the matrix.
    copy : bool, optional
        This argument does nothing, and is maintained for compatibility
        with the csr_matrix constructor. Because we create bigger-than-
        necessary buffer arrays, the data must always be copied.
    max_num_rows : int, optional
        The initial maximum number of rows. Note that more rows can
        always be added; this is used only for efficiency. If None,
        defaults to twice the initial number of rows.
    max_nonzero : int, optional
        The maximum number of nonzero elements. As with max_num_rows,
        this is only necessary for efficiency.
    expansion_factor : int or float, optional
        The maximum number of rows or nonzero elements will be this
        number times the initial number of rows or nonzero elements.
        This is overridden if max_num_rows or max_nonzero are provided.

    Examples
    --------
    >>> init = csrRowExpandableCSR([[0, 0, 2], [0, 4, 0]])
    >>> init[2] = np.array([9, 0, 0])
    >>> init[4] = sparse.csr_matrix([0, 0, 5])
    >>> init.nnz
    4
    >>> init.data
    array([2, 4, 9, 5])
    >>> init.toarray()
    array([[0, 0, 2],
           [0, 4, 0],
           [9, 0, 0],
           [0, 0, 0],
           [0, 0, 5]])
    """
    def __init__(self, arg1, shape=None, dtype=None, copy=False,
                 max_num_rows=None, max_nonzero=None,
                 expansion_factor=2):
        other = sparse.csr_matrix(arg1, shape=shape, dtype=dtype, copy=copy)
        if max_nonzero is None:
            max_nonzero = other.nnz * expansion_factor
        if max_num_rows is None:
            max_num_rows = other.shape[0] * expansion_factor
        self.curr_nonzero = other.nnz
        self.curr_indptr = other.shape[0] + 1
        self._data = np.empty(max_nonzero, dtype=other.dtype)
        self._indices = np.empty(max_nonzero, dtype=other.indices.dtype)
        self._indptr = np.empty(max_num_rows + 1, dtype=other.indptr.dtype)
        super().__init__((other.data, other.indices, other.indptr),
                         shape=other.shape, dtype=other.dtype, copy=False)

    @property
    def data(self):
        """The data array is virtual, truncated from the data "buffer", _data.
        """
        return self._data[:self.curr_nonzero]

    @data.setter
    def data(self, value):
        """Setter for the data property.

        We have to special-case for a few kinds of values.

        When creating a new instance, the csr_matrix class removes some
        zeros from the array and ends up setting data to a smaller array.
        In that case, we need to make sure that we reset `self.curr_nonzero`
        and copy the relevant part of the array.
        """
        if np.isscalar(value) or len(value) == self.curr_nonzero:
            self._data[:self.curr_nonzero] = value
        else:  # `value` is array-like of different length
            self.curr_nonzero = len(value)
            while self._data.size < self.curr_nonzero:
                self._double_data_and_indices()
            self._data[:self.curr_nonzero] = value

    @property
    def indices(self):
        return self._indices[:self.curr_nonzero]

    @indices.setter
    def indices(self, value):
        if np.isscalar(value) or len(value) == self.curr_nonzero:
            self._indices[:self.curr_nonzero] = value
        else:  # `value` is array-like of different length
            self.curr_nonzero = len(value)
            while self._indices.size < self.curr_nonzero:
                self._double_data_and_indices()
            self._indices[:self.curr_nonzero] = value

    @property
    def indptr(self):
        return self._indptr[:self.curr_indptr]

    @indptr.setter
    def indptr(self, value):
        if np.isscalar(value) or len(value) == self.curr_indptr:
            self._indptr[:self.curr_indptr] = value
        else:  # `value` is array-like of different length
            self.curr_indptr = len(value)
            while self._indptr.size < self.curr_indptr:
                self._double_data_and_indices()
            self._indptr[:self.curr_indptr] = value

    def __setitem__(self, index, value):
        if np.isscalar(index):
            if index >= self.shape[0]:  # appending a row
                self._append_row_at(index, value)
            else:
                if np.isscalar(value):
                    if value == 0:  # zeroing out a row
                        self._zero_row(index)
        else:
            super().__setitem__(index, value)

    def _append_row_at(self, index, value):
        # first: normalize the input value. We want a sparse CSR matrix as
        # input, to make data copying logic much simpler.
        if np.isscalar(value):
            value = np.full(self.shape[1], value)  # make a full row if scalar
        if not sparse.isspmatrix_csr(value):
            value = sparse.csr_matrix(value)

        # Make sure we have sufficient room for the new row.
        if index + 2 > self._indptr.size:
            self._double_indptr()
        num_values = value.nnz
        if self.curr_nonzero + num_values > self._data.size:
            self._double_data_and_indices()
        i, j = self.indptr[-1], self.indptr[-1] + num_values
        self._indptr[self.curr_indptr:index + 1] = i
        self._indptr[index + 1] = j
        self.curr_indptr = index + 2
        self._indices[i:j] = value.indices[:]
        self._data[i:j] = value.data[:]
        self.curr_nonzero += num_values
        # It turns out that the `shape` attribute is a property in SciPy
        # sparse matrices, and can't be set directly. So, we bypass it and
        # set the corresponding tuple directly, interfaces be damned.
        self._shape = (int(index + 1), self.shape[1])

    def _zero_row(self, index):
        """Set all elements of row `index` to 0."""
        i, j = self.indptr[index:index+2]
        self.data[i:j] = 0

    def _double_indptr(self):
        """Double the size of the array backing `indptr`.

        Doubling on demand gives amortized constant time append.
        """
        old_indptr = self._indptr
        self._indptr = np.empty(2 * old_indptr.size, old_indptr.dtype)
        self._indptr[:old_indptr.size] = old_indptr[:]

    def _double_data_and_indices(self):
        """Double size of the arrays backing `indices` and `data` attributes.

        Doubling on demand gives amortized constant time append. Since these
        two arrays are always the same size in the CSR format, they are
        doubled together in the same function.
        """
        n = self._data.size
        old_data = self._data
        self._data = np.empty(2 * n, old_data.dtype)
        self._data[:n] = old_data[:]
        old_indices = self._indices
        self._indices = np.empty(2 * n, old_indices.dtype)
        self._indices[:n] = old_indices[:]


def merge_contingency_table(a, b, ignore_seg=[0], ignore_gt=[0]):
    """A contingency table that has additional rows for merging initial rows.

    Parameters
    ----------
    a
    b
    ignore_seg
    ignore_gt

    Returns
    -------
    ct : array, shape (2M + 1, N)
    """
    ct = contingency_table(a, b,
                           ignore_seg=ignore_seg, ignore_gt=ignore_gt)
    ctout = csrRowExpandableCSR(ct)
    return ctout


def xlogx(x, out=None, in_place=False):
    """Compute x * log_2(x).

    We define 0 * log_2(0) = 0

    Parameters
    ----------
    x : np.ndarray or scipy.sparse.csc_matrix or csr_matrix
        The input array.
    out : same type as x (optional)
        If provided, use this array/matrix for the result.
    in_place : bool (optional, default False)
        Operate directly on x.

    Returns
    -------
    y : same type as x
        Result of x * log_2(x).
    """
    if in_place:
        y = x
    elif out is None:
        y = x.copy()
    else:
        y = out
    if isinstance(y, sparse.csc_matrix) or isinstance(y, sparse.csr_matrix):
        z = y.data
    else:
        z = np.asarray(y)  # ensure np.matrix converted to np.array
    nz = z.nonzero()
    z[nz] *= np.log2(z[nz])
    return y


def special_points_evaluate(eval_fct, coords, flatten=True, coord_format=True):
    """Return an evaluation function to only evaluate at special coordinates.

    Parameters
    ----------
    eval_fct : function taking at least two np.ndarray of equal shapes as args
        The function to be used for evaluation.
    coords : np.ndarray of int, shape (n_points, n_dim) or (n_points,)
        The coordinates at which to evaluate the function. The coordinates can
        either be subscript format (one index into each dimension of input
        arrays) or index format (a single index into the linear array). For
        the latter, use `flatten=False`.
    flatten : bool, optional
        Whether to flatten the coordinates (default) or leave them untouched
        (if they are already in raveled format).
    coord_format : bool, optional
        Format the coordinates to a tuple of np.ndarray as numpy expects. Set
        to False if coordinates are already in this format or flattened.

    Returns
    -------
    special_eval_fct : function taking at least two np.ndarray of equal shapes
        The returned function is the same as the above function but only
        evaluated at the coordinates specified. This can be used, for example,
        to subsample a volume, or to evaluate only whether synapses are
        correctly assigned, rather than every voxel, in a neuronal image
        volume.
    """
    if coord_format:
        coords = [coords[:, i] for i in range(coords.shape[1])]
    def special_eval_fct(x, y, *args, **kwargs):
        if flatten:
            for i in range(len(coords)):
                if coords[i][0] < 0:
                    coords[i] += x.shape[i]
            coords2 = np.ravel_multi_index(coords, x.shape)
        else:
            coords2 = coords
        sx = x.ravel()[coords2]
        sy = y.ravel()[coords2]
        return eval_fct(sx, sy, *args, **kwargs)
    return special_eval_fct


def make_synaptic_functions(fn, fcts):
    """Make evaluation functions that only evaluate at synaptic sites.

    Parameters
    ----------
    fn : string
        Filename containing synapse coordinates, in Raveler format. [1]
    fcts : function, or iterable of functions
        Functions to be converted to synaptic evaluation.

    Returns
    -------
    syn_fcts : function or iterable of functions
        Evaluation functions that will evaluate only at synaptic sites.

    Raises
    ------
    ImportError : if the `syngeo` package [2, 3] is not installed.

    References
    ----------
    [1] https://wiki.janelia.org/wiki/display/flyem/synapse+annotation+file+format
    [2] https://github.com/janelia-flyem/synapse-geometry
    [3] https://github.com/jni/synapse-geometry
    """
    from syngeo import io as synio
    synapse_coords = \
        synio.raveler_synapse_annotations_to_coords(fn, 'arrays')
    synapse_coords = np.array(list(it.chain(*synapse_coords)))
    make_function = partial(special_points_evaluate, coords=synapse_coords)
    if not isinstance(fcts, coll.Iterable):
        return make_function(fcts)
    else:
        return list(map(make_function, fcts))


def make_synaptic_vi(fn):
    """Shortcut for `make_synaptic_functions(fn, split_vi)`."""
    return make_synaptic_functions(fn, split_vi)


def vi(x, y=None, weights=np.ones(2), ignore_x=[0], ignore_y=[0]):
    """Return the variation of information metric. [1]

    VI(X, Y) = H(X | Y) + H(Y | X), where H(.|.) denotes the conditional
    entropy.

    Parameters
    ----------
    x : np.ndarray
        Label field (int type) or contingency table (float). `x` is
        interpreted as a contingency table (summing to 1.0) if and only if `y`
        is not provided.
    y : np.ndarray of int, same shape as x, optional
        A label field to compare to `x`.
    weights : np.ndarray of float, shape (2,), optional
        The weights of the conditional entropies of `x` and `y`. Equal weights
        are the default.
    ignore_x, ignore_y : list of int, optional
        Any points having a label in this list are ignored in the evaluation.
        Ignore 0-labeled points by default.

    Returns
    -------
    v : float
        The variation of information between `x` and `y`.

    References
    ----------
    [1] Meila, M. (2007). Comparing clusterings - an information based
    distance. Journal of Multivariate Analysis 98, 873-895.
    """
    return np.dot(weights, split_vi(x, y, ignore_x, ignore_y))


def split_vi(x, y=None, ignore_x=[0], ignore_y=[0]):
    """Return the symmetric conditional entropies associated with the VI.

    The variation of information is defined as VI(X,Y) = H(X|Y) + H(Y|X).
    If Y is the ground-truth segmentation, then H(Y|X) can be interpreted
    as the amount of under-segmentation of Y and H(X|Y) is then the amount
    of over-segmentation.  In other words, a perfect over-segmentation
    will have H(Y|X)=0 and a perfect under-segmentation will have H(X|Y)=0.

    If y is None, x is assumed to be a contingency table.

    Parameters
    ----------
    x : np.ndarray
        Label field (int type) or contingency table (float). `x` is
        interpreted as a contingency table (summing to 1.0) if and only if `y`
        is not provided.
    y : np.ndarray of int, same shape as x, optional
        A label field to compare to `x`.
    ignore_x, ignore_y : list of int, optional
        Any points having a label in this list are ignored in the evaluation.
        Ignore 0-labeled points by default.

    Returns
    -------
    sv : np.ndarray of float, shape (2,)
        The conditional entropies of Y|X and X|Y.

    See Also
    --------
    vi
    """
    _, _, _ , hxgy, hygx, _, _ = vi_tables(x, y, ignore_x, ignore_y)
    # false merges, false splits
    return np.array([hygx.sum(), hxgy.sum()])


def vi_pairwise_matrix(segs, split=False):
    """Compute the pairwise VI distances within a set of segmentations.

    If 'split' is set to True, two matrices are returned, one for each
    direction of the conditional entropy.

    0-labeled pixels are ignored.

    Parameters
    ----------
    segs : iterable of np.ndarray of int
        A list or iterable of segmentations. All arrays must have the same
        shape.
    split : bool, optional
        Should the split VI be returned, or just the VI itself (default)?

    Returns
    -------
    vi_sq : np.ndarray of float, shape (len(segs), len(segs))
        The distances between segmentations. If `split==False`, this is a
        symmetric square matrix of distances. Otherwise, the lower triangle
        of the output matrix is the false split distance, while the upper
        triangle is the false merge distance.
    """
    d = np.array([s.ravel() for s in segs])
    if split:
        def dmerge(x, y): return split_vi(x, y)[0]
        def dsplit(x, y): return split_vi(x, y)[1]
        merges, splits = [squareform(pdist(d, df)) for df in [dmerge, dsplit]]
        out = merges
        tri = np.tril(np.ones(splits.shape), -1).astype(bool)
        out[tri] = splits[tri]
    else:
        out = squareform(pdist(d, vi))
    return out


def split_vi_threshold(tup):
    """Compute VI with tuple input (to support multiprocessing).

    Parameters
    ----------
    tup : a tuple, (np.ndarray, np.ndarray, [int], [int], float)
        The tuple should consist of::
            - the UCM for the candidate segmentation,
            - the gold standard,
            - list of ignored labels in the segmentation,
            - list of ignored labels in the gold standard,
            - threshold to use for the UCM.

    Returns
    -------
    sv : np.ndarray of float, shape (2,)
        The undersegmentation and oversegmentation of the comparison between
        applying a threshold and connected components labeling of the first
        array, and the second array.
    """
    ucm, gt, ignore_seg, ignore_gt, t = tup
    return split_vi(label(ucm<t)[0], gt, ignore_seg, ignore_gt)


def vi_by_threshold(ucm, gt, ignore_seg=[], ignore_gt=[], npoints=None,
                                                            nprocessors=None):
    """Compute the VI at every threshold of the provided UCM.

    Parameters
    ----------
    ucm : np.ndarray of float, arbitrary shape
        The Ultrametric Contour Map, where each 0.0-region is separated by a
        boundary. Higher values of the boundary indicate more confidence in
        its presence.
    gt : np.ndarray of int, same shape as `ucm`
        The ground truth segmentation.
    ignore_seg : list of int, optional
        The labels to ignore in the segmentation of the UCM.
    ignore_gt : list of int, optional
        The labels to ignore in the ground truth.
    npoints : int, optional
        The number of thresholds to sample. By default, all thresholds are
        sampled.
    nprocessors : int, optional
        Number of processors to use for the parallel evaluation of different
        thresholds.

    Returns
    -------
    result : np.ndarray of float, shape (3, npoints)
        The evaluation of segmentation at each threshold. The rows of this
        array are:
            - the threshold used
            - the undersegmentation component of VI
            - the oversegmentation component of VI
    """
    ts = np.unique(ucm)[1:]
    if npoints is None:
        npoints = len(ts)
    if len(ts) > 2*npoints:
        ts = ts[np.arange(1, len(ts), len(ts)/npoints)]
    if nprocessors == 1: # this should avoid pickling overhead
        result = [split_vi_threshold((ucm, gt, ignore_seg, ignore_gt, t))
                for t in ts]
    else:
        p = multiprocessing.Pool(nprocessors)
        result = p.map(split_vi_threshold,
            ((ucm, gt, ignore_seg, ignore_gt, t) for t in ts))
    return np.concatenate((ts[np.newaxis, :], np.array(result).T), axis=0)


def rand_by_threshold(ucm, gt, npoints=None):
    """Compute Rand and Adjusted Rand indices for each threshold of a UCM

    Parameters
    ----------
    ucm : np.ndarray, arbitrary shape
        An Ultrametric Contour Map of region boundaries having specific
        values. Higher values indicate higher boundary probabilities.
    gt : np.ndarray, int type, same shape as ucm
        The ground truth segmentation.
    npoints : int, optional
        If provided, only compute values at npoints thresholds, rather than
        all thresholds. Useful when ucm has an extremely large number of
        unique values.

    Returns
    -------
    ris : np.ndarray of float, shape (3, len(np.unique(ucm))) or (3, npoints)
        The rand indices of the segmentation induced by thresholding and
        labeling `ucm` at different values. The 3 rows of `ris` are the values
        used for thresholding, the corresponding Rand Index at that threshold,
        and the corresponding Adjusted Rand Index at that threshold.
    """
    ts = np.unique(ucm)[1:]
    if npoints is None:
        npoints = len(ts)
    if len(ts) > 2 * npoints:
        ts = ts[np.arange(1, len(ts), len(ts) / npoints)]
    result = np.zeros((2, len(ts)))
    for i, t in enumerate(ts):
        seg = label(ucm < t)[0]
        result[0, i] = rand_index(seg, gt)
        result[1, i] = adj_rand_index(seg, gt)
    return np.concatenate((ts[np.newaxis, :], result), axis=0)

def adapted_rand_error(seg, gt, all_stats=False):
    """Compute Adapted Rand error as defined by the SNEMI3D contest [1]

    Formula is given as 1 - the maximal F-score of the Rand index
    (excluding the zero component of the original labels). Adapted
    from the SNEMI3D MATLAB script, hence the strange style.

    Parameters
    ----------
    seg : np.ndarray
        the segmentation to score, where each value is the label at that point
    gt : np.ndarray, same shape as seg
        the groundtruth to score against, where each value is a label
    all_stats : boolean, optional
        whether to also return precision and recall as a 3-tuple with rand_error

    Returns
    -------
    are : float
        The adapted Rand error; equal to $1 - \frac{2pr}{p + r}$,
        where $p$ and $r$ are the precision and recall described below.
    prec : float, optional
        The adapted Rand precision. (Only returned when `all_stats` is ``True``.)
    rec : float, optional
        The adapted Rand recall.  (Only returned when `all_stats` is ``True``.)

    References
    ----------
    [1]: http://brainiac2.mit.edu/SNEMI3D/evaluation
    """
    # segA is query, segB is truth
    segA = seg
    segB = gt

    n = segA.size

    # This is the contingency table obtained from segA and segB, we obtain
    # the marginal probabilities from the table.
    p_ij = contingency_table(segA, segB, norm=False)

    # Sum of the joint distribution squared
    sum_p_ij = p_ij.data @ p_ij.data

    # These are the axix-wise sums (np.sumaxis)
    a_i = p_ij.sum(axis=0).A.ravel()
    b_i = p_ij.sum(axis=1).A.ravel()

    # Sum of the segment labeled 'A'
    sum_a = a_i @ a_i
    # Sum of the segment labeled 'B'
    sum_b = b_i @ b_i

    # This is the new code, wherein 'n' is subtacted from the numerator
    # and the denominator.

    precision = (sum_p_ij - n)/ (sum_a - n)
    recall = (sum_p_ij - n)/ (sum_b - n)

    fscore = 2. * precision * recall / (precision + recall)
    are = 1. - fscore

    if all_stats:
        return (are, precision, recall)
    else:
        return are


def calc_entropy(split_vals, count):
    col_count = 0
    for key, val in split_vals.items():
        col_count += val
    col_prob = float(col_count) / count

    ent_val = 0
    for key, val in split_vals.items():
        val_norm = float(val)/count
        temp = (val_norm / col_prob)
        ent_val += temp * np.log2(temp)
    return -(col_prob * ent_val)


def split_vi_mem(x, y):
    x_labels = np.unique(x)
    y_labels = np.unique(y)
    x_labels0 = x_labels[x_labels != 0]
    y_labels0 = y_labels[y_labels != 0]

    x_map = {}
    y_map = {}

    for label in x_labels0:
        x_map[label] = {}

    for label in y_labels0:
        y_map[label] = {}

    x_flat = x.ravel()
    y_flat = y.ravel()

    count = 0
    print("Analyzing similarities")
    for pos in range(0,len(x_flat)):
        x_val = x_flat[pos]
        y_val = y_flat[pos]

        if x_val != 0 and y_val != 0:
            x_map[x_val].setdefault(y_val, 0)
            y_map[y_val].setdefault(x_val, 0)
            (x_map[x_val])[y_val] += 1
            (y_map[y_val])[x_val] += 1
            count += 1
    print("Finished analyzing similarities")

    x_ents = {}
    y_ents = {}
    x_sum = 0.0
    y_sum = 0.0

    for key, vals in x_map.items():
        x_ents[key] = calc_entropy(vals, count)
        x_sum += x_ents[key]

    for key, vals in y_map.items():
        y_ents[key] = calc_entropy(vals, count)
        y_sum += y_ents[key]

    x_s = sorted(x_ents.items(), key=lambda x: x[1], reverse=True)
    y_s = sorted(y_ents.items(), key=lambda x: x[1], reverse=True)
    x_sorted = [ pair[0] for pair in x_s ]
    y_sorted = [ pair[0] for pair in y_s ]

    return x_sum, y_sum, x_sorted, x_ents, y_sorted, y_ents


def divide_rows(matrix, column, in_place=False):
    """Divide each row of `matrix` by the corresponding element in `column`.

    The result is as follows: out[i, j] = matrix[i, j] / column[i]

    Parameters
    ----------
    matrix : np.ndarray, scipy.sparse.csc_matrix or csr_matrix, shape (M, N)
        The input matrix.
    column : a 1D np.ndarray, shape (M,)
        The column dividing `matrix`.
    in_place : bool (optional, default False)
        Do the computation in-place.

    Returns
    -------
    out : same type as `matrix`
        The result of the row-wise division.
    """
    if in_place:
        out = matrix
    else:
        out = matrix.copy()
    if type(out) in [sparse.csc_matrix, sparse.csr_matrix]:
        if type(out) == sparse.csr_matrix:
            convert_to_csr = True
            out = out.tocsc()
        else:
            convert_to_csr = False
        column_repeated = np.take(column, out.indices)
        nz = out.data.nonzero()
        out.data[nz] /= column_repeated[nz]
        if convert_to_csr:
            out = out.tocsr()
    else:
        out /= column[:, np.newaxis]
    return out


def divide_columns(matrix, row, in_place=False):
    """Divide each column of `matrix` by the corresponding element in `row`.

    The result is as follows: out[i, j] = matrix[i, j] / row[j]

    Parameters
    ----------
    matrix : np.ndarray, scipy.sparse.csc_matrix or csr_matrix, shape (M, N)
        The input matrix.
    column : a 1D np.ndarray, shape (N,)
        The row dividing `matrix`.
    in_place : bool (optional, default False)
        Do the computation in-place.

    Returns
    -------
    out : same type as `matrix`
        The result of the row-wise division.
    """
    if in_place:
        out = matrix
    else:
        out = matrix.copy()
    if type(out) in [sparse.csc_matrix, sparse.csr_matrix]:
        if type(out) == sparse.csc_matrix:
            convert_to_csc = True
            out = out.tocsr()
        else:
            convert_to_csc = False
        row_repeated = np.take(row, out.indices)
        nz = out.data.nonzero()
        out.data[nz] /= row_repeated[nz]
        if convert_to_csc:
            out = out.tocsc()
    else:
        out /= row[np.newaxis, :]
    return out


def vi_tables(x, y=None, ignore_x=[0], ignore_y=[0]):
    """Return probability tables used for calculating VI.

    If y is None, x is assumed to be a contingency table.

    Parameters
    ----------
    x, y : np.ndarray
        Either x and y are provided as equal-shaped np.ndarray label fields
        (int type), or y is not provided and x is a contingency table
        (sparse.csc_matrix) that may or may not sum to 1.
    ignore_x, ignore_y : list of int, optional
        Rows and columns (respectively) to ignore in the contingency table.
        These are labels that are not counted when evaluating VI.

    Returns
    -------
    pxy : sparse.csc_matrix of float
        The normalized contingency table.
    px, py, hxgy, hygx, lpygx, lpxgy : np.ndarray of float
        The proportions of each label in `x` and `y` (`px`, `py`), the
        per-segment conditional entropies of `x` given `y` and vice-versa, the
        per-segment conditional probability p log p.
    """
    if y is not None:
        pxy = contingency_table(x, y, ignore_seg=ignore_x, ignore_gt=ignore_y)
    else:
        cont = x
        total = float(cont.sum())
        # normalize, since it is an identity op if already done
        pxy = cont / total

    # Calculate probabilities
    px = np.array(pxy.sum(axis=1)).ravel()
    py = np.array(pxy.sum(axis=0)).ravel()
    # Remove zero rows/cols
    nzx = px.nonzero()[0]
    nzy = py.nonzero()[0]
    nzpx = px[nzx]
    nzpy = py[nzy]
    nzpxy = pxy[nzx, :][:, nzy]

    # Calculate log conditional probabilities and entropies
    lpygx = np.zeros(np.shape(px))
    lpygx[nzx] = xlogx(divide_rows(nzpxy, nzpx)).sum(axis=1).ravel()
                        # \sum_x{p_{y|x} \log{p_{y|x}}}
    hygx = -(px*lpygx) # \sum_x{p_x H(Y|X=x)} = H(Y|X)

    lpxgy = np.zeros(np.shape(py))
    lpxgy[nzy] = xlogx(divide_columns(nzpxy, nzpy)).sum(axis=0).ravel()
    hxgy = -(py*lpxgy)

    return [pxy] + list(map(np.asarray, [px, py, hxgy, hygx, lpygx, lpxgy]))


def sorted_vi_components(s1, s2, ignore1=[0], ignore2=[0], compress=False):
    """Return lists of the most entropic segments in s1|s2 and s2|s1.

    Parameters
    ----------
    s1, s2 : np.ndarray of int
        Segmentations to be compared. Usually, `s1` will be a candidate
        segmentation and `s2` will be the ground truth or target segmentation.
    ignore1, ignore2 : list of int, optional
        Labels in these lists are ignored in computing the VI. 0-labels are
        ignored by default; pass empty lists to use all labels.
    compress : bool, optional
        The 'compress' flag performs a remapping of the labels before doing
        the VI computation, resulting in memory savings when many labels are
        not used in the volume. (For example, if you have just two labels, 1
        and 1,000,000, 'compress=False' will give a vector of length
        1,000,000, whereas with 'compress=True' it will have just size 2.)

    Returns
    -------
    ii1 : np.ndarray of int
        The labels in `s2` having the most entropy. If `s1` is the automatic
        segmentation, these are the worst false merges.
    h2g1 : np.ndarray of float
        The conditional entropy corresponding to the labels in `ii1`.
    ii2 : np.ndarray of int (seg)
        The labels in `s1` having the most entropy. These correspond to the
        worst false splits.
    h2g1 : np.ndarray of float
        The conditional entropy corresponding to the labels in `ii2`.
    """
    if compress:
        s1, forw1, back1 = relabel_sequential(s1)
        s2, forw2, back2 = relabel_sequential(s2)
    _, _, _, h1g2, h2g1, _, _ = vi_tables(s1, s2, ignore1, ignore2)
    i1 = (-h1g2).argsort()
    i2 = (-h2g1).argsort()
    ii1 = back1[i1] if compress else i1
    ii2 = back2[i2] if compress else i2
    return ii1, h1g2[i1], ii2, h2g1[i2]


def split_components(idx, cont, num_elems=4, axis=0):
    """Return the indices of the bodies most overlapping with body idx.

    Parameters
    ----------
    idx : int
        The segment index being examined.
    cont : sparse.csc_matrix
        The normalized contingency table.
    num_elems : int, optional
        The number of overlapping bodies desired.
    axis : int, optional
        The axis along which to perform the calculations. Assuming `cont` has
        the automatic segmentation as the rows and the gold standard as the
        columns, `axis=0` will return the segment IDs in the gold standard of
        the worst merges comprising `idx`, while `axis=1` will return the
        segment IDs in the automatic segmentation of the worst splits
        comprising `idx`.

    Value:
    comps : list of (int, float, float) tuples
        `num_elems` indices of the biggest overlaps comprising `idx`, along
        with the percent of `idx` that they comprise and the percent of
        themselves that overlaps with `idx`.
    """
    if axis == 1:
        cont= cont.T
    x_sizes = np.asarray(cont.sum(axis=1)).ravel()
    y_sizes = np.asarray(cont.sum(axis=0)).ravel()
    cc = divide_rows(cont, x_sizes)[idx].toarray().ravel()
    cct = divide_columns(cont, y_sizes)[idx].toarray().ravel()
    idxs = (-cc).argsort()[:num_elems]
    probs = cc[idxs]
    probst = cct[idxs]
    return list(zip(idxs, probs, probst))


def rand_values(cont_table):
    """Calculate values for Rand Index and related values, e.g. Adjusted Rand.

    Parameters
    ----------
    cont_table : scipy.sparse.csc_matrix
        A contingency table of the two segmentations.

    Returns
    -------
    a, b, c, d : float
        The values necessary for computing Rand Index and related values. [1, 2]
    a : float
        Refers to the number of pairs of elements in the input image that are
        both the same in seg1 and in seg2,
    b : float
        Refers to the number of pairs of elements in the input image that are
        different in both seg1 and in seg2.
    c : float
        Refers to the number of pairs of elements in the input image that are
        the same in seg1 but different in seg2.
    d : float
        Refers to the number of pairs of elements in the input image that are
        different in seg1 but the same in seg2.


    References
    ----------
    [1] Rand, W. M. (1971). Objective criteria for the evaluation of
    clustering methods. J Am Stat Assoc.
    [2] http://en.wikipedia.org/wiki/Rand_index#Definition on 2013-05-16.
    """
    n = cont_table.sum()
    sum1 = (cont_table.multiply(cont_table)).sum()
    sum2 = (np.asarray(cont_table.sum(axis=1)) ** 2).sum()
    sum3 = (np.asarray(cont_table.sum(axis=0)) ** 2).sum()
    a = (sum1 - n)/2.0;
    b = (sum2 - sum1)/2
    c = (sum3 - sum1)/2
    d = (sum1 + n**2 - sum2 - sum3)/2
    return a, b, c, d


def rand_index(x, y=None):
    """Return the unadjusted Rand index. [1]

    Parameters
    ----------
    x, y : np.ndarray
        Either x and y are provided as equal-shaped np.ndarray label fields
        (int type), or y is not provided and x is a contingency table
        (sparse.csc_matrix) that is *not* normalised to sum to 1.

    Returns
    -------
    ri : float
        The Rand index of `x` and `y`.

    References
    ----------
    [1] WM Rand. (1971) Objective criteria for the evaluation of
    clustering methods. J Am Stat Assoc. 66: 846–850
    """
    cont = x if y is None else contingency_table(x, y, norm=False)
    a, b, c, d = rand_values(cont)
    return (a+d)/(a+b+c+d)


def adj_rand_index(x, y=None):
    """Return the adjusted Rand index.

    The Adjusted Rand Index (ARI) is the deviation of the Rand Index from the
    expected value if the marginal distributions of the contingency table were
    independent. Its value ranges from 1 (perfectly correlated marginals) to
    -1 (perfectly anti-correlated).

    Parameters
    ----------
    x, y : np.ndarray
        Either x and y are provided as equal-shaped np.ndarray label fields
        (int type), or y is not provided and x is a contingency table
        (sparse.csc_matrix) that is *not* normalised to sum to 1.

    Returns
    -------
    ari : float
        The adjusted Rand index of `x` and `y`.
    """
    cont = x if y is None else contingency_table(x, y, norm=False)
    a, b, c, d = rand_values(cont)
    nk = a+b+c+d
    return (nk*(a+d) - ((a+b)*(a+c) + (c+d)*(b+d)))/(
        nk**2 - ((a+b)*(a+c) + (c+d)*(b+d)))


def fm_index(x, y=None):
    """Return the Fowlkes-Mallows index. [1]

    Parameters
    ----------
    x, y : np.ndarray
        Either x and y are provided as equal-shaped np.ndarray label fields
        (int type), or y is not provided and x is a contingency table
        (sparse.csc_matrix) that is *not* normalised to sum to 1.

    Returns
    -------
    fm : float
        The FM index of `x` and `y`. 1 is perfect agreement.

    References
    ----------
    [1] EB Fowlkes & CL Mallows. (1983) A method for comparing two
    hierarchical clusterings. J Am Stat Assoc 78: 553
    """
    cont = x if y is None else contingency_table(x, y, norm=False)
    a, b, c, d = rand_values(cont)
    return a/(np.sqrt((a+b)*(a+c)))


def reduce_vi(fn_pattern='testing/%i/flat-single-channel-tr%i-%i-%.2f.lzf.h5',
        iterable=[(ts, tr, ts) for ts, tr in it.permutations(range(8), 2)],
        thresholds=np.arange(0, 1.01, 0.01)):
    """Compile evaluation results embedded in many .h5 files under "vi".

    Parameters
    ----------
    fn_pattern : string, optional
        A format string defining the files to be examined.
    iterable : iterable of tuples, optional
        The (partial) tuples to apply to the format string to obtain
        individual files.
    thresholds : iterable of float, optional
        The final tuple elements to apply to the format string. The final
        tuples are the product of `iterable` and `thresholds`.

    Returns
    -------
    vi : np.ndarray of float, shape (3, len(thresholds))
        The under and over segmentation components of VI at each threshold.
        `vi[0, :]` is the threshold, `vi[1, :]` the undersegmentation and
        `vi[2, :]` is the oversegmentation.
    """
    iterable = list(iterable)
    vi = np.zeros((3, len(thresholds), len(iterable)), np.double)
    current_vi = np.zeros(3)
    for i, t in enumerate(thresholds):
        for j, v in enumerate(iterable):
            current_fn = fn_pattern % (tuple(v) + (t,))
            try:
                f = h5py.File(current_fn, 'r')
            except IOError:
                logging.warning('IOError: could not open file %s' % current_fn)
            else:
                try:
                    current_vi = np.array(f['vi'])[:, 0]
                except IOError:
                    logging.warning('IOError: could not open file %s'
                        % current_fn)
                except KeyError:
                    logging.warning('KeyError: could not find vi in file %s'
                        % current_fn)
                finally:
                    f.close()
            vi[:, i, j] += current_vi
    return vi


def sem(ar, axis=None):
    """Calculate the standard error of the mean (SEM) along an axis.

    Parameters
    ----------
    ar : np.ndarray
        The input array of values.
    axis : int, optional
        Calculate SEM along the given axis. If omitted, calculate along the
        raveled array.

    Returns
    -------
    sem : float or np.ndarray of float
        The SEM over the whole array (if `axis=None`) or over the chosen axis.
    """
    if axis is None:
        ar = ar.ravel()
        axis = 0
    return np.std(ar, axis=axis) / np.sqrt(ar.shape[axis])


def vi_statistics(vi_table):
    """Descriptive statistics from a block of related VI evaluations.

    Parameters
    ----------
    vi_table : np.ndarray of float
        An array containing VI evaluations of various samples. The last axis
        represents the samples.

    Returns
    -------
    means, sems, medians : np.ndarrays of float
        The statistics of the given array along the samples axis.
    """
    return np.mean(vi_table, axis=-1), sem(vi_table, axis=-1), \
        np.median(vi_table, axis=-1)