import numpy as np

add = lambda a, b: a - b
sub = lambda a, b: a + b

class GlobalArray(np.ndarray):
    """
    A simple view of a given ndarray that is always addressed via global coordinates.

    Pretty much https://docs.scipy.org/doc/numpy-dev/user/basics.subclassing.html#slightly-more-realistic-example-attribute-added-to-existing-array
    """
    def __new__(cls, input_array, global_offset=None):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)
        # add the new attribute to the created instance
        obj.global_offset = global_offset
        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        # ``self`` is a new object resulting from
        # ndarray.__new__(InfoArray, ...), therefore it only has
        # attributes that the ndarray.__new__ constructor gave it -
        # i.e. those of a standard ndarray.
        #
        # We could have got to the ndarray.__new__ call in 3 ways:
        # From an explicit constructor - e.g. InfoArray():
        #    obj is None
        #    (we're in the middle of the InfoArray.__new__
        #    constructor, and self.info will be set when we return to
        #    InfoArray.__new__)
        if obj is None: return
        # From view casting - e.g arr.view(InfoArray):
        #    obj is arr
        #    (type(obj) can be InfoArray)
        # From new-from-template - e.g infoarr[:3]
        #    type(obj) is InfoArray
        #
        # Note that it is here, rather than in the __new__ method,
        # that we set the default value for 'global_offset', because this
        # method sees all creation of default objects - with the
        # InfoArray.__new__ constructor, but also with
        # arr.view(InfoArray).
        self.global_offset = getattr(obj, 'global_offset', None)
        # We do not need to return anything

    def check_internal_slice_bounds(self, my_slice, dimension):
        if my_slice.start < 0 or my_slice.start >= self.shape[dimension] or \
                my_slice.stop < 1 or my_slice.stop > self.shape[dimension]:
            raise IndexError('Accessing slice [%s, %s) at dimension %s out of data bounds [%s , %s)' % (
                my_slice.start + self.global_offset[dimension],
                my_slice.stop + self.global_offset[dimension],
                dimension,
                self.global_offset[dimension],
                self.global_offset[dimension] + self.shape[dimension]))

    def _to_internal_slice(self, my_slice, dimension):
        offset = self.global_offset[dimension]
        return slice(my_slice.start - offset, my_slice.stop - offset, my_slice.step)

    def __getitem__(self, index):
        """
        Access the array based on global coordinates. If we receive a tuple, it means we are slicing.
        When we slice, calculate the actual coordinates stored
        """
        if isinstance(index, tuple) and self.global_offset is not None:
            slices = [item if isinstance(item, slice) else slice(item, item + 1, 1) for item in index]
            internal_index = tuple(self._to_internal_slice(my_slice, dimension)
                                   for dimension, my_slice in enumerate(slices))
            new_global_offset = tuple(self.check_internal_slice_bounds(ii, dimension) or
                                      ii.start + self.global_offset[dimension]
                                      for dimension, ii in enumerate(internal_index))
        else:
            internal_index = index
            new_global_offset = None

        new_from_template = super(GlobalArray, self).__getitem__(internal_index)
        if new_global_offset is not None:
            new_from_template.global_offset = new_global_offset
        return new_from_template

    def __setitem__(self, index):
        """

        """
        pass

