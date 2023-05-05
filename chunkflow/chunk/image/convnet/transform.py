from abc import ABC, abstractmethod
from itertools import product
from typing import List

import numpy as np


class TransformBase(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, arr: np.ndarray):
        ...

    @abstractmethod
    def backward(self, arr: np.ndarray):
        pass

class Lazy(TransformBase):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, arr: np.ndarray):
        return arr
    
    def backward(self, arr: np.ndarray):
        return arr

class FlipLR(TransformBase):
    def __init__(self) -> None:
        super().__init__()

    def _flip(self, arr: np.ndarray):
        for z in range(arr.shape[-3]):
            arr[...,z,:,:] = np.fliplr(arr[...,z,:,:])
        return arr

    def forward(self, arr: np.ndarray):
        return self._flip(arr)

    def backward(self, arr: np.ndarray):
        return self._flip(arr)


class FlipUD(TransformBase):
    def __init__(self) -> None:
        super().__init__()
    
    def _flip(self, arr: np.ndarray):
        for z in range(arr.shape[-3]):
            arr[...,z,:,:] = np.flipud(arr[...,z,:,:])
        return arr

    def forward(self, arr: np.ndarray):
        return self._flip(arr)

    def backward(self, arr: np.ndarray):
        return self._flip(arr)

class FlipZ(TransformBase):
    def __init__(self) -> None:
        super().__init__()

    def _flip(self, arr: np.ndarray):
        for x in range(arr.shape[-1]):
            arr[..., :, :, x] = np.flipud(arr[..., :, :, x])
        return arr
    
    def forward(self, arr: np.ndarray):
        return self._flip(arr)

    def backward(self, arr: np.ndarray):
        return self._flip(arr)

# class Rotate(TransformBase):
#     def __init__(self, k: int = 1) -> None:
#         super().__init__()
#         self.k = k

#     def _rotate(self, arr: np.ndarray, k: int):
#         for z in range(arr.shape[-3]):
#             arr[...,z,:,:] = np.rot90(
#                 arr[...,z,:,:], 
#                 k = k,
#                 axes=(-1, -2)
#             )
#         return arr

#     def forward(self, arr: np.ndarray):
#         return self._rotate(arr, self.k)

#     def backward(self, arr: np.ndarray):
#         return self._rotate(arr, -self.k)

class Transpose(TransformBase):
    def __init__(self) -> None:
        super().__init__()

    def _transpose(self, arr: np.ndarray):
        arr = np.swapaxes(arr, -1, -2)
        #for z in range(arr.shape[-3]):
        #    #arr[..., z, :, :] = np.transpose(arr[..., z, :, :])
        #    arr = np.swapaxes(arr, -1, -2)
        return arr
    
    def forward(self, arr):
        return self._transpose(arr)

    def backward(self, arr: np.ndarray):
        return self._transpose(arr)


class TransformSequences:
    def __init__(self, 
            transpose: bool = True,
            fliplr: bool = True,
            flipud: bool = True,
            flipz: bool = False,
            ) -> None:

        options = []
        if transpose:
            options.append((Lazy(), Transpose()))
        if fliplr:
            options.append((Lazy(), FlipLR()))
        if flipud:
            options.append((Lazy(), FlipUD()))
        if flipz:
            options.append((Lazy(), FlipZ()))

        assert len(options) > 0
        self.transform_sequences = [x for x in product(*options)]
        assert len(self.transform_sequences) == 8
        print(f'get {len(self.transform_sequences)} transformation sequences.')
        
    def forward(self, arr: np.ndarray):
        transformed_arrays = []
        for transform_sequence in self.transform_sequences:
            tmp = np.copy(arr)
            for transform in transform_sequence:
                tmp = transform.forward(tmp)
            transformed_arrays.append(tmp)

        return transformed_arrays
    
    def backward(self, transformed_arrays: List[np.ndarray]):
        assert len(transformed_arrays) == len(self.transform_sequences)
        inversed_arrays = []
        for idx, transform_sequence in enumerate(self.transform_sequences):
            arr = np.copy(transformed_arrays[idx])
            for transform in transform_sequence:
                arr = transform.backward(arr)
            inversed_arrays.append(arr)
        
        return inversed_arrays


if __name__ == '__main__':
    img = np.zeros(shape=(4, 128, 128), dtype=np.uint8)
    img[:, 60:68, :] = 255

    transform_sequences = TransformSequences()

    transformed_images = transform_sequences.forward(img)
    inversed_images = transform_sequences.backward(transformed_images)

    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2, 2, subplot_kw=dict(projection="polar"))
    axs[0, 0].imshow(img[0,...])

    fig.show()

    breakpoint() 
