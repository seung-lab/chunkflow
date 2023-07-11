try:
    import napari
except:
    print('napari not installed and related operators could not be used. This does not impact other operators.')

import numpy as np
from .base import OperatorBase


class NapariOperator(OperatorBase):
    def __init__(self,
                 name: str = 'neuroglancer',
                 voxel_size: tuple = None):
        super().__init__(name=name)
        self.voxel_size = voxel_size

    def __call__(self, datas: dict, selected: str=None):
        if selected is None:
            selected = datas.keys()
        elif isinstance(selected, str):
            selected = selected.split(',')

        for name in selected:
            data = datas[name]

            if np.issubdtype(data.dtype, np.uint8): 
                viewer = napari.view_image(data.array, rgb=False)
        napari.run()  # start the event loop and show viewer