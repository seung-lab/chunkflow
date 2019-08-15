from .base import OperatorBase

from warnings import warn
import os
import numpy as np

from cloudvolume.lib import save_images


class SavePNGsOperator(OperatorBase):
    def __init__(self,
                 output_path: str = './saved_images/',
                 name: str = 'save-pngs',
                 verbose: bool = True):
        super().__init__(name=name, verbose=verbose)

        if not os.path.isdir(output_path):
            warn('output path do not exist, will create %s.' % output_path)
            os.makedirs(output_path)

        self.output_path = output_path

    def __call__(self, chunk):
        assert isinstance(chunk, np.ndarray)
        # cloudvolume use fortran order, and the index are xyzc
        warn('Note that the saved image will be transposed!')
        image = np.transpose(chunk)

        # currently cloudvolume save_images do not work with specific
        # channel for 4D array
        # https://github.com/seung-lab/cloud-volume/issues/206
        save_images(image, directory=self.output_path)
