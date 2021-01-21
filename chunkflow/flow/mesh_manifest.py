import logging
import os
import re
from collections import defaultdict
from time import sleep
from typing import Union

from tqdm import tqdm

from cloudvolume import CloudVolume
from cloudfiles import CloudFiles


from .base import OperatorBase


class MeshManifestOperator(OperatorBase):
    """Create mesh manifest files for Neuroglancer visualization."""
    def __init__(self,
                 volume_path: str,
                 lod: int = 0,
                 name: str = 'mesh-manifest'):
        """
        Parameters
        ------------
        volume_path: 
            path to store mesh manifest files
        lod:
            level of detail. we always use 0!
        """
        super().__init__(name=name)
        self.lod = lod
        vol = CloudVolume(volume_path)
        info = vol.info
        assert 'mesh' in info
        self.mesh_path = os.path.join(volume_path, info['mesh'])
        self.storage = CloudFiles(self.mesh_path)
   
    def __call__(self, prefix: Union[int, str], digits: int) -> None:
        assert int(prefix) < 10**digits
        prefix = str(prefix).zfill(digits)

        id2filenames = defaultdict(list)
        for filename in tqdm(
            self.storage.list(prefix=prefix),
            desc='list mesh files'
            ):

            filename = os.path.basename(filename)
            # `match` implies the beginning (^). `search` matches whole string
            matches = re.search(r'(\d+):(\d+):', filename)

            if not matches:
                continue

            seg_id, lod = matches.groups()
            seg_id, lod = int(seg_id), int(lod)
            # currently we are not using `level of detail`, it is always 0
            # will need to adjust code if we start using variants
            assert lod == self.lod
            id2filenames[seg_id].append(filename)

        for seg_id, frags in tqdm(
            id2filenames.items(),
            desc='upload aggregated manifest file'
            ):

            logging.info(f'segment id: {seg_id}')
            logging.info(f'fragments: {frags}')
            self.storage.put_json(
                path=f'{seg_id}:{self.lod}',
                content={"fragments": frags},
            )
            # the last few hundred files will not be uploaded without sleeping!
            sleep(0.01)
