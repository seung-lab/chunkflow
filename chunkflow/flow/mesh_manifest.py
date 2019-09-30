import os
import re
from collections import defaultdict
from time import sleep

from cloudvolume import CloudVolume
from cloudvolume.storage import Storage

from .base import OperatorBase


class MeshManifestOperator(OperatorBase):
    """Create mesh manifest files for Neuroglancer visualization."""
    def __init__(self,
                 volume_path: str,
                 lod: int = 0,
                 name: str = 'mesh-manifest',
                 verbose: bool = True):
        """
        Parameters
        ------------
        volume_path: 
            path to store mesh manifest files
        lod:
            level of detail. we always use 0!
        """
        super().__init__(name=name, verbose=verbose)
        self.lod = lod
        vol = CloudVolume(volume_path)
        info = vol.info
        assert 'mesh' in info
        self.mesh_path = os.path.join(volume_path, info['mesh'])
        self.storage = Storage(self.mesh_path)
   
    def __call__(self, prefix: str):
        id2filenames = defaultdict(list)
        for filename in self.storage.list_files(prefix=prefix):
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

        for seg_id, frags in id2filenames.items():
            if self.verbose:
                print('segment id: ', seg_id)
                print('fragments: ', frags)
            self.storage.put_json(
                # level of detail is alway
                file_path='{}:{}'.format(seg_id, self.lod),
                content={"fragments": frags},
            )
            # the last few hundred files will not be uploaded without sleeping!
            sleep(0.01)
