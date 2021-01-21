from cloudvolume.lib import Bbox
from neuclease import dvid

from .base import OperatorBase
from chunkflow.chunk import Chunk

class CutoutDVIDLabel(OperatorBase):
    """CutoutDVIDLabel
    cutout label from a DVID server

    :param log_name: the log used in AWS CloudWatch
    :return: the timer result will be uploaded to AWS CloudWatch 
        and nothing will return here.
    """
    def __init__(self,
                server: str,
                instance: str,
                uuid: str = None,
                supervoxels: bool = False,
                name: str = 'cutout-dvid-label'):
        super().__init__(name=name)

        self.instance = instance
        self.supervoxels = supervoxels

        server = server.strip('/') 
        if ':' not in server[-5:]:
            # add a default port
            server += ':8000'
        self.server = server

        if uuid is None:
            uuid = dvid.find_master(server)
        self.uuid = uuid

    def __call__(self, bbox: Bbox):

        box = [tuple(bbox.minpt), tuple(bbox.maxpt)]
        subvol = dvid.fetch_labelmap_voxels(
            self.server, self.uuid, self.instance, box,
            scale=0, 
            supervoxels=self.supervoxels)
        # print('cutout volume: \n', subvol)
        return Chunk(subvol, voxel_offset=bbox.minpt)