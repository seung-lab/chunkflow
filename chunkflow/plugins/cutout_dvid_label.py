from cloudvolume.lib import Bbox
from neuclease import dvid

from chunkflow.chunk import Chunk

# these example parameters are from the example of [neuclease](https://github.com/janelia-flyem/neuclease)
supervoxels = False
server = "http://hemibrain-dvid.janelia.org:80"
server = server.strip('/') 
if ':' not in server[-5:]:
    # add a default port
    server += ':8000'

instance = "segmentation"
# uuid = dvid.find_master(server)
uuid =  "20631f"
print('server: ', server)
print('instance: ', instance)
print('uuid: ', uuid)


def execute(bbox: Bbox):
    print('bounding box: ', bbox)
    box = [tuple(bbox.minpt), tuple(bbox.maxpt)]
    subvol = dvid.fetch_labelmap_voxels(
        server, uuid, instance, box,
        scale=0, 
        supervoxels=supervoxels)
    # print('cutout volume: \n', subvol)
    chunk = Chunk(subvol, voxel_offset=bbox.minpt)
    return [chunk]