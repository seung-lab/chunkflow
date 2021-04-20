import json
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

config = {}
config['supervoxels'] = False 
config['server'] = server
config['instance'] = 'segmentation'
config['uuid'] = uuid

config_str = json.dumps(config)

def execute(bbox: Bbox, args: str= config_str ):
    if args is not None:
        config = json.loads(args)
        if 'server' in config:
            server = config['server']
        if 'supervoxels' in config:
            supervoxels = config['supervoxels']
        if 'instance' in config:
            instance = config['instance']
        if 'uuid' in config:
            uuid = config['uuid']

    if ":main" in uuid or ":master" in uuid:
        uuid, _ = uuid.split(':')
        #uuid = dvid.find_master(uuid)
    
    print('bounding box: ', bbox)
    box = [tuple(bbox.minpt), tuple(bbox.maxpt)]
    subvol = dvid.fetch_labelmap_voxels(
        server, uuid, instance, box,
        scale=0, 
        supervoxels=supervoxels)
    # print('cutout volume: \n', subvol)
    chunk = Chunk(subvol, voxel_offset=bbox.minpt)
    
    #breakpoint()
    return [chunk]