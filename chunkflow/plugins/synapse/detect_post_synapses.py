from __future__ import annotations

import torch
from neutorch.model.IsoRSUNet import Model

import numpy as np
from chunkflow.chunk import Chunk
from chunkflow.chunk.probability_map import ProbabilityMap
from chunkflow.lib.synapses import Synapses
from chunkflow.lib.cartesian_coordinate import BoundingBox, Cartesian

# import patch_inferencer
# from .patch_inferencer import PatchInferencer


model_weight_file = '/mnt/ceph/users/neuro/wasp_em/jwu/14_post_synapse_net/12_net/model_570000.chkpt'
#model_weight_file = '/mnt/ceph/users/neuro/wasp_em/jwu/14_post_synapse_net/32/model_1000000.chkpt'
#model_weight_file = '/mnt/ceph/users/neuro/wasp_em/jwu/14_post_synapse_net/32/model_500000.chkpt'
patch_size = Cartesian(192, 192, 192)
is_data_parallel = True
CROP_MARGIN = 28

# model_weight_file = '/mnt/ceph/users/neuro/wasp_em/jwu/14_post_synapse_net/11_net/model_129000.chkpt'
# patch_size = Cartesian(256, 256, 256)
# is_data_parallel = False

resolution = Cartesian(8, 8, 8)

for ele in patch_size:
    assert ele == patch_size[0]


class PatchInferencer:
    def __init__(self, 
            model_weight_file: str, 
            output_patch_mask: np.ndarray):
        
        if output_patch_mask is not None:
            output_patch_mask = torch.tensor(output_patch_mask)
        self.output_patch_mask = output_patch_mask

        model = Model(1,1)
        if is_data_parallel:
            model = torch.nn.DataParallel(model)
        #model.load(model_weight_file)
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        checkpoint = torch.load(model_weight_file, map_location=device)
        # breakpoint() 
        model.load_state_dict(checkpoint['state_dict'])
        model.eval() 
        if torch.cuda.is_available():
            model.cuda()
            if self.output_patch_mask is not None:
                self.output_patch_mask = self.output_patch_mask.cuda()
        self.model = model

    @property
    def compute_device(self):
        return torch.cuda.get_device_name()

    def __call__(self, input_patch):
        with torch.no_grad():
            input_patch = torch.from_numpy(input_patch)
            if torch.cuda.is_available():
                input_patch = input_patch.cuda()
            output_patch = self.model( input_patch ).sigmoid()
            if self.output_patch_mask is not None:
                output_patch = output_patch * self.output_patch_mask
            output_patch = output_patch.cpu().numpy()
        return output_patch

inferencer = PatchInferencer(model_weight_file=model_weight_file, output_patch_mask=None)

def detect_post(pre_idx: int, synapses: Synapses, img: Chunk):
    pre = synapses.pre[pre_idx, :]
    # fetch the image patch
    bbox = BoundingBox.from_center(pre, extent=patch_size[0]//2)
    np.testing.assert_array_equal(bbox.size3(), patch_size)
    patch = img.cutout(bbox)
    patch = patch.array
    patch = patch.astype(np.float32)
    patch /= 255.
    # inference requires 5D array
    patch = patch.reshape((1,1) + patch.shape)

    post_map = inferencer(patch)
    post_map = np.squeeze(post_map)
    # print('post map statistics: ', describe(post_map.flatten()))
    np.testing.assert_array_equal(post_map.shape, bbox.size3())
    post_map = ProbabilityMap.from_array(
        post_map, bbox, 
        voxel_size=img.voxel_size
    )

    points, confidences = post_map.detect_points(
        min_distance=6, 
        threshold_rel=0.3, 
        exclude_border=CROP_MARGIN,
        num_peaks=15
    )

    if points is None:
        posts = None
        confidences = None
    else:
        posts = []
        for idx in range(points.shape[0]):
            post = (pre_idx, *points[idx, :])
            posts.append(post)

    return posts, confidences

def execute(img: Chunk, synapses: Synapses, aligned_bbox: BoundingBox):
    if synapses is None or img is None :
        return (None,)

    # this synapses should be downloaded from DVID with latest update
    # we also have added some unpredicted synapses in it
    # we only consider some synapses inside our bounding box 
    synapses.remove_synapses_outside_bounding_box(aligned_bbox)
    # synapses.remove_pre_duplicates()
    if synapses.pre_num == 0:
        return(None, )

    # this is for debug since we want to run it quickly
    # synapses.pre = synapses.pre[:10, :]

    # pre_index2post_indices = synapses.pre_index2post_indices
    posts = []
    predicted_connectivities = []
    connectivity_confidence = []
    post_confidences = []
    for pre_idx, post_indices in enumerate(synapses.pre_index2post_indices):
        if post_indices is not None:
            # skip the inference, use existing manually annotated post synapses
            for post_idx in post_indices:
                post = tuple(synapses.post[post_idx, :])
                posts.append(post)

                if synapses.post_confidence is None:
                    confidence = 1.
                else:
                    confidence = synapses.post_confidence[post_idx]
                post_confidences.append(confidence)
        else:
            predicted_posts, confidences = detect_post(pre_idx, synapses, img)
            if predicted_posts is not None:
                posts.extend(predicted_posts)
                post_confidences.extend(confidences)
                connectivity_confidence.extend(confidences)

                for predicted_post in predicted_posts:
                    predicted_connectivity = (*synapses.pre[pre_idx, :], *predicted_post[1:]) 
                    predicted_connectivities.append(predicted_connectivity)

    synapses.post = np.asarray(posts, dtype=np.int32)
    synapses.post_confidence = np.asarray(post_confidences, dtype=np.float32)

    return (synapses,)
