#!/usr/bin/env python
__doc__ = """
ConvNet Inference of an image chunk
"""
import os
import time
import numpy as np
from tqdm import tqdm
from warnings import warn
from typing import Union
from tempfile import mktemp

from chunkflow.chunk import Chunk
# from chunkflow.chunk.affinity_map import AffinityMap


class Inferencer(object):
    """
        Inferencer
    convnet inference for a whole chunk. 

    if the patches align with the input chunk size, we do not need chunk mask.
    if the patches do not align, we'll create a chunk mask to make sure that 
    the output have the same size with input.

    The output buffer is smaller than the input chunk size, and the cropped 
    margin area is not allocated. This will save about 20% of memory usage.
    what's more, the output buffer is formated as memory map and was mapped 
    to disk. This is particularly useful for multiple channel output with 
    large chunk size.
    """
    def __init__(self,
                 convnet_model: str,
                 convnet_weight_path: str,
                 input_patch_size: Union[tuple, list],
                 output_patch_size: Union[tuple, list] = None,
                 patch_num: Union[tuple, list] = None,
                 num_output_channels: int = 3,
                 output_patch_overlap: Union[tuple, list] = (4, 64, 64),
                 output_crop_margin: Union[tuple, list] = None,
                 dtype = 'float32',
                 framework: str = 'identity',
                 batch_size: int = 1,
                 bump: str = 'wu',
                 input_size: tuple = None,
                 mask_output_chunk: bool = False,
                 mask_myelin_threshold = None,
                 dry_run: bool = False,
                 verbose: int = 1):
        
        assert input_size is None or patch_num is None 
        
        if output_patch_size is None:
            output_patch_size = input_patch_size 
        
        self.input_patch_size = input_patch_size
        self.output_patch_size = output_patch_size
        self.output_patch_overlap = output_patch_overlap
        self.patch_num = patch_num
        self.batch_size = batch_size
        self.input_size = input_size
        
        if mask_output_chunk:
            # the chunk mask will handle the boundaries 
            self.output_crop_margin = (0, 0, 0)
        else:
            if output_crop_margin is None:
                self.output_crop_margin = self.output_patch_overlap
            else:
                self.output_crop_margin = output_crop_margin
                # we should always crop more than the patch overlap 
                # since the overlap region is reweighted by patch mask
                # To-Do: equal should also be OK
                assert np.alltrue([v<=m for v, m in zip(
                    self.output_patch_overlap, 
                    self.output_crop_margin)])

        self.output_patch_crop_margin = tuple((ips-ops)//2 for ips, ops in zip(
            input_patch_size, output_patch_size))
        
        self.output_offset = tuple(opcm+ocm for opcm, ocm in zip(
            self.output_patch_crop_margin, self.output_crop_margin))
    
        self.output_patch_stride = tuple(s - o for s, o in zip(
            output_patch_size, output_patch_overlap))

        self.input_patch_overlap = tuple(opcm*2+oo for opcm, oo in zip(
            self.output_patch_crop_margin, self.output_patch_overlap))

        self.input_patch_stride = tuple(ps - po for ps, po in zip(
            input_patch_size, self.input_patch_overlap))
        
        # no chunk wise mask, the patches should be aligned inside chunk
        if not mask_output_chunk:
            assert (self.input_size is not None) or (self.patch_num is not None)
            if patch_num is None:
                assert input_size is not None
                self.patch_num = tuple((isz - o)//s for isz, o, s in zip(
                    self.input_size, self.input_patch_overlap, self.input_patch_stride))

            if self.input_size is None:
                assert self.patch_num is not None 
                self.input_size = tuple(pst*pn + po for pst, pn, po in zip(
                    self.input_patch_stride, self.patch_num, self.input_patch_overlap))
             
            self.output_size = tuple(pst*pn + po - 2*ocm for pst, pn, po, ocm in zip(
                self.output_patch_stride, self.patch_num, 
                self.output_patch_overlap, self.output_crop_margin))
        else:
            # we can handle arbitrary input and output size
            self.input_size = None 
            self.output_size = None

        self.num_output_channels = num_output_channels
        self.verbose = verbose
        self.mask_output_chunk = mask_output_chunk
        self.output_chunk_mask = None
        self.dtype = dtype        
        self.mask_myelin_threshold = mask_myelin_threshold
        self.dry_run = dry_run
        
        # allocate a buffer to avoid redundant memory allocation
        self.input_patch_buffer = np.zeros((batch_size, 1, *input_patch_size),
                                           dtype=dtype)

        self.patch_slices_list = []
        
        if isinstance(convnet_model, str):
            convnet_model = os.path.expanduser(convnet_model)
        if isinstance(convnet_weight_path, str):
            convnet_weight_path = os.path.expanduser(convnet_weight_path)
        self._prepare_patch_inferencer(framework, convnet_model, convnet_weight_path, bump)
   
    @property
    def compute_device(self):
        return self.patch_inferencer.compute_device

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        pass
    
    def _update_parameters_for_input_chunk(self, input_chunk):
        """
        if the input size is consistent with old one, reuse the
        patch offset list and output chunk mask. Otherwise, recompute them.
        """
        if np.array_equal(self.input_size, input_chunk.shape):
            print('reusing output chunk mask.')
            assert self.patch_slices_list is not None
        else:
            if self.input_size is not None:
                warn('the input size has changed, using new intput size.')
            self.input_size = input_chunk.shape
            
            if not self.mask_output_chunk: 
                self._check_alignment()

            self.output_size = tuple(
                isz-2*ocso for isz, ocso in 
                zip(self.input_size, self.output_offset))
        
        self.output_patch_stride = tuple(s-o for s, o in zip(
            self.output_patch_size, self.output_patch_overlap))

        self._construct_patch_slices_list(input_chunk.global_offset)
        self._construct_output_chunk_mask(input_chunk)

    def _prepare_patch_inferencer(self, framework, convnet_model, convnet_weight_path, bump):
        # prepare for inference
        if framework == 'pznet':
            from .patch.pznet import PZNet as PatchInferencer
        elif framework == 'pytorch':
            # pytorch will not output consistent result if we use batch size > 1
            # https://discuss.pytorch.org/t/solved-inconsistent-results-during-test-using-different-batch-size/2265 
            assert self.batch_size == 1
            from .patch.pytorch import PyTorch as PatchInferencer
            # currently, we do not support pytorch backend with different
            # input and output patch size and overlap.
        elif framework == 'pytorch-multitask':
            # currently only this type of task support mask in device
            from .patch.pytorch_multitask import PyTorchMultitask as PatchInferencer
        elif framework == 'identity':
            from .patch.identity import Identity as PatchInferencer
        elif framework == 'general':
            from .patch.general import General as PatchInferencer
        else:
            raise Exception(f'invalid inference backend: {self.framework}')
        
        self.patch_inferencer = PatchInferencer(
            convnet_model,
            convnet_weight_path,
            input_patch_size=self.input_patch_size,
            output_patch_size=self.output_patch_size,
            output_patch_overlap=self.output_patch_overlap,
            num_output_channels=self.num_output_channels,
            dtype=self.dtype,
            bump=bump)

    def _check_alignment(self):
        is_align = tuple((i - o) % s == 0 for i, s, o in zip(
            self.input_size, 
            self.patch_inferencer.input_patch_stride, 
            self.patch_inferencer.input_patch_overlap))

        # all axis should be aligned
        # the patches should aligned with input size in case
        # we will not mask the output chunk
        assert np.all(is_align)
        if self.verbose:
            print('great! patches aligns in chunk.')

    def _construct_patch_slices_list(self, input_chunk_offset):
        """
        create the normalization mask and patch bounding box list
        """
        self.patch_slices_list = []
        # the step is the stride, so the end of aligned patch is
        # input_size - patch_overlap
        
        input_patch_size = self.input_patch_size
        output_patch_size = self.output_patch_size
        input_patch_overlap = self.input_patch_overlap 
        input_patch_stride = self.input_patch_stride 

        print('Construct patch slices list...')
        for iz in range(0, self.input_size[0] - input_patch_overlap[0], input_patch_stride[0]):
            if iz + input_patch_size[0] > self.input_size[0]:
                iz = self.input_size[0] - input_patch_size[0]
                assert iz >= 0
            iz += input_chunk_offset[-3]
            oz = iz + self.output_patch_crop_margin[0]
            for iy in range(0, self.input_size[1] - input_patch_overlap[1], input_patch_stride[1]):
                if iy + input_patch_size[1] > self.input_size[1]:
                    iy = self.input_size[1] - input_patch_size[1]
                    assert iy >= 0
                iy += input_chunk_offset[-2]
                oy = iy + self.output_patch_crop_margin[1]
                for ix in range(0, self.input_size[2] - input_patch_overlap[2], input_patch_stride[2]):
                    if ix + input_patch_size[2] > self.input_size[2]:
                        ix = self.input_size[2] - input_patch_size[2]
                        assert ix >= 0
                    ix += input_chunk_offset[-1]
                    ox = ix + self.output_patch_crop_margin[2]
                    input_patch_slice =  (slice(iz, iz + input_patch_size[0]),
                                          slice(iy, iy + input_patch_size[1]),
                                          slice(ix, ix + input_patch_size[2]))
                    output_patch_slice = (slice(oz, oz + output_patch_size[0]),
                                          slice(oy, oy + output_patch_size[1]),
                                          slice(ox, ox + output_patch_size[2]))
                    self.patch_slices_list.append((input_patch_slice, output_patch_slice))

    def _construct_output_chunk_mask(self, input_chunk):
        if not self.mask_output_chunk:
            return

        if self.verbose:
            print('creating output chunk mask...')
        
        if self.output_chunk_mask is None or not np.array_equal(
                input_chunk.shape, self.output_chunk_mask.shape):
            # To-Do: clean up extra temporal files if we created 
            # multiple mmap files
            #output_mask_mmap_file = mktemp(suffix='.dat')
            ## the memory map is initialized with 0 in default
            #output_mask_array = np.memmap(output_mask_mmap_file, 
            #                                   dtype=self.dtype, mode='w+', 
            #                                   shape=self.output_size)
            output_mask_array = np.zeros(self.output_size, self.dtype)
        else:
            output_chunk_mask_array = self.output_chunk_mask.array
            output_chunk_mask_array.fill(0)

        output_global_offset = tuple(io + ocso for io, ocso in zip(
            input_chunk.global_offset, self.output_offset))
 
        self.output_chunk_mask = Chunk(output_mask_array, global_offset=output_global_offset)
        
        assert len(self.patch_slices_list) > 0
        for _, output_patch_slice in self.patch_slices_list:
            # accumulate weights using the patch mask in RAM
            patch_global_offset = tuple(s.start for s in output_patch_slice)
            patch_mask = Chunk(self.patch_inferencer.output_patch_mask_numpy,
                               global_offset=patch_global_offset)
            self.output_chunk_mask.blend(patch_mask)
        
        # normalize weight, so accumulated inference result multiplies
        # this mask will result in 1
        self.output_chunk_mask.array = 1.0 / self.output_chunk_mask.array
    
    def _get_output_buffer(self, input_chunk):
        output_buffer_size = (self.patch_inferencer.num_output_channels, ) + self.output_size
        #if self.mask_myelin_threshold is None:
        # a random temporal file. will be removed later.
        #output_buffer_mmap_file = mktemp(suffix='.dat')
        ## the memory map is initialized with 0 in default
        #output_buffer_array = np.memmap(output_buffer_mmap_file, 
        #                               dtype=self.dtype, mode='w+', 
        #                               shape=output_buffer_size)
        ##else:
        #    # when we use myelin mask, the masking computation will create a full array in RAM!
        #    # and it will duplicate the array! thus, we should use normal array in this case.
        output_buffer_array = np.zeros(output_buffer_size, dtype=self.dtype)
        
        output_global_offset = tuple(io + ocso for io, ocso in zip(
            input_chunk.global_offset, self.output_offset))
        
        output_buffer = Chunk(output_buffer_array,
                                   global_offset=(0,) + output_global_offset)
        assert output_buffer == 0
        return output_buffer

    def __call__(self, input_chunk: np.ndarray):
        """
        args:
            input_chunk (Chunk): input chunk with global offset
        """
        assert isinstance(input_chunk, Chunk)
        
        self._update_parameters_for_input_chunk(input_chunk)
        output_buffer = self._get_output_buffer(input_chunk)

        if not self.mask_output_chunk:
            self._check_alignment()
         
        if self.dry_run:
            print('dry run, return a special artifical chunk.')
            size=output_buffer.shape
            
            if self.mask_myelin_threshold:
                # eleminate the myelin channel
                size = (size[0]-1, *size[1:])

            return Chunk.create(
                size=size,
                dtype = output_buffer.dtype,
                voxel_offset=output_buffer.global_offset
            )
       
        if input_chunk == 0:
            print('input is all zero, return zero buffer directly')
            return output_buffer
        
        if np.issubdtype(input_chunk.dtype, np.integer):
            # normalize to 0-1 value range
            dtype_max = np.iinfo(input_chunk.dtype).max
            input_chunk = input_chunk.astype(self.dtype) / dtype_max

        if self.verbose:
            chunk_time_start = time.time()

        # iterate the offset list
        for i in tqdm(range(0, len(self.patch_slices_list), self.batch_size),
                      disable=not self.verbose,
                      desc='ConvNet inference for patches: '):
            if self.verbose:
                start = time.time()

            batch_slices = self.patch_slices_list[i:i + self.batch_size]
            for batch_idx, slices in enumerate(batch_slices):
                self.input_patch_buffer[
                    batch_idx, 0, :, :, :] = input_chunk.cutout(slices[0]).array

            if self.verbose > 1:
                end = time.time()
                print('prepare %d input patches takes %3f sec' %
                      (self.batch_size, end - start))
                start = end

            # the input and output patch is a 5d numpy array with
            # datatype of float32, the dimensions are batch/channel/z/y/x.
            # the input image should be normalized to [0,1]
            output_patch = self.patch_inferencer(self.input_patch_buffer)

            if self.verbose > 1:
                assert output_patch.ndim == 5
                end = time.time()
                print('run inference for %d patch takes %3f sec' %
                      (self.batch_size, end - start))
                start = end

            for batch_idx, slices in enumerate(batch_slices):
                # only use the required number of channels
                # the remaining channels are dropped
                # the slices[0] is for input patch slice
                # the slices[1] is for output patch slice
                offset = (0,) + tuple(s.start for s in slices[1])
                output_chunk = Chunk(output_patch[batch_idx, :, :, :, :],
                                     global_offset=offset)

                ## save some patch for debug
                #bbox = output_chunk.bbox
                #if bbox.minpt[-1] < 94066 and bbox.maxpt[-1] > 94066 and \
                #        bbox.minpt[-2]<81545 and bbox.maxpt[-2]>81545 and \
                #        bbox.minpt[-3]<17298 and bbox.maxpt[-3]>17298:
                #    print('save patch: ', output_chunk.bbox)
                #    breakpoint()
                #    output_chunk.to_tif()
                #    #input_chunk.cutout(slices[0]).to_tif()

                output_buffer.blend(output_chunk)

            if self.verbose > 1:
                end = time.time()
                print('blend patch takes %3f sec' % (end - start))

        if self.verbose:
            print("Inference of whole chunk takes %3f sec" %
                  (time.time() - chunk_time_start))
        
        if self.mask_output_chunk:
            output_buffer *= self.output_chunk_mask
        
        # theoretically, all the value of output_buffer should not be greater than 1
        # we use a slightly higher value here to accomondate numerical precision issue
        np.testing.assert_array_less(output_buffer, 1.0001,
            err_msg='output buffer should not be greater than 1')

        if self.mask_myelin_threshold:
            # currently only for masking out affinity map 
            assert output_buffer.shape[0] == 4
            output_chunk = output_buffer.mask_using_last_channel(
                threshold = self.mask_myelin_threshold)

            # currently neuroglancer only support float32, not float16
            if output_chunk.dtype == np.dtype('float16'):
                output_chunk = output_chunk.astype('float32')

            return output_chunk
        else:
            return output_buffer
