# -*- coding: utf-8 -*-
"""
Flow of chunks.

The interface is separated to two parts. 
The chunk input and output part and the chunk processing part.
Input and Output part:
    input and output path 
    input and output chunk bounding box. The bounding box could be fetched from AWS SQS queue.

Chunk processing part:
    the parameters used to process the chunk.
"""

import numpy as np
import time
import os
import json 
from cloudvolume import CloudVolume, Storage
from cloudvolume.lib import Vec, Bbox

from .lib.validate import validate_by_template_matching
from .igneous.tasks import downsample_and_upload
from .igneous.downsample import downsample_with_averaging
from .lib.offset_array import OffsetArray
from .aws.cloud_watch import CloudWatch


class FlowBase(object):
    def __init__(self,
                 input_layer_path,
                 output_layer_path,
                 mip=0,
                 input_mask_layer_path=None,
                 output_mask_layer_path=None,
                 input_mask_mip=3,
                 output_mask_mip=3,
                 cropping_margin_size=(0,0,0),
                 fill_input_missing=False,
                 inverse_input_mask=True,
                 inverse_output_mask=True,
                 missing_section_ids_file_name=None,
                 input_validate_mip=None,
                 show_progress=False):
        self.input_layer_path = input_layer_path
        self.output_layer_path = output_layer_path
        self.input_mip = mip
        self.output_mip = mip
        self.input_mask_layer_path = input_mask_layer_path
        self.output_mask_layer_path = output_mask_layer_path
        self.input_mask_mip = input_mask_mip
        self.output_mask_mip = output_mask_mip
        self.cropping_margin_size = cropping_margin_size
        self.fill_input_missing=fill_input_missing
        self.inverse_input_mask = inverse_input_mask
        self.inverse_output_mask = inverse_output_mask 
        self.missing_section_ids_file_name = missing_section_ids_file_name
        self.show_progress = show_progress
        self.input_validate_mip = input_validate_mip

        self.log = dict()
        self.log['parameters']={
            'input_layer_path':         input_layer_path,
            'output_layer_path':        output_layer_path,
            'output_mask_layer_path':   output_mask_layer_path,
            'cropping_margin_size':     (*cropping_margin_size,),
            'mip':                      mip,
            'output_mask_mip':          output_mask_mip,
            'input_validate_mip':       input_validate_mip
        }
        
        self.aws_cloud_watch = CloudWatch('inference')
    
    def _prepare_input(self):
        """
        prepare the input chunk, such as read and masking
        """
        time_log = self.log['time_elapsed']

        start = time.time()
        self._read_input()
        elapsed = time.time() - start
        time_log['read_input'] = elapsed
        print("Read input takes %3f sec" % (elapsed))
        
        start = time.time()
        self._validate_input()
        elapsed = time.time() - start
        time_log['validate_input'] = elapsed
        print("Validate input takes %3f sec" % (elapsed))

        start = time.time()
        self._mask_missing_sections()
        elapsed = time.time() - start
        time_log['mask_missing_sections'] = elapsed
        print("Mask missing sections in input takes %3f sec" % (elapsed))
        
        if self.input_mask_layer_path:
            start = time.time()
            self._mask_input()
            elapsed = time.time() - start
            time_log['mask_input'] = elapsed
            print("Mask input takes %3f sec" % (elapsed))
    
    def _finish_output(self):
        """
        process and upload the output chunk
        """
        start = time.time()
        self._crop()
        elapsed = time.time() - start
        time_log['crop_output'] = elapsed
        print("Cropping takes %3f sec" % (elapsed))

        if self.output_mask_layer_path:
            start = time.time()
            self._mask_output()
            elapsed = time.time() - start
            time_log['mask_output'] = elapsed
            print("Mask output takes %3f sec" % (elapsed))

        start = time.time()
        self._upload_output()
        elapsed = time.time() - start
        time_log['upload_output'] = elapsed
        print("Upload output takes %3f min" % (elapsed / 60))

        start = time.time()
        self._create_output_thumbnail()
        elapsed = time.time() - start
        time_log['create_output_thumbnail_time'] = elapsed
        print("create output thumbnail takes %3f min" % (elapsed / 60))

        total_time = time.time() - total_start
        time_log['complete_task'] = total_time
        print("Whole task takes %3f min" % (total_time / 60))

    def _process_chunk(self):
        """
        this function should be inherited and reimplemented
        """
        pass
    
    def __call__(self, output_bbox):
        """
        perform the processing 
        """
        if isinstance(output_bbox, str):
            output_bbox = Bbox.from_filename(output_bbox)
        elif isinstance(output_bbox, tuple):
            output_bbox = Bbox.from_slices(output_bbox)
        else:
            assert isinstance(output_bbox, Bbox)

        self.log['output_bbox'] = output_bbox.to_filename()
        self.output_bbox = output_bbox
        output_slices = output_bbox.to_slices()
        input_slices = tuple(slice(s.start-c, s.stop+c) for s,c in 
                             zip(output_slices, self.cropping_margin_size))
        self.input_bbox = Bbox.from_slices(input_slices)

        total_start = time.time()
        self.log['time_elapsed'] = {}

        self._prepare_input()
        self._process_chunk()
        self._finish_output()

        log_path = os.path.join(self.output_layer_path, 'log')
        self._upload_log(log_path)
    
    def _read_mask(self, mask_layer_path, mask_mip, mip, bbox, 
                   inverse_mask=True):
        """
        bbox: the bounding box in input mip
        inverse_mask: (bool) whether inverse the mask or not
        """
        if not mask_layer_path:
            print('no mask layer path defined')
            return None
        
        # print("download mask chunk...")
        vol = CloudVolume(
            mask_layer_path,
            bounded=False,
            fill_missing=self.fill_input_missing,
            progress=self.show_progress,
            mip=mask_mip)
        # assume that input mip is the same with output mip
        xyfactor = 2**(mask_mip - mip)
        # only scale the indices in XY plane
        mask_slices = tuple(
            slice(a.start // xyfactor, a.stop // xyfactor)
            for a in bbox.to_slices()[1:3])
        mask_slices = (bbox.to_slices()[0], ) + mask_slices

        # the slices did not contain the channel dimension 
        mask = vol[mask_slices[::-1]]
        mask = np.transpose(mask)
        mask = np.squeeze(mask, axis=0)
        
        if inverse_mask:
            mask = (mask==0)
        return mask
    
    def _mask_missing_sections(self):
        """
        mask some missing sections if the section id was provided 
        """
        if self.missing_section_ids_file_name:
            zslice = self.input.slices[0]
            start = zslice.start
            stop = zslice.stop

            missing_section_ids = np.loadtxt(
                self.missing_section_ids_file_name, dtype='int64')
            for z in missing_section_ids:
                if z > stop:
                    # the section ID list was supposed to be ordered ascendingly
                    break
                elif z >= start and z <= stop:
                    self.input[z - self.input.global_offset[0], :, :] = 0
    
    def _mask_input(self):
        input_mask = self._read_mask(self.input_mask_layer_path, 
                                     self.input_mask_mip, self.input_mip,
                                     self.input_bbox, 
                                     inverse_mask=self.inverse_input_mask)
        if np.alltrue(input_mask == 0):
            print('the mask is all black, mask all the voxels directly')
            self.input = 0
            return
        if np.all(input_mask):
            print("mask elements are all positive, return directly")
            return
        if np.alltrue(self.input==0):
            print("input volume is all black, return directly")
            return

        # print("perform masking ...")
        assert np.any(input_mask)
        
        # make it the same type with input 
        input_mask = input_mask.astype(self.input.dtype)

        # print("upsampling mask ...")
        # upsampling factor in XY plane
        mask = np.zeros(self.input.shape, dtype=self.input.dtype)
        xyfactor = 2**(self.input_mask_mip - self.input_mip)
        for offset in np.ndindex((xyfactor, xyfactor)):
            mask[:, np.s_[offset[0]::xyfactor], 
                 np.s_[offset[1]::xyfactor]] = input_mask

        np.multiply(self.input, mask, out=self.input)

    def _mask_output(self):
        output_mask = self._read_mask(self.output_mask_layer_path, 
                                      self.output_mask_mip, self.output_mip, 
                                      self.output_bbox,
                                      inverse_mask=self.inverse_output_mask)
        # if the mask is black, no need to run inference
        if np.all(output_mask == 0):
            return

        if np.all(output_mask):
            print("mask elements are all positive, return directly")
            return
        if not np.any(self.output):
            print("output volume is all black, return directly")
            return

        print("perform masking ...")
        assert np.any(output_mask)
        
        # make it the same type with output  
        output_mask = output_mask.astype(self.output.dtype)

        print("upsampling mask ...")
        # upsampling factor in XY plane
        mask = np.zeros(self.output.shape[1:], dtype=self.output.dtype)
        xyfactor = 2**(self.output_mask_mip - self.input_mip)
        for offset in np.ndindex((xyfactor, xyfactor)):
            mask[:, np.s_[offset[0]::xyfactor], np.
                 s_[offset[1]::xyfactor]] = output_mask

        assert mask.shape == self.output.shape[1:]
        for channel in range(self.output.shape[0]):
            np.multiply(self.output[channel, :, :, :], mask, 
                        out=self.output[channel, :, :, :])
        assert np.any(self.output)

    def _read_input(self):
        self.input_vol = CloudVolume(
            self.input_layer_path,
            bounded=False,
            fill_missing=self.fill_input_missing,
            progress=self.show_progress,
            mip=self.input_mip,
            parallel=False)
        output_slices = self.output_bbox.to_slices()
        self.input_slices = tuple(
            slice(s.start - m, s.stop + m)
            for s, m in zip(output_slices, self.cropping_margin_size))
        # always reverse the indexes since cloudvolume use x,y,z indexing
        self.input = self.input_vol[self.input_slices[::-1]]
        # the cutout is fortran ordered, so need to transpose and make it C order
        self.input = np.transpose(self.input)
        self.input = np.ascontiguousarray(self.input)
        assert self.input.shape[0] == 1
        self.input = np.squeeze(self.input, axis=0)
        global_offset = tuple(s.start for s in self.input_slices)

        self.input = OffsetArray(self.input, global_offset=global_offset)

    def _validate_input(self):
        """
        check that all the input voxels was downloaded without black region  
        We have found some black regions in previous inference run, 
        so hopefully this will solve the problem.
        """
        if self.input_validate_mip is None:
            print('no validate mip parameter defined, skiping validation')
            return

        # only use the region corresponds to higher mip level
        # clamp the surrounding regions in XY plane
        # this assumes that the input dataset was downsampled starting from the
        # beginning offset in the info file
        global_offset = self.input.global_offset

        # factor3 follows xyz order in CloudVolume
        factor3 = np.array([
            2**(self.input_validate_mip - self.input_mip), 2**
            (self.input_validate_mip - self.input_mip), 1
        ],
                           dtype=np.int32)
        clamped_offset = tuple(go + f - (go - vo) % f for go, vo, f in zip(
            global_offset[::-1], self.input_vol.voxel_offset, factor3))
        clamped_stop = tuple(go + s - (go + s - vo) % f
                             for go, s, vo, f in zip(
                                 global_offset[::-1], self.input.shape[::-1],
                                 self.input_vol.voxel_offset, factor3))
        clamped_slices = tuple(
            slice(o, s) for o, s in zip(clamped_offset, clamped_stop))
        clamped_bbox = Bbox.from_slices(clamped_slices)
        clamped_input = self.input.cutout(clamped_slices[::-1])
        # transform to xyz order
        clamped_input = np.transpose(clamped_input)
        # get the corresponding bounding box for validation
        validate_bbox = self.input_vol.bbox_to_mip(
            clamped_bbox, mip=self.input_mip, to_mip=self.input_validate_mip)
        #validate_bbox = clamped_bbox // factor3

        # downsample the input using avaraging
        # keep the z as it is since the mip only applies to xy plane
        # recursivly downsample the input
        # if we do it directly, the downsampled input will not be the same with the recursive one
        # because of the rounding error of integer division
        for _ in range(self.input_validate_mip - self.input_mip):
            clamped_input = downsample_with_averaging(
                clamped_input, np.array([2, 2, 1], dtype=np.int32))

        # validation by template matching
        result = validate_by_template_matching(clamped_input)
        if result is False:
            # there is an error
            # save the log to error directory
            log_path = os.path.join(self.output_layer_path, 'error')
            self._upload_log(log_path)

        validate_vol = CloudVolume(
            self.input_layer_path,
            bounded=False,
            fill_missing=False,
            progress=self.show_progress,
            mip=self.input_validate_mip,
            parallel=False)
        validate_input = validate_vol[validate_bbox.to_slices()]
        assert validate_input.shape[3] == 1
        validate_input = np.squeeze(validate_input, axis=3)

        # use the validate input to check the downloaded input
        assert np.alltrue(validate_input == clamped_input)
    
        def _crop(self):
            self.output = self.output[:, self.cropping_margin_size[0]:self.output.
                                  shape[1] - self.cropping_margin_size[0], self
                                  .cropping_margin_size[1]:self.output.
                                  shape[2] - self.cropping_margin_size[1], self
                                  .cropping_margin_size[2]:self.output.
                                  shape[3] - self.cropping_margin_size[2]]

    def _upload_output(self):
        # this is for fast test
        #self.output = np.transpose(self.output)
        #return

        vol = CloudVolume(
            self.output_layer_path,
            fill_missing=True,
            bounded=False,
            autocrop=True,
            mip=self.input_mip,
            progress=self.show_progress)
        output_slices = self.output_bbox.to_slices()
        # transpose czyx to xyzc order
        self.output = np.transpose(self.output)
        vol[output_slices[::-1] +
            (slice(0, self.output.shape[-1]), )] = self.output

    def _create_output_thumbnail(self):
        """
        quantize the affinitymap and downsample to higher mip level 
        upload the data for visual inspection.
        """
        thumbnail_path = os.path.join(self.output_layer_path, 'thumbnail')
        thumbnail_vol = CloudVolume(
            thumbnail_path,
            compress='gzip',
            fill_missing=True,
            bounded=False,
            autocrop=True,
            mip=self.input_mip,
            progress=self.show_progress)
        # the output was already transposed to xyz/fortran order in previous step while uploading the output
        # self.output = np.transpose(self.output)

        # only use the last channel, it is the Z affinity if this is affinitymap
        output = self.output[:, :, :, -1]
        input_image = (output * 255.0).astype(np.uint8)

        # transform zyx to xyz
        output_bbox = Bbox.from_slices(self.output_bbox.to_slices()[::-1])
        shape = Vec(*(output.shape[:3]))

        downsample_and_upload(
            input_image,
            output_bbox,
            thumbnail_vol,
            shape,
            mip=self.input_mip,
            axis='z',
            skip_first=True,
            only_last_mip=True)

    def _upload_log(self, log_path):
        """
        upload internal log as a file to the same place of output 
        the file name is the output range 
        """
        print('log: ', self.log)
        if not self.log:
            print('no log found, will not do anything')
            return 

        # write to aws cloud watch 
        self.aws_cloud_watch.put_metric_data(self.log)

        log_text = json.dumps(self.log)
        
        # write to google cloud storage 
        with Storage(log_path) as storage:
            storage.put_file(
                file_path=self.output_bbox.to_filename() + '.json',
                content=log_text,
                content_type='application/json')


class InferenceExecutor(FlowBase):
    """
    run inference
    1. cutout input using cloudvolume
    2. run inference
    3. crop the margin to make the output aligned with cloud storage backend
    4. upload to cloud storage using cloudvolume
    Note that I always use z,y,x in python, but cloudvolume use x,y,z for indexing.
    So I always do a reverse of slices before indexing.
    Parameters:
        is_masked_in_device: the patch could be masked/normalized around the 
            boundary, so we only need to do summation in CPU end.
        input_validate_mip: the mip level of input used for checking whether all of 
            the nonzero voxels were downloaded or not.
        inverse_input_mask: (bool) whether need to inverse the input mask or not.
            if the input region need to be black out is non-zero in mask, then the 
            mask need to be inversed.
        inverse_output_mask: (bool) whether need to inverse the output mask or not.
            if the output region need to be black out is non-zero in mask, then the 
            mask need to be inversed.
    """
    def __init__(self,
                 input_layer_path,
                 output_layer_path,
                 convnet_model,
                 convnet_weight_path,
                 patch_size,
                 patch_overlap,
                 output_key='affinity',
                 original_num_output_channels=3,
                 num_output_channels=3,
                 framework='pytorch-multitask',
                 **kwargs):
        super().__init__(input_layer_path, output_layer_path, **kwargs)
        self.convnet_model = convnet_model
        self.convnet_weight_path = convnet_weight_path
        self.output_layer_path = output_layer_path
        self.patch_size = patch_size
        self.patch_overlap = patch_overlap
        self.output_key = output_key
        self.original_num_output_channels = original_num_output_channels
        self.num_output_channels = num_output_channels
        self.framework = framework
        
        # if the patch overlap is larger than the cropping size 
        # the patch mask will make the block value around margin incorrect
        for cs, ov in zip(self.cropping_margin_size, self.patch_overlap):
            assert cs >= ov

        if framework == 'pytorch-multitask':
            # currently only pytorch-multitask support in device masking.
            self.is_masked_in_device = True
        else:
            self.is_masked_in_device = False

        parameters = self.log['parameters']
        parameters['convnet_model'] = convnet_model 
        parameters['convnet_weight_path'] = convnet_weight_path 
        parameters['patch_size'] = patch_size 
        parameters['patch_overlap'] = patch_overlap 
        parameters['output_key'] = output_key 
        parameters['original_num_output_channels'] = original_num_output_channels 
        parameters['num_output_channels'] = num_output_channels 
        parameters['framework'] = framework 
        
    def _prepare_inference_engine(self):
        def _log_gpu_device():
            import torch 
            self.log['compute_device'] = torch.cuda.get_device_name(0)
        
        def _log_cpu_device():
            import platform
            self.log['compute_device'] = platform.processor() 

        # prepare for inference
        from .inference.block_inference_engine \
            import BlockInferenceEngine
        if self.framework == 'pznet':
            _log_cpu_device()
            from .inference.frameworks.pznet_patch_inference_engine \
                import PZNetPatchInferenceEngine
            patch_engine = PZNetPatchInferenceEngine(self.convnet_model, self.convnet_weight_path)
        elif self.framework == 'pytorch':
            _log_gpu_device()
            from .inference.frameworks.pytorch_patch_inference_engine \
                import PytorchPatchInferenceEngine
            patch_engine = PytorchPatchInferenceEngine(
                self.convnet_model,
                self.convnet_weight_path,
                patch_size=self.patch_size,
                output_key=self.output_key,
                num_output_channels=self.num_output_channels)
        elif self.framework == 'pytorch-multitask':
            _log_gpu_device()
            from .inference.frameworks.pytorch_multitask_patch_inference \
                import PytorchMultitaskPatchInferenceEngine
            patch_engine = PytorchMultitaskPatchInferenceEngine(
                self.convnet_model,
                self.convnet_weight_path,
                patch_size=self.patch_size,
                output_key=self.output_key,
                patch_overlap=self.patch_overlap,
                original_num_output_channels=self.original_num_output_channels,
                num_output_channels=self.num_output_channels)
        elif self.framework == 'identity':
            _log_cpu_device()
            from .inference.frameworks.identity_patch_inference_engine \
                import IdentityPatchInferenceEngine
            patch_engine = IdentityPatchInferenceEngine(num_output_channels=3)
        else:
            raise Exception('invalid inference backend: {}'.format(
                self.framework))

        self.block_inference_engine = BlockInferenceEngine(
            patch_inference_engine=patch_engine,
            patch_size=self.patch_size,
            patch_overlap=self.patch_overlap,
            output_key=self.output_key,
            num_output_channels=self.num_output_channels,
            is_masked_in_device=self.is_masked_in_device)

    def _process_chunk(self):
        # inference engine input is a OffsetArray rather than normal numpy array
        # it is actually a numpy array with global offset

        # build the inference engine
        self._prepare_inference_engine()

        input_offset = tuple(s.start for s in self.input_slices)
        input_chunk = OffsetArray(self.input, global_offset=input_offset)
        self.output = self.block_inference_engine(input_chunk)


