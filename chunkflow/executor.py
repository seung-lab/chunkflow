# -*- coding: utf-8 -*-
"""Executor of inference.
"""

import numpy as np
import time
import os
import json 
from google.cloud import logging
from cloudvolume import CloudVolume, Storage
from cloudvolume.lib import Vec, Bbox
from cloudvolume.secrets import google_credentials_path, PROJECT_NAME 

from .validate import validate_by_template_matching
from .igneous.tasks import downsample_and_upload
from .igneous.downsample import downsample_with_averaging
from .offset_array import OffsetArray

class Executor(object):
    """
    run inference like ChunkFlow.jl
    1. cutout image using cloudvolume
    2. run inference
    3. crop the margin to make the output aligned with cloud storage backend
    4. upload to cloud storage using cloudvolume
    Note that I always use z,y,x in python, but cloudvolume use x,y,z for indexing.
    So I always do a reverse of slices before indexing.
    Parameters:
        is_masked_in_device: the patch could be masked/normalized around the 
            boundary, so we only need to do summation in CPU end.
        image_validate_mip: the mip level of image used for checking whether all of 
            the nonzero voxels were downloaded or not.
    """

    def __init__(self,
                 image_layer_path,
                 output_layer_path,
                 convnet_model_path,
                 convnet_weight_path,
                 patch_size,
                 patch_overlap,
                 cropping_margin_size,
                 output_key='affinity',
                 num_output_channels=3,
                 mip=1,
                 image_mask_layer_path=None,
                 output_mask_layer_path=None,
                 image_mask_mip=3,
                 output_mask_mip=3,
                 framework='pytorch-multitask',
                 missing_section_ids_file_name=None,
                 image_validate_mip=None):
        self.image_layer_path = image_layer_path
        self.convnet_model_path = convnet_model_path
        self.convnet_weight_path = convnet_weight_path
        self.output_layer_path = output_layer_path
        self.patch_size = patch_size
        self.patch_overlap = patch_overlap
        self.cropping_margin_size = cropping_margin_size
        self.output_key = output_key
        self.num_output_channels = num_output_channels
        self.image_mip = mip
        self.output_mip = mip
        self.image_mask_layer_path = image_mask_layer_path
        self.output_mask_layer_path = output_mask_layer_path
        self.image_mask_mip = image_mask_mip
        self.output_mask_mip = output_mask_mip
        self.framework = framework
        self.missing_section_ids_file_name = missing_section_ids_file_name

        # if the patch overlap is larger than the cropping size 
        # the patch mask will make the block value around margin incorrect
        for cs, ov in zip(cropping_margin_size, patch_overlap):
            assert cs >= ov

        if framework == 'pytorch-multitask':
            # currently only pytorch-multitask support in device masking.
            self.is_masked_in_device = True
        else:
            self.is_masked_in_device = False

        self.image_validate_mip = image_validate_mip

        logging_client = logging.Client.from_service_account_json(
            google_credentials_path, project=PROJECT_NAME)
        self.logger = logging_client.logger('chunkflow')
         
        self.log = dict()
        self.log['parameters']={
            'image_layer_path':         image_layer_path,
            'output_layer_path':        output_layer_path,
            'convnet_model_path':       convnet_model_path,
            'convnet_weight_path':      convnet_weight_path,
            'output_mask_layer_path':   output_mask_layer_path,
            'patch_size':               (*patch_size,),
            'patch_overlap':            (*patch_overlap,),
            'cropping_margin_size':     (*cropping_margin_size,),
            'output_key':               output_key,
            'num_output_channels':      num_output_channels,
            'mip':                      mip,
            'output_mask_mip':          output_mask_mip,
            'framework':                framework,
            'image_validate_mip':       image_validate_mip
        }

    def __call__(self, output_bbox):
        if isinstance(output_bbox, str):
            output_bbox = Bbox.from_filename(output_bbox)
        elif isinstance(output_bbox, tuple):
            output_bbox = Bbox.from_slices(output_bbox)
        else:
            assert isinstance(output_bbox, Bbox)

        self.log['output_bbox'] = output_bbox.to_filename()
        self.output_bbox = output_bbox
        output_slices = output_bbox.to_slices()
        image_slices = tuple(slice(s.start-c, s.stop+c) for s,c in 
                             zip(output_slices, self.cropping_margin_size))
        self.image_bbox = Bbox.from_slices(image_slices)

        total_start = time.time()

        start = time.time()
        self._read_image()
        elapsed = time.time() - start
        self.log['read_image_time'] = elapsed
        print("Read image takes %3f sec" % (elapsed))
        
        start = time.time()
        self._validate_image()
        elapsed = time.time() - start
        self.log['validate_image_time'] = elapsed
        print("Validate image takes %3f sec" % (elapsed))

        start = time.time()
        self._mask_missing_sections()
        elapsed = time.time() - start
        self.log['mask_missing_sections_time'] = elapsed
        print("Mask missing sections in image takes %3f sec" % (elapsed))
        
        if self.image_mask_layer_path:
            start = time.time()
            self._mask_image()
            elapsed = time.time() - start
            self.log['mask_image_time'] = elapsed
            print("Mask image takes %3f sec" % (elapsed))

        start = time.time()
        self._inference()
        elapsed = time.time() - start
        self.log['inference_time'] = elapsed
        print("Inference takes %3f min" % (elapsed / 60))

        start = time.time()
        self._crop()
        elapsed = time.time() - start
        self.log['crop_output_time'] = elapsed
        print("Cropping takes %3f sec" % (elapsed))

        if self.output_mask_layer_path:
            start = time.time()
            self._mask_output()
            elapsed = time.time() - start
            self.log['mask_output_time'] = elapsed
            print("Mask output takes %3f sec" % (elapsed))

        start = time.time()
        self._upload_output()
        elapsed = time.time() - start
        self.log['upload_output_time'] = elapsed
        print("Upload output takes %3f min" % (elapsed / 60))

        start = time.time()
        self._create_output_thumbnail()
        elapsed = time.time() - start
        self.log['create_output_thumbnail_time'] = elapsed
        print("create output thumbnail takes %3f min" % (elapsed / 60))

        total_time = time.time() - total_start
        self.log['total_time'] = total_time
        print("Whole task takes %3f min" % (total_time / 60))

        log_path = os.path.join(self.output_layer_path, 'log')
        self._upload_log(log_path)
    
    def _read_mask(self, mask_layer_path, mask_mip, mip, bbox):
        """
        bbox: the bounding box in image mip
        """
        if not mask_layer_path:
            print('no mask layer path defined')
            return None
        
        print("download mask chunk...")
        vol = CloudVolume(
            mask_layer_path,
            bounded=True,
            fill_missing=False,
            progress=False,
            mip=mask_mip)
        # assume that image mip is the same with output mip
        xyfactor = 2**(mask_mip - mip)
        # only scale the indices in XY plane
        mask_slices = tuple(
            slice(a.start // xyfactor, a.stop // xyfactor)
            for a in bbox.to_slices()[1:3])
        mask_slices = (bbox.to_slices()[0], ) + mask_slices

        # the slices did not contain the channel dimension 
        print("mask slices: {}".format(mask_slices))
        mask = vol[mask_slices[::-1]]
        mask = np.transpose(mask)
        print("shape of mask: {}".format(mask.shape))
        mask = np.squeeze(mask, axis=0)
        return mask
    
    def _mask_missing_sections(self):
        """
        mask some missing sections if the section id was provided 
        """
        if self.missing_section_ids_file_name:
            zslice = self.image.slices[0]
            start = zslice.start
            stop = zslice.stop

            missing_section_ids = np.loadtxt(
                self.missing_section_ids_file_name, dtype='int64')
            for z in missing_section_ids:
                if z > stop:
                    # the section ID list was supposed to be ordered ascendingly
                    break
                elif z >= start and z <= stop:
                    self.image[z - self.image.global_offset[0], :, :] = 0
    
    def _mask_image(self):
        image_mask = self._read_mask(self.image_mask_layer_path, 
                                     self.image_mask_mip, self.image_mip,
                                     self.image_bbox)
        if np.alltrue(image_mask == 0):
            print('the mask is all black, mask all the voxels directly')
            self.image = 0
            return
        if np.all(image_mask):
            print("mask elements are all positive, return directly")
            return
        if np.alltrue(self.image==0):
            print("image volume is all black, return directly")
            return

        print("perform masking ...")
        assert np.any(image_mask)
        
        # make it the same type with image 
        image_mask = image_mask.astype(self.image.dtype)

        print("upsampling mask ...")
        # upsampling factor in XY plane
        mask = np.zeros(self.image.shape, dtype=self.image.dtype)
        xyfactor = 2**(self.image_mask_mip - self.image_mip)
        for offset in np.ndindex((xyfactor, xyfactor)):
            mask[:, np.s_[offset[0]::xyfactor], 
                 np.s_[offset[1]::xyfactor]] = image_mask

        np.multiply(self.image, mask, out=self.image)

    def _mask_output(self):
        output_mask = self._read_mask(self.output_mask_layer_path, 
                                      self.output_mask_mip, self.output_mip, 
                                      self.output_bbox)
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
        xyfactor = 2**(self.output_mask_mip - self.image_mip)
        for offset in np.ndindex((xyfactor, xyfactor)):
            mask[:, np.s_[offset[0]::xyfactor], np.
                 s_[offset[1]::xyfactor]] = output_mask

        assert mask.shape == self.output.shape[1:]
        for channel in range(self.output.shape[0]):
            np.multiply(self.output[channel, :, :, :], mask, 
                        out=self.output[channel, :, :, :])
        assert np.any(self.output)

    def _read_image(self):
        self.image_vol = CloudVolume(
            self.image_layer_path,
            bounded=True,
            fill_missing=False,
            progress=False,
            mip=self.image_mip,
            parallel=False)
        output_slices = self.output_bbox.to_slices()
        self.input_slices = tuple(
            slice(s.start - m, s.stop + m)
            for s, m in zip(output_slices, self.cropping_margin_size))
        # always reverse the indexes since cloudvolume use x,y,z indexing
        self.image = self.image_vol[self.input_slices[::-1]]
        # the cutout is fortran ordered, so need to transpose and make it C order
        self.image = np.transpose(self.image)
        self.image = np.ascontiguousarray(self.image)
        assert self.image.shape[0] == 1
        self.image = np.squeeze(self.image, axis=0)
        global_offset = tuple(s.start for s in self.input_slices)

        self.image = OffsetArray(self.image, global_offset=global_offset)

    def _validate_image(self):
        """
        check that all the image voxels was downloaded without black region  
        We have found some black regions in previous inference run, 
        so hopefully this will solve the problem.
        """
        if self.image_validate_mip is None:
            print('no validate mip parameter defined, skiping validation')
            return

        # only use the region corresponds to higher mip level
        # clamp the surrounding regions in XY plane
        # this assumes that the image dataset was downsampled starting from the
        # beginning offset in the info file
        global_offset = self.image.global_offset

        # factor3 follows xyz order in CloudVolume
        factor3 = np.array([
            2**(self.image_validate_mip - self.image_mip), 2**
            (self.image_validate_mip - self.image_mip), 1
        ],
                           dtype=np.int32)
        clamped_offset = tuple(go + f - (go - vo) % f for go, vo, f in zip(
            global_offset[::-1], self.image_vol.voxel_offset, factor3))
        clamped_stop = tuple(go + s - (go + s - vo) % f
                             for go, s, vo, f in zip(
                                 global_offset[::-1], self.image.shape[::-1],
                                 self.image_vol.voxel_offset, factor3))
        clamped_slices = tuple(
            slice(o, s) for o, s in zip(clamped_offset, clamped_stop))
        clamped_bbox = Bbox.from_slices(clamped_slices)
        clamped_image = self.image.cutout(clamped_slices[::-1])
        # transform to xyz order
        clamped_image = np.transpose(clamped_image)
        # get the corresponding bounding box for validation
        validate_bbox = self.image_vol.bbox_to_mip(
            clamped_bbox, mip=self.image_mip, to_mip=self.image_validate_mip)
        #validate_bbox = clamped_bbox // factor3

        # downsample the image using avaraging
        # keep the z as it is since the mip only applies to xy plane
        # recursivly downsample the image
        # if we do it directly, the downsampled image will not be the same with the recursive one
        # because of the rounding error of integer division
        for _ in range(self.image_validate_mip - self.image_mip):
            clamped_image = downsample_with_averaging(
                clamped_image, np.array([2, 2, 1], dtype=np.int32))

        # validation by template matching
        result = validate_by_template_matching(clamped_image)
        if result is False:
            # there is an error
            # save the log to error directory
            log_path = os.path.join(self.output_layer_path, 'error')
            self._upload_log(log_path)

        validate_vol = CloudVolume(
            self.image_layer_path,
            bounded=True,
            fill_missing=False,
            progress=False,
            mip=self.image_validate_mip,
            parallel=False)
        validate_image = validate_vol[validate_bbox.to_slices()]
        assert validate_image.shape[3] == 1
        validate_image = np.squeeze(validate_image, axis=3)

        # use the validate image to check the downloaded image
        assert np.alltrue(validate_image == clamped_image)
    
    def _prepare_inference_engine(self):
        def _log_device():
            import torch 
            self.log['device'] = torch.cuda.get_device_name(0)

        # prepare for inference
        from chunkflow.block_inference_engine import BlockInferenceEngine
        if self.framework == 'pznet':
            from chunkflow.frameworks.pznet_patch_inference_engine import PZNetPatchInferenceEngine
            patch_engine = PZNetPatchInferenceEngine(self.convnet_model_path, self.convnet_weight_path)
        elif self.framework == 'pytorch':
            _log_device()
            from chunkflow.frameworks.pytorch_patch_inference_engine import PytorchPatchInferenceEngine
            patch_engine = PytorchPatchInferenceEngine(
                self.convnet_model_path,
                self.convnet_weight_path,
                patch_size=self.patch_size,
                output_key=self.output_key,
                num_output_channels=self.num_output_channels)
        elif self.framework == 'pytorch-multitask':
            _log_device()
            from chunkflow.frameworks.pytorch_multitask_patch_inference import PytorchMultitaskPatchInferenceEngine
            patch_engine = PytorchMultitaskPatchInferenceEngine(
                self.convnet_model_path,
                self.convnet_weight_path,
                patch_size=self.patch_size,
                output_key=self.output_key,
                patch_overlap=self.patch_overlap,
                num_output_channels=self.num_output_channels)
        elif self.framework == 'identity':
            from chunkflow.frameworks.identity_patch_inference_engine import IdentityPatchInferenceEngine
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

    def _inference(self):
        # inference engine input is a OffsetArray rather than normal numpy array
        # it is actually a numpy array with global offset

        # build the inference engine
        self._prepare_inference_engine()

        input_offset = tuple(s.start for s in self.input_slices)
        input_chunk = OffsetArray(self.image, global_offset=input_offset)
        self.output = self.block_inference_engine(input_chunk)

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
            bounded=True,
            autocrop=True,
            mip=self.image_mip,
            progress=False)
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
            bounded=True,
            autocrop=True,
            mip=self.image_mip,
            progress=True)
        # the output was already transposed to xyz/fortran order in previous step while uploading the output
        # self.output = np.transpose(self.output)

        # only use the last channel, it is the Z affinity if this is affinitymap
        output = self.output[:, :, :, -1]
        image = (output * 255.0).astype(np.uint8)

        # transform zyx to xyz
        output_bbox = Bbox.from_slices(self.output_bbox.to_slices()[::-1])
        shape = Vec(*(output.shape[:3]))

        downsample_and_upload(
            image,
            output_bbox,
            thumbnail_vol,
            shape,
            mip=self.image_mip,
            axis='z',
            skip_first=True,
            only_last_mip=True)

    def _upload_log(self, log_path):
        """
        upload internal log as a file to the same place of output 
        the file name is the output range 
        """
        log_text = json.dumps(self.log)
        
        # write to google cloud stack driver
        self.logger.log_text(log_text)

        # write to google cloud storage 
        with Storage(log_path) as storage:
            storage.put_file(
                file_path=self.output_bbox.to_filename() + '.json',
                content=log_text,
                content_type='application/json')
