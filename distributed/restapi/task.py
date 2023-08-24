from __future__ import annotations

import json
import math

from functools import cached_property
import multiprocessing as mp

import psutil
import GPUtil

from chunkflow.lib.cartesian_coordinate import BoundingBox, Cartesian


class ComputationalResource:
    def __init__(self, cores: int, ram: int, 
            gpus: int = None, gpu_ram: int = None) -> None:
        """the required computational resources

        Args:
            cores (int): the number of CPU cores
            ram (int): the minimum size of RAM (GB)
            gpus (int): the number of GPUs.
            gpu_ram (int): the minimum size of GPU RAM (GB)
        """
        self.cores = cores
        self.ram = ram
        self.gpus = gpus
        self.gpu_ram = gpu_ram
    
    @cached_property
    def is_satisfied(self, max_availability: float=1.0) -> bool:
        gpus = GPUtil.getGPUs(
            maxMemory=max_availability, maxLoad=max_availability)

        return self.cores <= mp.cpu_count() and \
            self.ram <= psutil.virtual_memory().available // 1e9 and \
            len(gpus)>=self.gpus if (
                self.gpus is not None and self.gpus>0) else True

    @property
    def dict(self):
        return {
            'cores': self.cores,
            'ram': self.ram,
            'gpus': self.gpus,
            'gpu_ram': self.gpu_ram
        }


class Task:
    def __init__(self, bbox: BoundingBox, resource: ComputationalResource = None) -> None:
        assert isinstance(bbox, BoundingBox)

        self.bbox = bbox
        self.resource = resource






class SpatialTaskTree:
    def __init__(self, bbox: BoundingBox, block_size: Cartesian, parent: SpatialTaskTree = None) -> None:
        self.bbox = bbox
        self.parent = parent
        self.block_size = block_size

        # find the axis with maximum number of blocks
        block_num = [math.ceil(x) for x in (bbox.shape / block_size)]
        max_block_num = max(block_num)
        axis = block_num.index(max_block_num)


        if block_num[axis] == 1:
            # this is already the leaf chunk, no spliting any more
            # leaf chunk do not have any dependent children, so it 
            # is naturally ready to execute
            self.state = 'ready'
            self.left = None
            self.right = None
            return
        else:
            self.state = 'not ready'
            left_size = (block_num[axis] // 2) * block_size[axis]
            left_shape = [*bbox.shape]
            left_shape[axis] = left_size
            left_bbox = BoundingBox.from_delta(bbox.start, left_shape)
            self.left = SpatialTaskTree(left_bbox, block_size, parent=self)

            right_start = [*bbox.start]
            right_start[axis] = left_bbox.stop[axis]
            right_bbox = BoundingBox(right_start, bbox.stop)
            self.right = SpatialTaskTree(right_bbox, block_size, parent=self)
            return

    @classmethod
    def from_json(cls, data: str):
        pass

    @property
    def json(self):
        if self.left is None:
            left = 'None'
        else:
            left = self.left.bbox.string
        
        if self.right is None:
            right = 'None'
        else:
            right = self.right.bbox.string

        if self.parent is None:
            parent = 'None'
        else:
            parent = self.parent.bbox.string

        return json.dumps({
            'state': self.state,
            'bbox': self.bbox.string,
            'block_size': self.block_size,
            'left': left,
            'right': right,
            'parent': parent,
        })

    @property
    def is_leaf(self):
        return self.left is None and self.right is None

    def set_state_working_on(self):
        self.state = 'working on'

    def try_to_set_parent_done(self):
        if self.parent.left.is_done and self.parent.right.is_done:
            self.parent.set_state_done()

    def set_state_done(self):
        self.state = 'done'
        self.try_to_set_parent_done()

    @property
    def is_done(self):
        return self.state == 'done'

    @property
    def leaf_list(self):
        """the list of leaf bounding boxes

        Returns:
            list: the bounding box list
        """
        lst = []
        if self.left.is_leaf:
            lst.append(self.left)
        else:
            lst.extend(self.left.leaf_list)

        if self.right.is_leaf:
            lst.append(self.right)
        else:
            lst.extend(self.right.leaf_list)
        return lst
