import logging
import os
import re
from collections import defaultdict
from time import sleep

from cloudvolume.storage import Storage
from cloudvolume import PrecomputedSkeleton

import kimimaro

from .base import OperatorBase


class AggregateSkeletonFragmentsOperator(OperatorBase):
    """Merge skeleton fragments for Neuroglancer visualization."""
    def __init__(self,
                 fragments_path: str,
                 output_path: str,
                 name: str = 'aggregate-skeleton-fragments'):
        """
        Parameters
        ------------
        fragments_path: 
            path to store fragment files
        output_path:
            save the merged skeleton file here.
        """
        super().__init__(name=name)
        self.fragments_storage = Storage(fragments_path)
        self.output_storage = Storage(output_path)
   
    def __call__(self, prefix: str):
        logging.info(f'aggregate skeletons with prefix of {prefix}')
        
        id2filenames = defaultdict(list)
        for filename in self.fragments_storage.list_files(prefix=prefix):
            filename = os.path.basename(filename)
            # `match` implies the beginning (^). `search` matches whole string
            matches = re.search(r'(\d+):', filename)
            
            if not matches:
                continue
            
            # skeleton ID
            skl_id = int(matches.group(0)[:-1])
            id2filenames[skl_id].append(filename)

        for skl_id, filenames in id2filenames.items():
            logging.info(f'skeleton id: {skl_id}')
            frags = self.fragments_storage.get_files(filenames)
            frags = [PrecomputedSkeleton.from_precomputed(x['content']) for x in frags]
            skel = PrecomputedSkeleton.simple_merge(frags).consolidate()
            skel = kimimaro.postprocess(
                skel,
                dust_threshold=1000,
                tick_threshold=3500
            )
            self.output_storage.put_file(
                file_path=str(skl_id),
                content=skel.to_precomputed(),
            )
            # the last few hundred files will not be uploaded without sleeping!
            sleep(0.01)
