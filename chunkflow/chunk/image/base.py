__doc__ = """Image chunk class"""

from tqdm import tqdm
import numpy as np

from chunkflow.chunk import Chunk
from .adjust_grey import normalize_section_shang
from .convnet.inferencer import Inferencer


class Image(Chunk):
    """
    a chunk of image volume.
    """
    def __init__(self, array: np.ndarray, **kwargs):
        super().__init__(array, **kwargs)

    @classmethod
    def from_chunk(cls, chk: Chunk):
        return cls(chk.array, **chk.properties)

    def inference(self, inferencer: Inferencer):
        """run convolutional net inference for this image chunk"""
        return inferencer(self)

    def normalize_section_shang(self, nominalmin, nominalmax, clipvalues):
        return normalize_section_shang(self.array, nominalmin, nominalmax,
                                       clipvalues)

    def _find_section_clamping_values(self, 
            hist: np.ndarray, lower_clip_fraction: float, upper_clip_fraction: float):
        """compute the clamping values for each section."""
        # remove the np.copy from original code since we only need this once
        filtered = hist

        # remove pure black from frequency counts as
        # it has no information in our images
        filtered[0] = 0

        cdf = np.zeros(shape=(len(filtered), ), dtype=np.uint64)
        cdf[0] = filtered[0]
        for i in range(1, len(filtered)):
            cdf[i] = cdf[i - 1] + filtered[i]

        total = cdf[-1]

        if total == 0:
            return (0, 0)

        lower = 0
        for i, val in enumerate(cdf):
            if float(val) / float(total) > lower_clip_fraction:
                break
            lower = i

        upper = 0
        for i, val in enumerate(cdf):
            if float(val) / float(total) > 1 - upper_clip_fraction:
                break
            upper = i
        
        return lower, upper
    
    def _hist_to_lookup_table(self, 
            hist: np.ndarray, lower_clip_fraction: float, upper_clip_fraction: float,
            minval: int = 1,
            maxval: int = 255):
        """histogram to lookup table

        Args:
            hist (np.ndarray): histogram

        Returns:
            np.ndarray: lookup table
        """
        lower, upper = self._find_section_clamping_values(
            hist, lower_clip_fraction, upper_clip_fraction)
             
        if lower == upper:
            #lookup_table = np.arange(0, 256, dtype=np.uint8)
            # no need to perform any transform
            return None
        else:
            # compute the lookup table
            lookup_table = np.arange(0, 256, dtype=np.float32)
            lookup_table = (lookup_table - float(lower)) * (
                maxval / (float(upper) - float(lower)))
            np.clip(lookup_table, minval, maxval, out=lookup_table)
            lookup_table = np.round(lookup_table)
            lookup_table = lookup_table.astype( np.uint8 ) 
            return lookup_table


    def normalize_contrast(self, 
            lower_clip_fraction: float = 0.01, 
            upper_clip_fraction: float = 0.01,
            minval: int = 1,
            maxval: int = 255,
            per_section: bool = True
            ):

        def _normalize_array(array: np.ndarray, 
                lower_clip_fraction: float, 
                upper_clip_fraction: float,
                minval: int = 1,
                maxval: int = 255):
            hist = np.bincount(array.flatten(), minlength=255)
            lookup_table = self._hist_to_lookup_table(
                hist, lower_clip_fraction, upper_clip_fraction, 
                minval=minval, maxval=maxval)
            if lookup_table is not None:
                array = lookup_table[array]
            return array
        

        if per_section:
            for z in tqdm(range(self.bbox.start.z, self.bbox.stop.z)):
                slices = (slice(z, z+1), *self.slices[-2:])
                section = self.cutout(slices)
                # section = Image.from_chunk(section)
                section.array = _normalize_array(
                    section.array,
                    lower_clip_fraction, upper_clip_fraction,
                    minval=minval, maxval=maxval
                )
                self.save(section)
            else:
                # chunk = Image.from_chunk(chunk)
                self.array = _normalize_array(
                    self.array,
                    lower_clip_fraction, upper_clip_fraction,
                    minval=minval, maxval=maxval 
                )
