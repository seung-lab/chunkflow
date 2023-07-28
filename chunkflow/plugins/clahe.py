from tqdm import tqdm
import cv2

from chunkflow.chunk import Chunk

clipLimit = 20.0
tileGridSize = (512, 512)

#clahe = cv2.createCLAHE(clipLimit=5., tileGridSize = (18, 26))
clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize = tileGridSize)

def execute(img: Chunk):
    assert img.is_image
    for z in tqdm(range(img.shape[-3])):
        img.array[...,z,:,:] = clahe.apply(img.array[...,z,:,:])
    return img
