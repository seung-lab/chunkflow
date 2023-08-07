#%%
from itertools import product

# import numpy as np
import matplotlib.pyplot as plt

from chunkflow.chunk import Chunk
from chunkflow.flow.divid_conquer.transform import TransformSequences

# img = np.zeros(shape=(4, 128, 128), dtype=np.uint8)
# img[:, 60:68, :] = 255
# img = Chunk.create()
img = Chunk.from_h5('/Users/jwu/dropbox/40_gt/13_wasp_sample3/vol_03700/img_zyx_3600-4056_4900-5356_4150-4606.h5')
# img = img.array

transform_sequences = TransformSequences()

transformed_images = transform_sequences.forward(img)
inversed_images = transform_sequences.backward(transformed_images)


fig, axs = plt.subplots(4, 4, figsize=(15,15))

for x, y in product(range(4), range(4)):
    idx = y*4 + x
    arr = transformed_images[idx]
    axs[x, y].imshow(arr[0,...], cmap='gray')


fig.show()


# %%
