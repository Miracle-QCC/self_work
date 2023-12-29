import os
import numpy as np
import binvox_rw
with open('/NMC/groundtruth/hhhh+(1).binvox', 'rb') as f:
    voxels = binvox_rw.read_as_3d_array(f)
voxels = voxels.data.astype(np.float32)

print(voxels.shape)
