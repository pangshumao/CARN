import numpy as np
import os
import scipy.io as sio
import pylab as plt
import math
from utils.preprocessing import actual2scale, imResize
maskDir = 'J:\\ImageData\\spine\\aligned\\label'
distDir = 'J:\\ImageData\\spine\\aligned\\dists'
outDir = 'J:\\ImageData\\spine\\aligned\\DenseNet'
samplesNum = 195
height = 512
width = 256
y_mode = 'disc_vb_height'
distsData = sio.loadmat('/'.join([distDir, 'disc_actual_height.mat']))
in_pixelSize = distsData['pixelSize']
masks = np.zeros((samplesNum, height, width), dtype=np.float32)
for i in range(0, samplesNum):
    maskData = sio.loadmat('/'.join([maskDir, 'case' + str(i + 1) + '_disc.mat']))
    mask = maskData['label_crop']
    mask[mask > 0] = 1
    mask = np.float32(mask)
    mask, _ = imResize(mask, height, width, in_pixelSize)
    mask[mask > 0] = 1
    mask = np.float32(mask)
    # plt.figure()
    # plt.imshow(augment_image(x[i], mask), cmap='gray')
    # plt.axis('off')
    # plt.show()
    # plt.close()
    masks[i, :, :] = mask

masks = masks[:,:,:,np.newaxis]
sampleData = np.load('/'.join([outDir,y_mode, 'samples.npz']))
x = sampleData['x']
y = sampleData['y']
pixelSizes = sampleData['pixelSizes']
np.savez('/'.join([outDir,y_mode, 'samples.npz']), x=x, y=y, pixelSizes=pixelSizes, height=height, width=width, masks=masks)