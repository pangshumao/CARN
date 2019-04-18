import numpy as np
import os
import scipy.io as sio
import pylab as plt
import math
from utils.preprocessing import actual2scale, imResize
imagesDir = 'F:\\ImageData\\spine\\allData\\aligned\\MR'
maskDir = 'F:\\ImageData\\spine\\allData\\aligned\\label'
distDir = 'F:\\ImageData\\spine\\allData\\aligned\\dists'
outDir = 'F:\\ImageData\\spine\\allData\\aligned\\DenseNet-Gan'
samplesNum = 215
height = 512
width = 256
outDim = 30
y_mode = 'disc_vb_height'
is_align = True
use_act_y = False


in_y = np.zeros((samplesNum, outDim), dtype=np.float32)
if y_mode == 'disc_height':
    distsData = sio.loadmat('/'.join([distDir, 'disc_actual_height.mat']))
    in_y = distsData['disc_height']
elif y_mode == 'vb_height':
    distsData = sio.loadmat('/'.join([distDir, 'vb_actual_height.mat']))
    in_y = distsData['vb_height']
elif y_mode == 'disc_vb_height':
    distsData = sio.loadmat('/'.join([distDir, 'disc_actual_height.mat']))
    disc_y = distsData['disc_height']
    distsData = sio.loadmat('/'.join([distDir, 'vb_actual_height.mat']))
    vb_y = distsData['vb_height']
    in_y = np.concatenate((disc_y, vb_y), axis=1)
in_pixelSize = distsData['pixelSize']

def augment_image(image, mask):
    image_shape = image.shape
    new_image = image + np.random.normal(0, 0.03, size=image_shape) * mask
    # new_image = image - np.random.rand(image_shape[0], image_shape[1]) * mask * 0.1
    return new_image

x = np.zeros((samplesNum, height, width), dtype=np.float32)
y = np.zeros((samplesNum, outDim), dtype=np.float32)
masks = np.zeros((samplesNum, height, width), dtype=np.float32)
pixelSizes = np.zeros((samplesNum, 2), dtype=np.float32)
for i in range(0, samplesNum):
    imData = sio.loadmat('/'.join([imagesDir, 'case' + str(i + 1) + '.mat']))
    maskData = sio.loadmat('/'.join([maskDir, 'case' + str(i + 1) + '_disc.mat']))
    if is_align:
        in_x = imData['I']
        x[i, :, :], pixelSizes[i, :] = imResize(in_x, height, width, in_pixelSize)
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
    else:
        x[i, :, :] = imData['I']
    if use_act_y:
        y = in_y
    else:
        if is_align:
            y[i, :] = actual2scale(x[i, :, :], in_y[i, :], pixelSizes[i, :], mode='height')
        else:
            pixelSizes[i, :] = in_pixelSize.flat
            y[i, :] = actual2scale(x[i, :, :], in_y[i, :], pixelSizes[i, :], mode='height')

# normalization
x_mean = np.mean(x)
x_std = np.std(x)
x = (x - x_mean) / x_std

x2 = x[:,:,:,np.newaxis]
masks2 = masks[:,:,:,np.newaxis]
np.savez('/'.join([outDir,y_mode, 'samples.npz']), x=x2, y=y, pixelSizes=pixelSizes, height=height, width=width, masks=masks2)
# x = np.transpose(x, axes=[1,2,0])
# masks = np.transpose(masks, axes=[1,2,0])
# sio.savemat('/'.join([outDir, y_mode, 'samples.mat']), {'x':x, 'y':y, 'pixelSizes':pixelSizes, 'height':height, 'width':width, 'masks':masks})