import numpy as np
import scipy.io as sio
import os
from utils.preprocessing import scale2actual
from scipy.stats import ttest_rel as ttest

rootDir = 'H:\\ImageData\\spine\\allData\\DenseNet-Gan-allData\\disc_vb_height\\folder5\\loss'
subDir = 'GCNN-SDNE_depth_8_dataset_SPINE+_gamma_0.5_lambda_g_0.0_lr_0.04_knn_20_laeweight_101.0_sdneweight_0.0'
data = np.load('/'.join([rootDir, subDir, 'loss.npz']))
train_mae_loss = data['train_mae_loss']
val_mae_loss = data['val_mae_loss']
train_sdne_loss = data['train_sdne_loss']
val_sdne_loss = data['val_sdne_loss']
sio.savemat('/'.join([rootDir, subDir, 'loss.mat']), {'train_mae_loss':train_mae_loss, 'val_mae_loss':val_mae_loss, 'train_sdne_loss':train_sdne_loss, 'val_sdne_loss':val_sdne_loss})