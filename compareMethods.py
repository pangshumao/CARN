import numpy as np
import scipy.io as sio
import os
from utils.preprocessing import scale2actual
from scipy.stats import ttest_rel as ttest

rootDir = 'F:\\ImageData\\spine\\aligned(results from graham-previous landmarks)\\DenseNet-Gan\\disc_vb_height\\results'
foldDir = 'F:\\ImageData\\spine\\aligned(results from graham-previous landmarks)\\DenseNet-Gan'
outDir = 'F:\\ImageData\spine\\aligned(results from graham-previous landmarks)\\DenseNet-Gan'
depth = '8'
method1 = 'GCNN'
method2 = 'CNN'
subDir2 = method2 + '_growth_rate_48_depth_' + depth + '_dataset_SPINE_gamma_0.5_lambda_g_0.0_lr_0.06'
subDir1 = method1 + '_growth_rate_48_depth_' + depth + '_dataset_SPINE_gamma_0.5_lambda_g_0.0_lr_0.06_knn_20_laeweight_7.0'

data1 = np.load('/'.join([rootDir, subDir1, 'total_act_val_err.npy']))
data2 = np.load('/'.join([rootDir, subDir2, 'total_act_val_err.npy']))
results = ((data2 - data1) > 0.5)
# sio.savemat('/'.join([outDir, 'compare.mat']), {'results':results})
caseInd = 114
indiceInd = 16
print(data1[caseInd - 1,indiceInd - 1])
print(data2[caseInd - 1,indiceInd - 1])
