import numpy as np
import os
from scipy.stats import ttest_rel as ttest

# rootDir = 'I:\\ImageData\\spine\\aligned\\DenseNet-Gan\\disc_vb_height\\results'
# subDir1 = 'CNN_growth_rate_48_depth_16_dataset_SPINE_gamma_0.5_lambda_g_0.0_lr_0.1'
# subDir2 = 'GCNN_growth_rate_48_depth_16_dataset_SPINE_gamma_0.5_lambda_g_0.0_lr_0.1'

rootDir = 'H:\\ImageData\\spine\\allData\\DenseNet-Gan-allData\\disc_vb_height\\results'
subDir1 = 'GCNN-SDNE_depth_8_dataset_SPINE+_gamma_0.5_lambda_g_0.0_lr_0.04_knn_20_laeweight_101.0_sdneweight_0.005'
subDir2 = 'GCNN-SDNE_depth_8_dataset_SPINE+_gamma_0.5_lambda_g_0.0_lr_0.04_knn_20_laeweight_0.0_sdneweight_0.005'

# rootDir2 = 'H:\\ImageData\\spine\\allData\\DenseNet-Gan-allData-auNum\\disc_vb_height\\results'
# subDir1 = 'CNN-SDNE_depth_8_dataset_SPINE+_gamma_0.5_lambda_g_0.0_lr_0.04_knn_20_laeweight_101.0_sdneweight_0.001'
# subDir2 = 'GCNN_depth_7_dataset_SPINE+_gamma_0.5_lambda_g_0.0_lr_0.04_knn_20_laeweight_101.0_sdneweight_0.005'
total_mae1 = np.zeros([215])
total_mae2 = np.zeros([215])
for i in range(1,6):
    data1 = np.load(os.sep.join([rootDir, subDir1, 'predict-fold' + str(i) + '.npz']))
    data2 = np.load(os.sep.join([rootDir, subDir2, 'predict-fold' + str(i) + '.npz']))
    total_mae1[(i-1) * 43 : i * 43] = data1['val_mae_err']
    total_mae2[(i - 1) * 43: i * 43] = data2['val_mae_err']

s, p = ttest(total_mae1, total_mae2)
mean_mae1 = np.mean(total_mae1)
mean_mae2 = np.mean(total_mae2)
print('mean_mae1 = ', mean_mae1)
print('mean_mae2 = ', mean_mae2)
print('s = ', s)
print('p = ', p)