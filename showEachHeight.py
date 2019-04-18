import numpy as np
import scipy.io as sio
import os
from utils.preprocessing import scale2actual
from scipy.stats import ttest_rel as ttest

# rootDir = 'F:\\ImageData\\spine\\aligned(results from graham-previous landmarks)\\DenseNet-Gan\\disc_vb_height\\results'
# foldDir = 'F:\\ImageData\\spine\\aligned(results from graham-previous landmarks)\\DenseNet-Gan'
# depth = '8'
# method = 'GCNN'
# method = 'CNN'
# subDir = method + '_growth_rate_48_depth_' + depth + '_dataset_SPINE_gamma_0.5_lambda_g_0.0_lr_0.06'
# subDir = method + '_growth_rate_48_depth_' + depth + '_dataset_SPINE_gamma_0.5_lambda_g_0.0_lr_0.06_knn_20_laeweight_7.0'

rootDir = 'H:\\ImageData\\spine\\allData\\DenseNet-Gan-allData\\disc_vb_height\\results'
foldDir = 'H:\\ImageData\\spine\\allData\\DenseNet-Gan-allData'
depth = '8'
subDir = 'GCNN-SDNE_depth_8_dataset_SPINE+_gamma_0.5_lambda_g_0.0_lr_0.04_knn_20_laeweight_101.0_sdneweight_0.005'

# rootDir = 'H:\\ImageData\\spine\\allData\\DenseNet-Gan-allData\\disc_vb_height\\results'
# foldDir = 'H:\\ImageData\\spine\\allData\\DenseNet-Gan-allData'
# subDir = 'DenseNet_depth_26_dataset_SPINE+_gamma_0.5_lambda_g_0.0_lr_0.04_knn_20_laeweight_0.0_sdneweight_0.0'

# rootDir = 'H:\\ImageData\\spine\\allData\\DenseNet-Gan-allData-auNum\\disc_vb_height\\results'
# foldDir = 'H:\\ImageData\\spine\\allData\\DenseNet-Gan-allData-auNum'
# subDir = 'GCNN_depth_7_dataset_SPINE+_gamma_0.5_lambda_g_0.0_lr_0.04_knn_20_laeweight_101.0_sdneweight_0.005'

total_act_train_pre_y = np.zeros([860, 30])
total_act_train_y = np.zeros([860, 30])
total_act_val_pre_y = np.zeros([215, 30])
total_act_val_y = np.zeros([215, 30])
for i in range(1,6):
    data = np.load(os.sep.join([rootDir, subDir, 'predict-fold' + str(i) + '.npz']))
    pixelSizes = data['pixelSizes']
    height = data['height']
    width = data['width']
    train_pre_y = data['train_pre_y']
    train_y = data['train_y']
    val_pre_y = data['val_pre_y']
    val_y = data['val_y']
    foldData = sio.loadmat(os.sep.join([foldDir, 'fold' + str(i) + '-ind.mat']))
    trainInd = foldData['trainInd'].flatten()
    valInd = foldData['valInd'].flatten()
    act_train_pre_y = scale2actual(train_pre_y, pixelSizes[trainInd, :], np.tile(height, train_pre_y.shape),
                                 np.tile(width, train_pre_y.shape), mode='height')
    act_train_y = scale2actual(train_y, pixelSizes[trainInd, :], np.tile(height, train_y.shape),
                             np.tile(width, train_y.shape), mode='height')
    act_val_pre_y = scale2actual(val_pre_y, pixelSizes[valInd, :], np.tile(height, val_pre_y.shape),
                                 np.tile(width, val_pre_y.shape), mode='height')
    act_val_y = scale2actual(val_y, pixelSizes[valInd, :], np.tile(height, val_y.shape),
                                 np.tile(width, val_y.shape), mode='height')
    total_act_train_pre_y[(i - 1) * 172: i * 172, :] = act_train_pre_y
    total_act_train_y[(i - 1) * 172: i * 172, :] = act_train_y
    total_act_val_pre_y[(i-1) * 43 : i * 43, :] = act_val_pre_y
    total_act_val_y[(i - 1) * 43: i * 43, :] = act_val_y
    pass

total_act_train_err = np.abs(total_act_train_pre_y - total_act_train_y)
total_act_val_err = np.abs(total_act_val_pre_y - total_act_val_y)
temp = np.mean(total_act_train_err, axis=1)
print(temp[172*3+76])
np.save('/'.join([rootDir, subDir, 'total_act_val_err.npy']), total_act_val_err)
train_mean_mae = np.mean(total_act_train_err, axis=0)
val_mean_mae = np.mean(total_act_val_err, axis=0)
print('train mean_mae = ', train_mean_mae)
print('val mean_mae = ', val_mean_mae)
sio.savemat(os.sep.join([rootDir, subDir, 'total_act_val_err.mat']), {'total_act_val_err':total_act_val_err})
train_disc_mae = np.mean(total_act_train_err[:, :15])
train_disc_std = np.std(total_act_train_err[:, :15])
train_vb_mae = np.mean(total_act_train_err[:, 15:])
train_vb_std = np.std(total_act_train_err[:, 15:])
train_total_mae = np.mean(total_act_train_err)
train_total_std = np.std(total_act_train_err)

val_disc_mae = np.mean(total_act_val_err[:, :15])
val_disc_std = np.std(total_act_val_err[:, :15])
val_vb_mae = np.mean(total_act_val_err[:, 15:])
val_vb_std = np.std(total_act_val_err[:, 15:])
val_total_mae = np.mean(total_act_val_err)
val_total_std = np.std(total_act_val_err)
print('............................................................................')
print('train disc_mae = %.4f, train disc_std = %.4f' % (train_disc_mae, train_disc_std))
print('val disc_mae = %.4f, val disc_std = %.4f' % (val_disc_mae, val_disc_std))
print('............................................................................')
print('train vb_mae = %.4f, train vb_std = %.4f' % (train_vb_mae, train_vb_std))
print('val vb_mae = %.4f, val vb_std = %.4f' % (val_vb_mae, val_vb_std))
print('............................................................................')
print('train total_mae = %.4f, train total_std = %.4f' % (train_total_mae, train_total_std))
print('val total_mae = %.4f, val total_std = %.4f' % (val_total_mae, val_total_std))
print('............................................................................')
best_val_disc = np.min(np.mean(total_act_val_err[:, :15], axis=1))
best_val_vb = np.min(np.mean(total_act_val_err[:, 15:], axis=1))
best_val_total = np.min(np.mean(total_act_val_err, axis=1))
print('best val disc_mae = %.4f' % best_val_disc)
print('best val vb_mae = %.4f' % best_val_vb)
print('best val total_mae = %.4f' % best_val_total)

worst_val_disc = np.max(np.mean(total_act_val_err[:, :15], axis=1))
worst_val_vb = np.max(np.mean(total_act_val_err[:, 15:], axis=1))
worst_val_total = np.max(np.mean(total_act_val_err, axis=1))
print('worst val disc_mae = %.4f' % worst_val_disc)
print('worst val vb_mae = %.4f' % worst_val_vb)
print('worst val total_mae = %.4f' % worst_val_total)