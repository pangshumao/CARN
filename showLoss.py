import numpy as np
import pylab as plt
import os
plt.switch_backend('agg')

rootDir = 'H:\\ImageData\\spine\\allData\\DenseNet-Gan-allData\\disc_vb_height\\folder5\\loss'
subDir1 = 'GCNN-SDNE_depth_8_dataset_SPINE+_gamma_0.5_lambda_g_0.0_lr_0.04_knn_20_laeweight_101.0_sdneweight_0.005'
subDir2 = 'GCNN-SDNE_depth_8_dataset_SPINE+_gamma_0.5_lambda_g_0.0_lr_0.04_knn_20_laeweight_101.0_sdneweight_0.0'
spmr_data = np.load('/'.join([rootDir, subDir1, 'loss.npz']))
nonSpmr_data = np.load('/'.join([rootDir, subDir2, 'loss.npz']))
spmr_train_loss = spmr_data['train_sdne_loss']
spmr_val_loss = spmr_data['val_sdne_loss']

nonSpmr_train_loss = nonSpmr_data['train_sdne_loss']
nonSpmr_val_loss = nonSpmr_data['val_sdne_loss']

x = np.arange(len(spmr_train_loss))
# plt.figure(figsize=(24/2.54, 16/2.54))
plt.figure()
p1, = plt.plot(x, spmr_train_loss)
p2, = plt.plot(x, nonSpmr_train_loss)
plt.xlabel('epoch')
plt.ylim(0,1)
plt.legend(handles=[p1, p2, ], labels=['proposed', 'non-LSPMR'], loc='best')
plt.show()
plt.savefig(os.sep.join([rootDir, subDir1, 'compared_sdne_loss.eps']), dpi=300)
plt.close()