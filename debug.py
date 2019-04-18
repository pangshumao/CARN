import numpy as np
import os
from matplotlib import pyplot as plt

outDir = 'H:\\ImageData\\spine\\allData\\DenseNet-Gan-allData\\disc_vb_height\\results\\GCNN-SDNE_depth_8_dataset_SPINE+_gamma_0.5_lambda_g_0.0_lr_0.04_knn_20_laeweight_101.0_sdneweight_0.005'
data = np.load('/'.join([outDir, 'evaluation.npz']))
val_mae = data['val_mae']
pass

