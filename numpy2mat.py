import numpy as np
import os
import scipy.io as sio

fileDir = 'F:\\ImageData\\spine\\allData\\aligned\\DenseNet-Gan\\disc_vb_height'
data = np.load(os.sep.join([fileDir, 'samples.npz']))
y = data['y']
sio.savemat(os.sep.join([fileDir, 'y.mat']), {'y':y})