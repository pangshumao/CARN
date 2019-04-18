import numpy as np

outDir = '/scratch/l/lishuo/psm/data/aligned/DenseNet'
y_mode = 'disc_height'
outDim = 15
folderNum = 5
total_train_mae = np.zeros((folderNum, outDim))
total_train_mse = np.zeros((folderNum, outDim))
total_train_pearson = np.zeros((folderNum, outDim))
total_val_mae = np.zeros((folderNum, outDim))
total_val_mse = np.zeros((folderNum, outDim))
total_val_pearson = np.zeros((folderNum, outDim))
for folderInd in range(0,folderNum):
    data = np.load('/'.join([outDir, y_mode, 'results', 'predict-fold' + str(folderInd + 1) + '.npz']))
    total_train_mae[folderInd,:] = data['train_mae']
    total_train_mse[folderInd, :] = data['train_mse']
    total_train_pearson[folderInd, :] = data['train_pearson']
    total_val_mae[folderInd,:] = data['val_mae']
    total_val_mse[folderInd,:] = data['val_mse']
    total_val_pearson[folderInd,:] = data['val_pearson']

train_pearson = np.mean(total_train_pearson, axis=0)
train_mae = np.mean(total_train_mae, axis=0)
train_mse = np.mean(total_train_mse, axis=0)
val_pearson = np.mean(total_val_pearson, axis=0)
val_mae = np.mean(total_val_mae, axis=0)
val_mse = np.mean(total_val_mse, axis=0)
np.savez('/'.join([outDir, y_mode, 'results', 'evaluation.npz']), val_pearson=val_pearson, val_mae=val_mae,
         val_mse=val_mse, train_pearson=train_pearson, train_mae=train_mae, train_mse=train_mse)
print('{} mean train mae= {}  mean train pearson= {} mean train mse= {}'.format(
    y_mode, str(np.mean(train_mae)), str(np.mean(train_pearson)), str(np.mean(train_mse))))
print('{} mean val mae= {}  mean val pearson= {} mean val mse= {}'.format(
    y_mode, str(np.mean(val_mae)), str(np.mean(val_pearson)), str(np.mean(val_mse))))
