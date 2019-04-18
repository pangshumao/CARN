import numpy as np

def call_evaluate(dataDir,y_mode,model_identifier):
    folderNum = 5
    data = np.load('/'.join([dataDir, y_mode, 'results', model_identifier, 'predict-fold1.npz']))
    outDim = data['train_mae'].shape[-1]

    total_train_mae = np.zeros((folderNum, outDim))
    total_train_mse = np.zeros((folderNum, outDim))
    total_train_pearson = np.zeros((folderNum, outDim))
    total_val_mae = np.zeros((folderNum, outDim))
    total_val_mse = np.zeros((folderNum, outDim))
    total_val_pearson = np.zeros((folderNum, outDim))
    for folderInd in range(0,folderNum):
        data = np.load('/'.join([dataDir, y_mode, 'results', model_identifier, 'predict-fold' + str(folderInd + 1) + '.npz']))
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
    return train_mae, train_mse, train_pearson, val_mae, val_mse, val_pearson