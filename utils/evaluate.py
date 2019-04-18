import numpy as np

def mse(pred_y, true_y):
    return np.mean(np.square((pred_y - true_y)), axis=0)

def mae(pred_y, true_y):
    return np.mean(np.abs((pred_y - true_y)), axis=0)


def pearson(pred_y, true_y):
    eps = 1e-8
    EX = np.mean(pred_y, axis=0)
    EY = np.mean(true_y, axis=0)
    EXY = np.mean(pred_y * true_y, axis=0)
    top = EXY - EX * EY
    std1 = np.std(true_y, axis=0)
    std2 = np.std(pred_y, axis=0)
    bot = std1 * std2 + eps
    return top / bot