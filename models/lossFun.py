import tensorflow as tf
import numpy as np

def mse_lossFun(pred_y, true_y):
    # return tf.reduce_mean(tf.square(pred_y - true_y))
    return tf.reduce_mean(tf.reduce_sum(tf.square(pred_y - true_y), axis=1))

def l1_lossFun(pred_y, true_y):
    return tf.reduce_mean(tf.reduce_sum(tf.abs(pred_y - true_y), axis=1))

def mae_lossFun(pred_y, true_y):
    return tf.reduce_mean(tf.abs((pred_y - true_y)))

def adaptive_lossFun(pred_y, true_y, recon_y, sigma, laeWeight):
    euDist2 = tf.reduce_sum(tf.square(recon_y - true_y), axis=1)
    lambda_1 = tf.exp(-euDist2/(2*np.square(sigma)))
    lambda_2 = 1 - lambda_1
    err_1 = tf.reduce_mean(tf.abs((pred_y - true_y)), axis=1)
    err_2 = tf.reduce_mean(tf.abs((pred_y - recon_y)), axis=1)
    # return tf.reduce_mean(tf.multiply(err_1, lambda_1) + laeWeight * tf.multiply(err_2, lambda_2))
    return tf.reduce_mean(err_1 + laeWeight * tf.multiply(err_2, lambda_2))

def lspmr_lossFun(fea, adjacentMatrix):
    shape = tf.shape(fea)
    batchSize = shape[0]
    batchSize = tf.cast(batchSize, tf.float32)

    # count = tf.reduce_sum(tf.cast(tf.greater(adjacentMatrix,0.0), tf.float32))
    count = tf.clip_by_value(tf.cast(tf.count_nonzero(adjacentMatrix), tf.float32),1.0,100000000.0)

    D = tf.diag(tf.reduce_sum(adjacentMatrix, 1))
    L = D - adjacentMatrix  ## L is laplation-matriX
    loss = 2 * tf.trace(tf.matmul(tf.matmul(tf.transpose(fea), L), fea))
    loss = tf.divide(loss,count)
    return loss

def robust_lossFun(pred_y, true_y, alpha, c):
    """
    reference: A More General Robust Loss Function
    :param pred_y:
    :param true_y:
    :param alpha:
    :param c:
    :return:
    """
    err = pred_y - true_y
    if alpha == 0:
        out_matrix =  np.log(0.5 * np.power(err/c, 2) + 1)
    elif alpha == '-inf':
        out_matrix = (1 - np.exp(-0.5 * np.power(err/c, 2)))
    else:
        z = max(1, 2-alpha)
        out_matrix = z/alpha * (np.power((1 + np.power(err/c,2)/z), alpha/2) - 1)
    return tf.reduce_mean(out_matrix)

def pearson_lossFun(pred_y, true_y):
    eps = 1e-8
    axis = 0
    EX = tf.reduce_mean(pred_y, axis=axis)
    EY = tf.reduce_mean(true_y, axis=axis)
    EXY = tf.reduce_mean(tf.multiply(pred_y, true_y), axis=axis)
    top = tf.subtract(EXY, tf.multiply(EX, EY))
    _, var1 = tf.nn.moments(true_y, axes=[axis])
    _, var2 = tf.nn.moments(pred_y, axes=[axis])
    std1 = tf.sqrt(var1)
    std2 = tf.sqrt(var2)
    bot = tf.add(tf.multiply(std1, std2), eps)
    # return (1 - tf.reduce_mean(tf.divide(top, bot)))
    return tf.reduce_mean(tf.divide(top, bot))

def logHC_lossFun(pre_y, true_y):
    err = pre_y - true_y
    return tf.reduce_mean(tf.log((tf.exp(err) + tf.exp(-err))/2))