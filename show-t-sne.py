# encoding=utf-8
# -*- coding:utf-8 -*

# 切换工作路径
import os
import sys
os.chdir( os.path.split( os.path.realpath( sys.argv[0] ) )[0] )

import numpy
from numpy import *
import numpy as np

from sklearn.manifold import TSNE
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

class chj_data(object):
    def __init__(self,data,target):
        self.data=data
        self.target=target

def chj_load_file(fdata,ftarget):
    data=numpy.loadtxt(fdata, dtype=float32)
    target=numpy.loadtxt(ftarget, dtype=int32)

    print(data.shape)
    print(target.shape)
    # pexit()

    res=chj_data(data,target)
    return res

fdata="fdata.txt"
ftarget="ftarget.txt"

# iris = load_iris() # 使用sklearn自带的测试文件
iris = chj_load_file(fdata,ftarget)

#print(iris.data)
#print(iris.target)
#exit()
X_tsne = TSNE(n_components=2,learning_rate=200,perplexity=5,n_iter=5000).fit_transform(iris.data)
#X_pca = PCA().fit_transform(iris.data)
print("finishe!")
plt.figure(figsize=(12, 6))
#plt.subplot(121)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=iris.target)
#plt.subplot(122)
#plt.scatter(X_pca[:, 0], X_pca[:, 1], c=iris.target)
plt.colorbar()
plt.show()