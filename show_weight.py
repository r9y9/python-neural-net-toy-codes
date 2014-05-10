#!/usr/bin/python
# coding: utf-8

# 入力層 -> 隠れ層の重みを描画するだけ

import numpy as np
from pylab import *
from sklearn.externals import joblib
 
nn = joblib.load("nn_mnist.pkl")

W = nn.W1
M = int(W.shape[1])
w,h = int(np.sqrt(M)), int(np.sqrt(M))
M = w*h
for i in range(M):
    subplot(w, h, i+1)
    imshow(W[:,i].reshape(28, 28), cmap=cm.Greys_r)
    xticks(())
    yticks(())

show()
