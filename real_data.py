#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 03:01:57 2017

@author: hayatiibis
"""
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD

mat = scipy.io.loadmat('matlab35.mat')
val = np.array(mat['val'])

X = np.asarray(val[:,1:]).astype('float64')
Y = np.asarray(val[:,0]).astype('int64') 

X_reduced = TruncatedSVD(n_components=50, random_state=0).fit_transform(X)
X_embedded = TSNE(n_components=2, perplexity=50, verbose=2).fit_transform(X_reduced)

fig = plt.figure(figsize=(8, 8))
ax = plt.axes(frameon=False)
plt.setp(ax, xticks=(), yticks=())
plt.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=0.9,
                wspace=0.0, hspace=0.0)
plt.scatter(X_embedded[:, 0], X_embedded[:, 1],
        c=Y, marker="x",cmap=plt.cm.get_cmap("bwr", 10))
