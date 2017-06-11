import scipy.io
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

#mat = scipy.io.loadmat('matlab.mat')
#val = np.array(mat['val'])

mat = scipy.io.loadmat('matlab48.mat')
val = np.array(mat['dataset2_val'])

X = np.asarray(val[:,1:]).astype('float64')
Y = np.asarray(val[:,0]).astype('int64') 

X_tsne = TSNE(learning_rate=1000).fit_transform(X)
X_pca = PCA(n_components=50).fit_transform(X)

plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=Y,cmap=plt.cm.get_cmap("bwr", 10))
plt.subplot(122)
plt.scatter(X_pca[:, 0], X_pca[:, 1],c=Y,cmap=plt.cm.get_cmap("bwr", 10))


from sklearn.decomposition import TruncatedSVD

X_reduced = TruncatedSVD(n_components=2048, random_state=0).fit_transform(X)
X_embedded = TSNE(n_components=13, 
                  perplexity=50, 
                  verbose=2,
                  early_exaggeration=4).fit_transform(X_reduced)

fig = plt.figure(figsize=(10, 10))
ax = plt.axes(frameon=False)
plt.setp(ax, xticks=(), yticks=())
plt.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=0.9,
                wspace=0.0, hspace=0.0)
plt.scatter(X_embedded[:, 0], X_embedded[:, 1],
        c=Y, marker="x",cmap=plt.cm.get_cmap("bwr", 10))