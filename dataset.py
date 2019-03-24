# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 02:25:58 2019

@author: Ju
"""

import numpy as np
import pandas as pd  

from matplotlib import pyplot as plt


v=np.random.randint(2, size=(3000, 50))


from sklearn.decomposition import PCA
pca = PCA(n_components=4)
pca.fit(v)
X = pca.transform(v)

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters = 20)

#Compute cluster centers and predict cluster indices
X_clustered = kmeans.fit_predict(v)

# Plot the scatter digram
plt.figure(figsize = (7,7))
plt.scatter(X[:,0],X[:,2], c=X_clustered , alpha=0.5,cmap='viridis') 
plt.show()

df=[v[:,2m] != '0']