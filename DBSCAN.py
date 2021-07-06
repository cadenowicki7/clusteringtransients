# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 20:18:08 2021

@author: User
"""


# DBSCAN
X = 5
from dbscan import DBSCAN
labels, core_samples_mask = DBSCAN(X, eps=0.3, min_samples=10)

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

# #############################################################################
# Generate sample data
centers = [[1, 1], [-1, -1], [1, -1]]
X, labels_true = make_blobs(n_samples=750, centers=centers, cluster_std=0.4,
                            random_state=0)
X = StandardScaler().fit_transform(X)

# #############################################################################
# Compute DBSCAN
from dbscan import DBSCAN
labels, core_samples_mask = DBSCAN(X, eps=0.3, min_samples=10)

# #############################################################################
# Plot result
import matplotlib.pyplot as plt

n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)
unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]

for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]
    class_member_mask = (labels == k)
    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=14)
    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6)

plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()


######
#@inproceedings{wang2020theoretically,
#  author = {Wang, Yiqiu and Gu, Yan and Shun, Julian},
#  title = {Theoretically-Efficient and Practical Parallel DBSCAN},
#  year = {2020},
#  isbn = {9781450367356},
#  publisher = {Association for Computing Machinery},
#  address = {New York, NY, USA},
#  url = {https://doi.org/10.1145/3318464.3380582},
#  doi = {10.1145/3318464.3380582},
#  booktitle = {Proceedings of the 2020 ACM SIGMOD International Conference on Management of Data},
#  pages = {2555–2571},
#  numpages = {17},
#  keywords = {parallel algorithms, spatial clustering, DBScan},
#  location = {Portland, OR, USA},
#  series = {SIGMOD ’20}
#}