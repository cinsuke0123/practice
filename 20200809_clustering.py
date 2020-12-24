"""
20200809
クラスター分析（階層的クラスタリング、k-meansクラスタリング）
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs #クラスタリング用のデータセット
from sklearn.datasets import make_moons
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage

def plot_dataset(X, Y):
    plt.plot(X[:, 0][Y == 0], X[:, 1][Y == 0], "bo", ms=15)
    plt.plot(X[:, 0][Y == 1], X[:, 1][Y == 1], "r^", ms=15)
    plt.plot(X[:, 0][Y == 2], X[:, 1][Y == 2], "gs", ms=15)
    plt.xlabel("$X_0$", fontsize=15)
    plt.ylabel("$X_1$", fontsize=15)

### make_blobsデータ
X, Y = make_blobs(n_samples=200, n_features=2, centers=3, random_state=10)

plt.figure(figsize=(10, 10))
plot_dataset(X, Y)
plt.show()

k_means =KMeans(n_clusters=3).fit(X)
print("k_means labels（クラスタリングしたラベル）: \n{}".format(k_means.labels_))
plot_dataset(X, k_means.labels_)
plt.show()

### make_moonsデータ
X, Y = make_moons(n_samples=200, noise=0.1, random_state=0)
plt.figure(figsize=(12, 8))
plot_dataset(X, Y)
plt.show()

k_means = KMeans(n_clusters=2).fit(X)
print("k_means labels（クラスタリングしたラベル）: \n{}".format(k_means.labels_))
plot_dataset(X, k_means.labels_)
plt.show()

### デンドログラム
iris = load_iris()
X = iris.data
Z = linkage(X, method="average", metric="euclidean") #群平均法, euclid距離（ユークリッド）
plt.figure(figsize=(18, 12))
dendrogram(Z)
plt.show()