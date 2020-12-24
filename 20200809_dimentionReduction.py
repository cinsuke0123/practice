"""
20200809
次元圧縮
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D #3次元可視化

### データ読み込み
df = pd.read_csv("/Users/tsuchiyayoshimi/Desktop/practice/titanic/train.csv")

### 加工
df["Age"] = df["Age"].fillna(df["Age"].mean())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

X = df.drop(columns=["PassengerId", "Survived", "Name", "Ticket", "Cabin"])
Y = df["Survived"]
# カテゴリカル変数を数値に
cat_features = ["Sex", "Embarked"]
for col in cat_features:
    lbl = LabelEncoder()
    X[col] = lbl.fit_transform(list(X[col].values))
# 大きなスケールの値をとる変数を標準化
num_features = ["Age", "Fare"]
for col in num_features:
    scaler = StandardScaler()
    X[col] = scaler.fit_transform(np.array(df[col].values).reshape(-1, 1))
### PCA
pca = PCA()
X_pca = pca.fit_transform(X)
print(X_pca)
print(X_pca.shape)
### 可視化
def plot_2d(X, Y):
    plt.plot(X[:, 0][Y == 0], X[:,1][Y == 0], "bo", ms=15)
    plt.plot(X[:, 0][Y == 1], X[:, 1][Y == 1], "r^", ms=15)
    plt.xlabel("First Principal Component")
    plt.ylabel("Second Principal Component")
    plt.legend(["Not Survived", "Survived"], loc="best")
def plot_3d(X, Y):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(X[:, 0][Y == 0], X[:, 1][Y == 0],  X[:, 2][Y == 0], "bo", ms=15)
    ax.plot(X[:, 0][Y == 1], X[:, 1][Y == 1],  X[:, 2][Y == 1], "r^", ms=15)
    ax.set_xlabel("First Principal Component", fontsize=15)
    ax.set_ylabel("Second Principal Component", fontsize=15)
    ax.set_zlabel("Third Principal Component", fontsize=15)
    ax.legend(["Not Survived", "Survived"], loc="best", fontsize=15)

plt.figure(figsize=(10,10))
plot_2d(X_pca, Y)
plt.show()
plot_3d(X_pca, Y)
plt.show()

### 寄与率
print("累積寄与率: {}".format(pca.explained_variance_ratio_))
# 各主成分の寄与率
plt.figure(figsize=(12,8))
plt.plot(pca.explained_variance_ratio_)
plt.xlabel("n_components")
plt.ylabel("explained variance ratio")
plt.show()
# 累積寄与率
plt.figure(figsize=(12,8))
plt.plot(np.hstack([0, pca.explained_variance_ratio_.cumsum()]))#y軸の値を0スタートに
plt.xlabel("n_components")
plt.ylabel("explained variance ratio")
plt.show()
# 変数ごとの
plt.matshow(pca.components_, cmap="Greys")
plt.yticks(range(len(pca.components_)), range(1, len(pca.components_) + 1)) #第一引数が目盛の個数、第２引数がそれぞれの目盛の値
plt.colorbar()
plt.xticks(range(X.shape[1]), X.columns.values, rotation=60, ha="left")
plt.xlabel("Features")
plt.ylabel("Principal Components")
plt.show()
print("shape関数（X.shape）\nXの1次元目: {}\n Xの2次元目: {}".format(X.shape[0], X.shape[1]))