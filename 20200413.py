"""
svmの実装
"""

import mglearn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn import svm

iris = load_iris()

x = iris.data[50:,2:]
y = iris.target[50:]-1

mglearn.discrete_scatter(x[:,0],x[:,1],y) #yで●と▲を分類
plt.legend(["versicolor","virsinica"], loc = "best") #locは場所
plt.show()

#stratify=yはyを同じ割合で分割する。層化サンプリング
#random_stateは乱数。今回は０で固定
x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, random_state=0)

svm = svm.SVC()
svm = svm.fit(x_train, y_train)
def plot_searater(model):
    plt.figure(figsize=(10, 6))
    mglearn.plots.plot_2d_separator(model, x)
    mglearn.discrete_scatter(x[:,0], x[:,1], y)
    plt.xlabel("petal_length")
    plt.ylabel("petal_width")
    plt.xlim(2.8, 7.0)
    plt.ylim(0.8, 2.6)
    plt.show()


plot_searater(svm)
