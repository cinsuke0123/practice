#!coding:utf-8-*-
"""
0,1を分類する問題
"""
import numpy as np
def sigmoid(x):
    return 1 / 1 + np.exp(x)
def softmax(x):
    return np.exp(x)/np.sum(np.exp(x))
def error(y, t):
    return 0.5 * np.sum((y - t)**2)

classes = ["dog", "cat"]
t = np.array([1,0])
X_0 = np.array([0.3, 0.2, 0.1])
W_1 = np.random.randn(3, 3).T #初期値は何でも良いが、正規乱数が収束が早いらしい
u_1 = np.dot(X_0, W_1)

X_1 = sigmoid(u_1)
W_2 = np.random.rand(2,3).T #2行3列の転置行列
u_2 = np.dot(X_1, W_2)

y = list(softmax(u_2))
result = y.index( max(y) )
e = error(y, t)

print("result_animal: {}\nsoftmax: {}\nerror: {}".format(classes[result], y, e))
