#!coding:utf-8-*-
"""
学習アルゴリズム
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation as anime

def error(y_hat, t):
    return 0.5*np.sum(y_hat - t)**2
### 相関係数rの乱数を生成
r = 0.7
eta = 0.001 #学習係数
epoch = 100 #学習回数
a = np.sort(np.random.randn(1000, 1))
b = np.random.randn(1000, 1)
c = r*a + np.sqrt(1 - r**2) * b
X = np.concatenate([a, c], 1)

### modelの定義
w = np.random.randn(2, 1)
t = a*r
e_array = []
figs = []
fig, ax = plt.subplots()
for i in range(epoch):
    y_hat = np.dot(X, w)
    plt.plot(a, c, "o")
    im = ax.plot(a, y_hat)
    figs.append(im)
    E = error(y_hat, t)
    e_array.append(E)
    grad = np.dot(np.dot(X.T, X), w) - np.dot(X.T, t)
    w -= eta*grad

animation = anime(fig, figs, interval = 1000)
plt.show()

"""
plt.grid()
plt.plot(a, c, "o")
plt.show()
"""
