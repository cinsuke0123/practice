#!coding:utf-8-*-
"""
誤差関数
"""
import numpy as np

y_hat = np.array([1, 1])
t = np.array([0.5, 0.5])

### 二乗和誤差の実装
e = 0
for i in range( len(y_hat) ):
    e += (y_hat[i] - t[i])**2

e = 0.5 * e
print(e)

### 二乗和誤差の実装(numpyを用いた場合)
e_np = 0.5 * np.sum( (y_hat - t)**2 )
print( e_np )
