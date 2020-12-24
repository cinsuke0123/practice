import sys
import numpy as np
import matplotlib.pyplot as plt

def activate(u):
    # シグモイド関数
    return 1 / (1 + np.exp(-u))

def activate_prime(u):
    # 活性化関数の導関数
    return activate(u)*(1 - activate(u))

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

def E(y, t):
    # 二乗和誤差
    return 1 /2 * np.sum((y -t ) ** 2)

def E_prime(y, t):
    return y - t

def forward(X1):
    global W2
    global W3
    z1 = X1.dot(W2)
    X2 = activate(u=z1)
    z2 = X2.dot(W3)
    X3 = softmax(x = z2)

    return X3, z2

def backward(y, t, z, x):
    global W2
    global W3
    dout = E_prime(y=y, t=t)
    grad_W3 = np.dot(z.T, dout) # W3の勾配
    d_hidden = np.dot(dout, W3.T) * activate_prime( u = z )
    grad_W2 = np.dot(x.T, d_hidden) # W2の勾配

    W2 -= 0.001 * grad_W2
    W3 -= 0.001 * grad_W3

def learn(x, t, epochs = 100): # epochは何回試行するか
    earray = []
    for i in range(epochs):
        y, z = forward(X1 = x)
        e = E(y=y, t=t)
        backward(y=y, t=t,z=z, x=x)
        earray.append(e)
    plt.xlabel("epochs")
    plt.ylabel("error")
    plt.grid()
    plt.plot(np.arange(0, epochs, 1), earray)
    plt.show()



# One hot
T = np.array([
[1,0],
])
X1 = np.array([
[0.3, 0.2],
])
W2 = np.array([
[0.1, 0.3],
[0.2, 0.4]
])
W3 = np.array([
[0.1, 0.3],
[0.2, 0.4]
])

y, z = forward(X1 = X1)
print(backward(y = y, t = T, z = z, x = X1))
learn(x=X1, t=T, epochs = 10000)
