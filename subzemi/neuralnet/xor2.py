#!-*-coding:utf-8-*
import numpy as np

def step( x, theta = 0):
    # ステップ関数
    return np.where(x >= theta, 1, 0)

def XOR( x_1, x_2 ):
    # XOR GATE
    X_1 = np.array([x_1, x_2])
    W_1 = np.array([[-0.2, -0.2], [0.1, 0.1]]) #-0.2のベクトルがNAND関数もう一方がOR関数
    b_1 = np.array([0.2, -0.1]) #定数項(バイアス項)
    u_1 = np.dot( W_1, X_1 ) + b_1 #内積とる

    X_2 = step( x=u_1, theta = 0 )
    W_2 = np.array([0.5, 0.5])
    b_2 = np.array([0,0])
    u_2 = np.dot(W_2, X_2) + b_2

    y = step( x =u_2, theta = 0.6 )

    return y

def test( GATE ):
    for i in [0,1]:
        for j in [0,1]:
            print( "x_1={}, x_2={} : y={}".format( i , j, GATE( x_1 = i, x_2 = j ) ) )
            print( "" )

test( GATE = XOR )
