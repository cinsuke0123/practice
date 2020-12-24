#!-*-coding!utif-8-*-
import numpy as np
import matplotlib.pyplot as plt

#ハイパボリック関数
def tanh( x ):
    sinh = ( np.exp( x ) - np.exp( -x ) ) / 2
    cosh = (np.exp( x ) + np.exp( -x ) ) / 2

    return sinh / cosh

# tanhを2乗するとシグモイド関数
def tan_prime( x ):
    return 1 - tanh( x )**2

x = np.arange( -5, 5, 0.1 )
plt.grid()
plt.plot( x, tanh( x ) )
plt.plot( x, tan_prime( x ) )
plt.savefig( "./graph/tanh.png" )
