#!-*-coding:utf-8-*-

import numpy as np
import matplotlib.pyplot as plt

def softmax( x ):
    return np.exp( x ) / np.sum( np.exp( x ) )

x = np.array( [ 5, 1, 3 ] )
y = softmax( x )
print( np.sum( y ) )
