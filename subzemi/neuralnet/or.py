#!-*-coding:utf-8-*
import numpy as np
#!-*-coding:utf-8-*-
import numpy as np

###########
#x1=1 x2=1#
###########
X_1 = np.array([1,1])
W_1 = np.array([0.5,0.5])

u_1 = np.dot( X_1 , W_1 ) #内積をとる

print( "x1=1 x2=1", u_1 , "1" )

###########
#x1=0 x2=1#
###########
X_2 = np.array([0,1])
W_2 = np.array([0.5,0.5])

u_2 = np.dot( X_2 , W_2 ) #内積をとる

print( "x1=0 x2=1",u_2 , "0" )

###########
#x1=1 x2=0#
###########
X_3 = np.array([1,0])
W_3 = np.array([0.5,0.5])

u_3 = np.dot( X_3 , W_3 ) #内積をとる

print( "x1=1 x2=0",u_3 , "0")

###########
#x1=0 x2=0#
###########
X_4 = np.array([0,0])
W_4 = np.array([0.5,0.5])

u_4 = np.dot( X_4 , W_4 ) #内積をとる

###########
##線形分離##
###########

# 単位ステップ関数: ある敷地を越えると1, 超えないと0
def step( x , theta=0.4 ): #任意のパラメータで1or0に分ける
  return np.where( x >= theta , 1 , 0 )

def OR( x_1, x_2):
    X = np.array([x_1, x_2])
    W = np.array([0.5, 0.5])
    w = np.dot(X, W)
    return w

def test( GATE ):
    for i in [0,1]:
        for j in [0,1]:
            print("x_1 = {}, x_2 = {}: y = {}".format(i, j, GATE(i,j) ) )　  
            print("x_1 = {}, x_2 = {}: y = {}".format(i, j, step(x = GATE(i,j))))

def and_gate( x_1 , x_2 ):
  X = np.array([x_1,x_2])
  W = np.array([0.5,0.5])
  return step( x=np.dot( X , W ) , theta=0.7 ) #np.dotで内積を取る

print("step function")
print( "x1=0 x2=0",u_4 , "0", step(x = u_4, theta = 0.7))
print("test function")
test(GATE = OR)
print("and_gate")
print( and_gate( 0,0 ) )
print( and_gate( 1,0 ) )
print( and_gate( 0,1 ) )
print( and_gate( 1,1 ) )
