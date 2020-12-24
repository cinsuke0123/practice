#!-*-coding:utf-8-*
import sys #局所的に値を確認したいときに処理を中断してくれる
import pandas as pd
import numpy as np #行列演算

def ate( df, y, z ):
    """
    平均処置効果をreturnする
    """
    y_1 = np.sum( df.y * df.z * df ) * ( 1 / np.sum( df.z == 1 ) ) #y=1
    y_0 = np.sum( df.y * ( 1 - df.z ) ) * (1 / np.sum( 1 - df.z == 1 ) ) #y=0

    return round( y_1 - y_0, 2)

df = pd.read_csv("data/mail_marketing_biased.csv")
ATE = ate( df = df, y = y , z = z )
print( ATE )
