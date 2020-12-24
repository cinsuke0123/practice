#!-*-coding:utf-8-*
import sys #局所的に値を確認したいときに処理を中断してくれる
import pandas as pd
import numpy as np #行列演算

"""
仮装データを用いてRCTの効果を実験するプログラム
本来は観測ができないy_0やy_1が擬似的に作られており、ATEは100に設定されている
"""

"""
1. 購買意欲の高いユーザに絞ってクーポンメールを送信した際の平均処置効果を調べる
"""

df = pd.read_csv("data/mail_marketing_biased.csv")
"""
print( df.head())
print( type(df) )
sys.exit()
"""

"""
ATEの算出
"""
y_1 = np.sum( df.y * df.z * df ) * ( 1 / np.sum( df.z == 1 ) ) #y=1
y_0 = np.sum( df.y * ( 1 - df.z ) ) * (1 / np.sum( 1 - df.z == 1 ) ) #y=0
print( "平均処置効果:", y_1 - y_0 )

"""
biasの算出
"""
y_0_z_0 = np.sum( df.y_0 * ( 1 - df.z ) ) * ( 1 / np.sum( df.z == 0 ) )
y_0_z_1 = np.sum( df.y_0 * df.z ) * ( 1 / np.sum( df.z == 1 ) )

print( "bias: ", y_0_z_0 - y_0_z_1)
