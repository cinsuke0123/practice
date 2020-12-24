#!coding:utf-8-*-
import sys # sys.exit()でそこまでで処理をやめれる
import numpy as np
import pandas as pd
import statsmodels.api as sm # scipy == 1.1.0
# pipenv, pyenvで特定のファイルだけの環境を個々に定義して、それぞれのファイルでバージョン管理する
# 「このファイルではscipy==1.1.0だけど、あのファイルではscypy==1.2.0でないといけない」みたいな状況で
# グローバル環境を変えずに個々のファイルで環境を変更できるように


###################
#  データの読み込み  #
###################
df = pd.read_csv("./datas/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv")

###################
#  preprocessing  #
###################
male_df = df[df.segment != "Womens E-Mail"]
print( np.unique( male_df["segment"] ) ) #np.unique: 大文字小文字の確認などできる
print( np.unique( df["segment"] ) )
print( np.unique( male_df["channel"] ) )
del df # 処理で食うメモリの量を軽くする

male_df["treatment"] = np.where( male_df.segment == "Mens E-Mail", 1, 0)
male_df["channelPhone"] = np.where( male_df.channel == "Phone", 1, 0)
male_df["channelWeb"] = np.where( male_df.channel == "Web", 1, 0)
male_df["channelMultichannel"] = np.where( male_df.channel == "Multichannel", 1, 0)
# sys.exit()
########################
#  choosing variables  #
########################
X_1 = male_df[["treatment", "history"]]
X_1 = sm.add_constant( X_1 ) # 定数項
X_2 = male_df[["treatment", "recency", "history", "channelPhone", "channelWeb", "channelMultichannel"]]
X_2 = sm.add_constant( X_2 )
Y = male_df[["spend"]]

#################
#      OLS      #
#################

model_1 = sm.OLS(Y, X_1)
results_1 = model_1.fit()
print( results_1.summary() )


model_2 = sm.OLS(Y, X_2)
results_2 = model_2.fit()
print( results_2.summary() )
