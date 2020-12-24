#!coding:utf-8-*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import sys

mail_df = pd.read_csv("./datas/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv")

### 女性向けメールが配信されたデータを削除したデータを作成
male_df = mail_df[mail_df.segment != 'Womens E-Mail'].copy()  # 女性向けメールが配信されたデータを削除
male_df["treatment"] = np.where(male_df.segment =="Mens E-Mail",1,0)

# バイアスのあるデータの作成(ここが本では明示的に書かれていなかった。1章のバイアスの作り方を適用)
sample_rules = (male_df.history > 300) | (male_df.recency < 6) | (male_df.channel == 'Multichannel')

biased_df = pd.concat([
    male_df[(sample_rules) & (male_df.treatment == 0)].sample(frac=0.5, random_state=1), #fracで割合指定
    male_df[(sample_rules) & (male_df.treatment == 1)],
    male_df[(~sample_rules) & (male_df.treatment == 0)],
    male_df[(~sample_rules) & (male_df.treatment == 1)].sample(frac=0.5, random_state=1)
], axis=0, ignore_index=True) # axis = 0で縦に結合, ingnore_index = TRUEで結合後にindexを振り直す


#介入変数(treatment)とvisitの相関を確認
Y = biased_df[['treatment']]
X = pd.get_dummies(biased_df[['visit','recency','channel','history']],columns=['channel'],drop_first=True)
X = sm.add_constant(X)
results = sm.OLS(Y,X).fit()
table = results.summary().tables[1] #tables[1]だと回帰表だけ表示
#print(table)

#visitを含めた重回帰分析
Y = biased_df[['spend']]
X = pd.get_dummies(biased_df[['treatment','visit','recency','channel','history']],columns=['channel'],drop_first=True)
X = sm.add_constant(X)
results = sm.OLS(Y,X).fit()
table = results.summary()
print(table)
sys.exit()
#pd.get_dummiesを使わずにダミー変数を作るパターン
biased_df['channelPhone'] = np.where(biased_df.channel == "Phone",1,0)
biased_df['channnelWeb'] = np.where(biased_df.channel == 'Web',1,0)

Y = biased_df[['spend']]
X = biased_df[['treatment','visit','recency','channelPhone','channnelWeb','history']]
X = sm.add_constant(X)
results = sm.OLS(Y,X).fit()
table = results.summary().tables[1]
print(table)
