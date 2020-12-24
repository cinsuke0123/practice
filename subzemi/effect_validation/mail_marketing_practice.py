#!-*-coding:utf-8-*-
import warnings
import numpy as np
import pandas as pd
from scipy import stats as st

### データ読み込み
warnings.simplefilter("ignore")
df = pd.read_csv("datas/hoge.csv")

male_df = df[df.segment != "Womens E-Mail"]
male_df["treatment"] = np.where( male_df.segment == "Mens E-Mail", 1, 0 )
print(male_df.head())

### グループ集計
summary_by_segment = male_df.groupby("treatment").mean()
print(summary_by_segment)

mens_email = np.array( male_df[ male_df.treatment == 1].spend)
no_mail = np.array( male_df[ male_df.treatment == 0].spend)

rct_ttest = st.ttest_ind( mens_email, no_mail, equal_var = True)
print(rct_ttest)
