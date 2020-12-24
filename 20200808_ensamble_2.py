"""
20200808
アンサンブル学習後編
AdaBoost, 勾配ブースティング
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

train_df = pd.read_csv("/Users/tsuchiyayoshimi/Desktop/practice/titanic/train.csv")
test_df = pd.read_csv("/Users/tsuchiyayoshimi/Desktop/practice/titanic/test.csv")
print(test_df.head())

### 加工・前処理
# 前処理のためにtraindataとtestdataを結合
# 本当は欠損値処理でtestdataも考慮したmeanがtraindataに含まれるため望ましくない
# testdataはtraindataに対して全く未知の状態にしておくのが基本
all_df = pd.concat((train_df.loc[:, "Pclass":"Embarked"], test_df.loc[:, "Pclass":"Embarked"]))
print(all_df.head())
# 加工
all_df["Age"] = all_df["Age"].fillna(all_df["Age"].mean())
all_df["Fare"] = all_df["Fare"].fillna(all_df["Fare"].mean())
all_df["Embarked"] = all_df["Embarked"].fillna(all_df["Embarked"].mode()[0])
cat_features = ["Sex", "Embarked"]
for col in cat_features:
    lbl = LabelEncoder()
    all_df[col] = lbl.fit_transform(list(all_df[col].values))
all_df = all_df.drop(columns=["Name", "Ticket", "Cabin"])
# 説明変数をtrainとtestに分割
train = all_df[:train_df.shape[0]] #train_dfのデータ数を取ってくる, shapeのindex0は行数を表す
test = all_df[:train_df.shape[0]]

Y = train_df["Survived"]
ID = test_df["PassengerId"]
X_train, X_test, Y_train, Y_test = train_test_split(train, Y, random_state=0)

prameters = {
    "objective": "binary:logistic", #どのような分析を行ったか.今回binary
    "eval_metric": "auc", #正解率
    "eta": 0.1, #学習率
    "max_depth": 6,
    "subsample": 1,
    "colsample_bytree": 1,
    "silent": 1
}

dtrain = xgb.DMatrix(X_train, label = Y_train)
dtest = xgb.DMatrix(X_test, label=Y_test)
model = xgb.train(params=parameters,
                  dtrain=dtrain,
                  num_boost_round=100,
                  early_stopping_rounds=10,
                  evals=[(dtest, "test")]
                  )
prediction = model.predict(xgb.DMatrix(test), ntree_limit=model.best_ntree_limit)
submission = pd.DataFrame({
    "PassengerId" : ID,
    "Survived": prediction
})
submission.to_csv("./titanic/submission_2.csv", index=False)