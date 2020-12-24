"""
20200809
モデルの改良と前処理
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

### データ読み込み
df = pd.read_csv("/Users/tsuchiyayoshimi/Desktop/practice/titanic/train.csv")

### データ加工
# 欠損値処理
df["Age"] = df["Age"].fillna(df["Age"].mean())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
# XとYに分ける
X = df.drop(columns=["PassengerId", "Survived", "Name", "Ticket", "Cabin"])
Y = df["Survived"]
# カテゴリカル変数を数値に
cat_features = ["Sex", "Embarked"]
for col in cat_features:
    lbl = LabelEncoder()
    X[col] = lbl.fit_transform(list(X[col].values))
# カテゴリかる変数をダミー変数に
print("Pclassのvalue counts: \n{}".format(X["Pclass"].value_counts())) #\nは改行
print("SibSpのvalue counts: \n{}".format(X["SibSp"].value_counts()))
X = pd.get_dummies(X, columns=["Pclass", "SibSp", "Embarked"], drop_first=True) #drop_firstでリファレンスを覗く
# 大きなスケールの値をとる変数を標準化
num_features = ["Age", "Fare"]
for col in num_features:
    scaler = StandardScaler()
    X[col] = scaler.fit_transform(np.array(df[col].values).reshape(-1, 1))
##＃ 分析
# 学習器
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0)
svm = SVC(C=50).fit(X_train, Y_train) #Cはソフトマージンのペナルティ項
# 正解率
print("svm score: {}".format(svm.score(X_test, Y_test)))
# Cを操作してベストなパラメータの大きさを探す
# 本当はこのやり方は好ましくない
# test dataを見ながらパラメータをいじるのはだめ。test setは全く未知の状態にしておく
best_score = 0
for C in [1, 10, 100, 1000, 10000]:
    svm = SVC(C=C).fit(X_train, Y_train)
    score = svm.score(X_test, Y_test)
    if score > best_score:
        best_score = score
        best_parameter = C
print("best_parameter: {}".format(best_parameter))
print("best_score: {}".format(best_score))
# 上記問題のために、validation dataを作成してグリットサーチ
X_train, X_validation, Y_train, Y_validation = train_test_split(X_train, Y_train, random_state=0)
best_score = 0
for C in [1, 10, 100, 1000, 10000]:
    svm = SVC(C=C).fit(X_train, Y_train)
    score = svm.score(X_validation, Y_validation)
    if score > best_score:
        best_score = score
        best_parameter = C
svm_best = SVC(C=best_parameter).fit(X_train, Y_train)
svm_best_score = svm.score(X_test, Y_test)
print("validation dataを用いた場合\nbest_parameter: {}\nbest_score:{}".format(best_parameter,svm_best_score))
# 上記のグリッドサーチを実装したGridSearchCVを用いた検証
param = {
    "C": [0.01, 0.1, 1, 10, 100]
}
grid_search = GridSearchCV(SVC(), param_grid=param, cv=5) #CVは訓練データと検証データの黄砂検証
grid_search.fit(X_train, Y_train)
print("grid search")
print("best parameter :{}".format(grid_search.best_params_))
print("正解率: {}".format(grid_search.score(X_test, Y_test))) #best parameterでの学習器を生成してくれているので、そこにtest dataを当てはめる
# 訓練+検証データとテストデータ、訓練データと検証データを二重にクロスバリデーション
scores = cross_val_score(GridSearchCV(SVC(), param_grid=param, cv=5), X, Y, cv=6)
print("grid search with cross validation")
print(scores) #cv=6なので6分割している
print(scores.mean())
