"""
20200807
アンサンブル学習前編（ランダムフォレスト）
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_moons
from sklearn.datasets import load_iris
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


### データ作成
moons = make_moons(n_samples=200, noise=0.2, random_state=0)
X = moons[0]
Y = moons[1]

def plot_decision_boundary(model, X, Y, margin = 0.3):
    _x1 = np.linspace(X[:, 0].min() - margin, X[:, 0].max() + margin, 100) #linspaceは、第一引数start, 第２引数stop, 第３引数要素数で配列作成
    _x2 = np.linspace(X[:, 1].min() - margin, X[:, 1].max() + margin, 100)
    x1, x2 = np.meshgrid(_x1, _x2) #meshgridは各座標の要素列から格子座標を作成
    X_new = np.c_[x1.ravel(), x2.ravel()] #ravelは一次元化する
    Y_pred = model.predict(X_new).reshape(x1.shape) #reshapeで等高線をx1にする
    custom_cmap = ListedColormap(["mediumblue", "orangered"])
    plt.contourf(x1, x2, Y_pred, alpha=0.3, cmap=custom_cmap) #countourfは等高線（塗りつぶしあり）

def plot_datasets(X, Y):
    plt.plot(X[:, 0][Y == 0], X[:, 1][Y == 0], "bo", ms=15)
    plt.plot(X[:, 0][Y == 1], X[:, 1][Y == 1], "r^", ms=15)
    plt.xlabel("$x_0$", fontsize=30)
    plt.xlabel("$x_1$", fontsize=30, rotation=0)

plt.figure(figsize=(12,8))
plot_datasets(X, Y)
plt.show()

### 加工
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0)
tree_clf = DecisionTreeClassifier().fit(X_train, Y_train)

plt.figure(figsize=(12,8))
plot_decision_boundary(tree_clf, X, Y)
plot_datasets(X, Y)
plt.show()

### ランダムフォレスト
random_forest = RandomForestClassifier(n_estimators = 100, random_state=0).fit(X_train, Y_train)

plt.figure(figsize=(12, 8))
plot_decision_boundary(random_forest, X, Y)
plot_datasets(X, Y)
plt.show()

### irisデータ
iris = load_iris()
X_iris = iris.data
Y_iris = iris.target
random_forest_iris = RandomForestClassifier(n_estimators=100, random_state=0).fit(X_iris, Y_iris)
print(iris.feature_names) #特徴量
print(random_forest_iris.feature_importances_) #これらの特徴量がどれくらい重要度を持っているのか
plt.figure(figsize=(12, 8))
plt.barh(range(iris.data.shape[1]), random_forest_iris.feature_importances_, height=0.1) #shapeは配列の要素数（何次元目に何個）、配列の変更のインスタンス
plt.yticks(range(iris.data.shape[1]), iris.feature_names, fontsize=20) #y軸の凡例
plt.xlabel("Feature Importances", fontsize=30)
plt.show()

### tatanicデータを分析
# https://www.kaggle.com/c/titanic/data
df = pd.read_csv("/Users/tsuchiyayoshimi/Desktop/practice/titanic/train.csv")
print(df.head())
print(df.info())
### 欠損値を抜く（キャビン以外、ageとembarked）
df["Age"] = df["Age"].fillna(df["Age"].mean()) #平均値をnaに入れる
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0]) #最頻値
print(df.info())
### Labelencoder
cat_features = ["Sex", "Embarked"]
for col in cat_features:
    lbl = LabelEncoder()
    df[col] = lbl.fit_transform(list(df[col].values))
print(df.head())

### random forestの実装
X = df.drop(columns=["PassengerId", "Survived", "Name", "Ticket", "Cabin"])
Y = df["Survived"]
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, random_state=0)
tree = DecisionTreeClassifier().fit(X_train, Y_train)
print("tree.score: {}".format(tree.score(X_test, Y_test)))

rnd_forest = RandomForestClassifier(n_estimators=500, max_depth=5, random_state=0).fit(X_train, Y_train)
print("rnd_forest.score: {}".format(rnd_forest.score(X_test, Y_test)))

### testデータで確認
test_df = pd.read_csv("/Users/tsuchiyayoshimi/Desktop/practice/titanic/test.csv")
print("test_df")
print(test_df.head())
### 欠損値を抜く（キャビン以外、ageとembarked）
test_df["Age"] = test_df["Age"].fillna(test_df["Age"].mean()) #平均値をnaに入れる
test_df["Embarked"] = test_df["Embarked"].fillna(test_df["Embarked"].mode()[0]) #最頻値
test_df["Fare"] = test_df["Fare"].fillna(test_df["Fare"].mean()) #平均値をnaに入れる
#Sex, Embarkedがカテゴリカルなstringの変数になっているので離散値に変更
cat_features = ["Sex", "Embarked"]
for col in cat_features:
    lbl = LabelEncoder()
    test_df[col] = lbl.fit_transform(list(test_df[col].values))
# 必要な変数を取ってくる
X_pred = test_df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])
print("X_pred")
print(X_pred.info())
ID = test_df["PassengerId"]

prediction = rnd_forest.predict(X_pred)
### 提出用データ
submission = pd.DataFrame({
    "PassengerId": ID,
    "Survived": prediction
})
submission.to_csv("./titanic/submission.csv", index=False) #index=Falseは最左列の列番号をなくす
