"""
最近傍法
"""

import mglearn
import matplotlib.pyplot as plt
#matplotlib inline：グラフをjupyter内にに描画してくれる
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split as split

###データの読み込み
mglearn.plots.plot_knn_classification(n_neighbors= 3) #k=3でk最近傍法を実施
plt.show()
x, y = mglearn.datasets.make_forge()

###データの型を理解
print(x.shape) #26人分の2種類のデータ
print(y.shape) #26人分の１種類のデータ

###データの可視化
mglearn.discrete_scatter(x[:,0], x[:,1], y)
#第一引数: 横軸に取るデータ; 第二引数: 縦軸に取るデータ; 第三引数: 分類、グルーピング
#plt.show()

#データの加工
x_train, x_test, y_train, y_test = split(x, y, random_state=0)
#random_state: 乱数指定、ランダムにデータ分割する際の乱数の発生仕方を固定できる→何回やっても結果同じになるので学習用に使える
#print(x_train.shape) #(19,2)

###モデリング
##k=3の場合
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(x_train, y_train)

pred = clf.predict(x_test) #予測
score = clf.score(x_test, y_test) #正答率
#print(round(score,3))
print("k=3の場合の正答率: {:.2f}".format(score))

##k=2の場合
clf = KNeighborsClassifier(n_neighbors=15)
clf.fit(x_train, y_train)

pred = clf.predict(x_test) #予測
score = clf.score(x_test, y_test) #正答率
#print(round(score,3))
print("k=15の場合の正答率: {:.2f}".format(score))

#kをforループで複数一気に検証
for n_neighbors in range(1, 16):
    clf = KNeighborsClassifier(n_neighbors = n_neighbors)
    clf.fit(x_train, y_train)
    score = clf.score(x_test, y_test)
    print("k = {} : {:2f}".format(n_neighbors, score))

#１行５列, figsizeは大きさ

fig, axes = plt.subplots(1, 5, figsize = (15, 3))
#figはキャンバス, axesはキャンバスの中の一つ一つのスペースを指すイメージ

for n_neighbors, ax in zip([1,3,5,10,15], axes):
    clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(x_train, y_train)

    mglearn.plots.plot_2d_separator(clf, x, fill= True, ax = ax, alpha = 0.5)
    #引数 1:分類のモデル; 2: 入力データ; fill: 色塗るか; ax=どのスペースに書くのか; alpha: 透明度

    mglearn.discrete_scatter(x[:,0],x[:,1],y, ax = ax)
    #▲や●の点一つ一つを表示する

    ax.set_title("{} neighbors".format(n_neighbors))
    #タイトル

#ここまでがfor文
plt.show()


###より実践的なデータを用いる
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
print(cancer.feature_names) #列名取得

#mglearn.discrete.scatterと同じことをplt.scatterでやる
plt.scatter(cancer.data[:,0],cancer.data[:,1], c="orange")
x0_blue = cancer.data[:,0][cancer.target == 0]
x1_blue = cancer.data[:,1][cancer.target == 0]
x0_red = cancer.data[:,0][cancer.target == 1]
x1_red = cancer.data[:,1][cancer.target == 1]
plt.scatter(x0_blue, x1_blue, c = "blue", alpha=0.5)
plt.scatter(x0_red, x1_red, c = "red", alpha=0.5)
plt.show()

x_train, x_test, y_train, y_test = split(cancer.data, cancer.target, random_state=0, stratify=cancer.target)

clf = KNeighborsClassifier(n_neighbors=3)
clf = clf.fit(x_train, y_train)
prad = clf.predict(x_test)
score = clf.score(x_test, y_test)
print("正解率: {:2f}".format(score))

#np.contatement()で列を追加できる