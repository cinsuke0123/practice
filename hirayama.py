"""
ゼロからやさしくはじめるPython入門
"""



"""
def main()だと初めに関数を指定してから、関数の中身をあとで作れる。
"""
"""
def main():
    print(tesu(3, 4))
    print(sub(6))
    gin("おもてなし")

def tesu(a, b):
        return a+b
def sub( c ):
    return c*5
def gin( d ):
    print(d)
main()
"""


"""
天気情報を表示するプログラムを作成
"""
"""
#東京の天気をjsonで取得できるurl
#http://weather.livedoor.com/forecast/webservice/json/v1?city=130010
url = "http://weather.livedoor.com/forecast/webservice/json/v1"
url += "?city=130010"

#webから天気情報を取得する
import urllib.request as req
res = req.urlopen(url)
json_data = res.read()

#json dataをpythonのデータ型（今回は辞書型）に変換
import json
data = json.loads(json_data)

#結果を出力
for row in data["forecasts"]:
    label = row["dateLabel"]
    telop = row["telop"]
    print(label + ":" + telop)

#for k in data["description"]:　エラー
#    text = k["text"]
#    print(text)

desc = data["description"]
print(desc["text"])

def print_name():
    print(__name__)

if __name__ == '__main__':
    print_name()

"""

"""
グローバル変数：関数外で定義される、関数内でも参照可能な変数
ローカル変数：関数内で定義される、関数内でしか参照できない変数
"""
"""
value = 3939

def hoge1():
    print(value)

def hoge2():
    global value #グローバル変数として定義
    value = 5775 #グローバル変数として書き換え
    print(value)

hoge1()
hoge2()
print(value)
"""

"""
任意の個数の引数を指定する方法
*を引数名の前につけることで、リスト型で全ての引数の値を取得
"""
"""
def tasu(*args): #*でリスト型で全ての引数の値を取得
    r = 0
    for i in args:
        r += i
        return r

print(tasu(45,12,5,89))

#引数の前に**をつけた場合、辞書型で任意の数のキーワード引数を受け取ることができる。
def show_keywords(**args):
    print("===")
    for key, value in args.items():
        print(key + "=" + str(value))
    print("===")

show_keywords(a = 3, b = 4)
"""


"""
無名関数 lambda
変数 = lambda 引数: 式
"""
"""
tasu = lambda a,b: a + b
print(tasu(2,5))
"""

######ここから機械学習
"""
グラフの描画
"""
"""
import matplotlib.pyplot as plt
import numpy as np

x = np.arange(0, 20, 0.1) #0から200までの数列を0.1刻みで作成
y = np.sin(x) #xのsin
plt.plot(x,y) #グラフを描画
plt.show() #表示
"""

"""
大量データ入手
"""
"""
from sklearn import datasets
iris = datasets.load_iris()
#print(iris.DESCR)
#print(iris.data)
#print(iris.target)
"""

"""
train_test_split()関数の書式
学習用データ, テスト用データ, 学習用の正解ラベル, テスト用の正解ラベル =
    train_test_split(
        データ, データの正解ラベル,
        train_size = 学習用データの割合,
        test_size = テスト用データの割合
    )
"""
"""
from sklearn.model_selection import train_test_split as split
x_train, x_test, y_train, y_test = \
    split(iris.data, iris.target,
          train_size = 0.8,
          test_size = 0.2
          )

#二つのリストを合成して一つに。この出力で出てくるarrayはndarray型と言ってnumpy独自の高速処理できるリスト
print(list(zip(x_train, y_train))) #print(list(zip(x_train, y_train)))...

###zip関数について
#c = list(zip(a,b))　リストの合成
#zip(*c) リストの分解

from sklearn import svm

#SVCメソッドでSVCアルゴリズムのオブジェクトを生成
clf = svm.SVC()

#fit()メソッドでデータを学習（モデルに実際の値を読み込ませる作業）
clf.fit(x_train, y_train)
print(clf.fit(x_train, y_train))

#predict()メソッドでテストデータを利用して予測（fit()メソッドで読み込ませた値から推定値を計算して求める作業）
pred = clf.predict(x_test)

#予測が答えとどれくらいあっているかカウントして割合を出す
result = list(pred == y_test).count(True) / len(y_test)
print("正解率: " + str(result))
print()
cnt = 0
for pre, y in zip(pred, y_test):
    result = (pre == y)
    print(pre, y, result)
    if result: cnt += 1 #もしresultがtrueだった場合、cntに１を足す
print("{0}/{1}={2}".format(cnt, len(y_test), cnt/len(y_test)))
"""
print()

"""
手書き文字認識
"""
"""
from sklearn.datasets import load_digits
digits = load_digits()
#print(digits.DESCR) #データの説明を表示
index = 0 #先頭（0番目）の要素
print(digits.data[index]) #dataの一覧から0番目の画像データを取ってくる
# n番目の画像データには、それぞれ64個数字が入っているが、
# これは一つの画像を8*8の64ピクセルに分解したそれぞれのピクセルがどれくらい色が濃いのかを0~16で示している
# 0に近い方が黒いことを示している
print(digits.target[index]) #実際にどの数字を表すデータなのか（これはn番目の画像データがどの数字を書いたデータなのかを示すデータ）

import matplotlib.pyplot as plt
plt.matshow(digits.images[0], cmap = "gray") #cmapで色を指定できる, digits.imagesは画像を画面に出力するために用意されたデータ
plt.show()

###numpyのndarray型（配列型）は配列データの次元の数を変えることができる。
import numpy as np
a = np.array([1,2,3,4,5,6,7,8]) #1組だけ（8つ）の1次元配列
a.reshape(4,2) #4*2の2次元配列に変える

for i in [3,10,12]:
    n = 16-digits.data[i]
    nr = n.reshape(8,8)
    plt.matshow(nr, cmap="gray")
plt.show()

print()
###SVMアルゴリズムを使用
from sklearn.model_selection import train_test_split as split
x, x_test, y, y_test = split(digits.data, digits.target, train_size = 0.75,
        test_size = 0.25)

from sklearn import svm
clf = svm.SVC()
clf.fit(x,y)

pred = clf.predict(x_test)
result = list(pred == y_test).count(True)/len(y_test)
print("SVCアルゴリズムの正解率: " + str(result))
print()

###学習アルゴリズムを変更
clf = svm.LinearSVC()
clf.fit(x,y)

pred = clf.predict(x_test)
result= list(pred == y_test).count(True)/len(y_test)
print("Linear SVCアルゴリズムの正解率: " + str(result))

###学習済みモデルを保存
from sklearn.externals import joblib
joblib.dump(clf, "digits.pkl", compress=True) #dump関数でファイルに学習済みモデルを保存したり、読み込むことができる。
"""



"""
flikcr apiを用いる
"""

from flickrapi import FlickrAPI
from urllib.request import urlretrieve    #画像をダウンロードする際に利用
import os, time, sys

key = "88a9081c4a90da6f67768b534f1b29ca"
secret = "1d12639432894723"
wait_time = 1 #検索のWait時間を指定（1秒間隔で取得）
keyword_cat = "ねこ"   #検索したいキーワードを指定
savedir_cat = "./" + keyword_cat   #画像を保存するディレクトリを指定
keyword_dog = "いぬ"
savedir_dog = "./" + keyword_dog   #画像を保存するディレクトリを指定
flickr = FlickrAPI(key, secret, format='parsed-json')   #FlickrAPIクラスからインスタンスを生成

result_cat = flickr.photos.search(
    text = keyword_cat,   #検索キーワードを指定
    per_page = 100,   #取得枚数を指定
    media = 'photos',   #データの種類を指定
    sort = 'relevance', #最新から取得
    safe_search = 1,  #暴力的な画像は対象外にする
    extras = 'url_q, license' #URLとライセンス情報を取得する。
    )
result_dog = flickr.photos.search(
    text = keyword_dog,   #検索キーワードを指定
    per_page = 100,   #取得枚数を指定
    media = 'photos',   #データの種類を指定
    sort = 'relevance', #最新から取得
    safe_search = 1,  #暴力的な画像は対象外にする
    extras = 'url_q, license' #URLとライセンス情報を取得する。
    )

"""
#猫の画像をダウンロードし、ねこファイルに格納
photos_cat = result_cat['photos']  # 写真データ部分を取り出す。この中にはもちろん、url_qとlicenseの内容もそれぞれの写真データに入っている。
download_path = os.path.join("/Users/tsuchiyayoshimi/Desktop/practice", keyword_cat)  # 検索ワードのフォルダパス

for i, photo in enumerate(photos_cat['photo']):
    url = photo['url_q']
    filepath = savedir_cat + '/' + photo['id'] + '.jpg'
    if not os.path.exists(download_path):  # 検索ワードのフォルダがなければ生成
        current_dir = os.path.dirname(os.path.abspath("__file__"))
        os.mkdir(os.path.join(current_dir, keyword_cat))
    urlretrieve(url, filepath)  # 写真データをダウンロード
    time.sleep(wait_time)  # 1秒待機
"""
"""
#犬の画像をダウンロードし、いぬファイルに格納
photos_dog = result_dog['photos']  # 写真データ部分を取り出す。この中にはもちろん、url_qとlicenseの内容もそれぞれの写真データに入っている。
download_path = os.path.join("/Users/tsuchiyayoshimi/Desktop/practice", keyword_dog)  # 検索ワードのフォルダパス

for i, photo in enumerate(photos_dog['photo']):
    url = photo['url_q']
    filepath = savedir_dog + '/' + photo['id'] + '.jpg'
    if not os.path.exists(download_path):  # 検索ワードのフォルダがなければ生成
        current_dir = os.path.dirname(os.path.abspath("__file__"))
        os.mkdir(os.path.join(current_dir, keyword_dog))
    urlretrieve(url, filepath)  # 写真データをダウンロード
    time.sleep(wait_time)  # 1秒待機
"""
"""
データ読み込み
"""
import glob2 as glob
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import pandas as pd
from sklearn.model_selection import train_test_split as split
from sklearn import svm
import numpy as np

kind_list = ["ねこ", "いぬ"]
#画像をリストにarrayにしていれる。ラベルもリストの形で作る
#空の配列を作成
data_img=[] #画像用
kind_label=[] #ラベル用
#リストに入れる際の画像のサイズを指定
img_size = 224

#print(glob.glob(savedir_cat + '/' '*.jpg'))


for j in  kind_list:
            #file_listのなかに画像ファイルのpathを取得
            file_list = glob.glob("./" + str(j) + '/' '*.jpg')
            for file in file_list:
                img_path = file
                #画像を読み込む
                img = load_img(img_path, target_size=(img_size, img_size))
                #読み込んだ画像をarrayに変換
                x = img_to_array(img)
                #作成したdata_imgのリストの中に入れる
                data_img.append(x)
                #画像の名前の種類の部分をラベルとしてkind_labelのリストの中に入れる
                kind_label.append(j)

#ラベル(kind_label)をダミー変数に直す
Y_dummy = pd.get_dummies(kind_label)
#Y_dummyが2列（ねこダミー列、いぬダミー列）あるので、どちらか一つにアウトカムを統一しないといけない。ValueError: bad input shape (180, 2)
Y_dummy = Y_dummy["いぬ"]

#画像データ(data_img)とラベル(Y_dummy)をtrainとtestに分ける
x_train, x_test, y_train, y_test = split(
    data_img, Y_dummy, test_size=0.1,  stratify=Y_dummy)

dataset_size = len(x_train)
x_train = np.array(x_train)
#reshape(1次元目の数, 2次元目の数,...,n次元目の数)
#今回は2次元目まで欲しいので、2次元目まで指定。2次元目の-1は成り行きに任せるみたいな意味。
#1次元目がdataset_sizeと指定されており、-1はそれに合わせる形で2次元目が決めるという指定。
x_train = x_train.reshape(dataset_size, -1)
x_test = np.array(x_test)
x_test = x_test.reshape(len(x_test), -1)


"""
推定
"""

clf = svm.SVC()
clf.fit(x_train, y_train)

pred = clf.predict(x_test)
result = list(pred == y_test).count(True)/len(y_test)
print("SVCアルゴリズムの正解率: " + str(result))
print()






