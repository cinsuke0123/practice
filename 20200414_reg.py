import mglearn
import pandas as pd
from sklearn.model_selection import train_test_split as split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, Lasso

x, y = mglearn.datasets.load_extended_boston()

df_x = pd.DataFrame(x)
df_y = pd.DataFrame(y)

x_train, x_test, y_train, y_test = split(df_x, df_y, random_state=0)

###線形回帰
lin_reg = LinearRegression().fit(x_train, y_train)

print(round(lin_reg.score(x_train, y_train), 3)) #訓練データでの正解率
print(round(lin_reg.score(x_test,y_test),3)) #テストデータでの正解率

###ridge回帰
ridge = Ridge().fit(x_train, y_train) #デフォではalpha=1　係数の値を小さくしようとする

def print_score(model):
    print("訓練データの正解率: {}".format(round(model.score(x_train, y_train),3)))
    print("テストデータの正解率: {}".format(round(model.score(x_test, y_test),3)))

print_score(ridge)

#Ridge()のalphaをいじる
ridge_10 = Ridge(alpha=10).fit(x_train,y_train)
print_score(ridge_10)

###lasso回帰
lasso = Lasso().fit(x_train, y_train)
print_score(lasso)