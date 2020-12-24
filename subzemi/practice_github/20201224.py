import numpy as np
import folder1 as fl

"""
for文
"""

### 3の倍数だけ表示する
def nabeatsu(value):
    if value % 3 == 0 and value != 0:
        print(value)
for i in range(100):
    nabeatsu(i)

### 辞書からkey, valueを取り出す
dic = { "key1": "item1", "key2": "item2"}
for key, value in dic.keys(), dic.items():
    print(key)
    print(value)

def f(x):
    return x ** 2

x = np.arange(-10, 10, 0.1)
y = f(x = x)

l_obj = fl.lineplot(xlim = (-10, 10), ylim = (-1, 100), filename = "graphs/lineplot.png")
s_obj = fl.scatter(xlim = (-10, 10), ylim = (-1, 100), filename = "graphs/scatter.png")

l_obj.plot(x = x, y = y)
s_obj.plot(x = x, y = y)
