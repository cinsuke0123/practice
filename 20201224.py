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
for key, value in dic:
    print(key)
    print(value)
