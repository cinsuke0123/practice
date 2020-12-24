#1　変数について
print("(1)")
a = 3
b = 5
print(a*b)
print(a+b)
print(a/b)

print()
#2　if文について。a*bをcに代入し、2の倍数なら"ワカチコ"と表示するプログラムを書く
print("(2)")
c = a * b
if c % 2 == 0:
    print("ワカチコ")
else :
    print("うんてぃうんてぃ")

print()
#3　配列について。要素を3つ含む配列を定義し、左から2番目の値をprintしなさい
print("(3)")
unti = [ "a", "b", "c" ]
print(unti[1]) #0から数えるので2番目の要素は数字の1

print()
#4　for文について。for文を用いて3で定義した配列をいてレートしなさい
#いてレートとは、for文で一個ずつ出すこと
print("(4)")
for u in unti :
    print( u )
