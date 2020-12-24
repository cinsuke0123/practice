"""
関数について

用途
①コードの再利用
②引数（input）と返り値（output）
"""
#(1)
#a = 1, b = 3, c = a + b
a = 1
b = 3
c = a + b
print(c)
print()

#(2)
#cが3のばいい数の時 print("わーい")を実行するプログラム
if c % 3 == 1:
    print("わーい")
print()

#(3) arr = [3,4,5]をいてレートしなさい
arr = [3,4,5]
for i in arr:
    print(i)
print()

#(4) (1)~(3)を関数にする
def add( x, y ):
    print ( x + y )

add(40, 60)
print()

def two(x):
    if x % 3 == 1:
        print("わーい")

two(4)
print()

def three(x):
    for i in x:
        print( i )

x = [3,4,5]
three( x )

print()


#(5)フィボナッチ数列
#初項を任意の引数で持つフィボナッチ数列を第１万行まで求めよ
#なお、出力結果が1万個ある証拠を示せ
#関数を定義する
#漸化式

def fibonachi( a_1 , a_2 , n ):
    print( "第1項目 :" , a_1 )
    print( "第2項目 :" , a_2 )
    for i in range( n-2 ):
        a_3 = a_1 + a_2
        a_1 = a_2
        a_2 = a_3
        print( "第"+str(i+3)+"項目 :" , a_3 )

fibonachi( a_1 = 1 , a_2 = 1 , n=100 )



