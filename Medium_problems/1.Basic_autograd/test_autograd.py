from basic_autograd import Value
a = Value(2)
b = Value(3)
c = Value(10)
d = a + b * c
e = Value(7) * Value(2)
f = e + d
g = f.relu()
g.backward(); 
print("result :\n",a,b,c,d,e,f,g)
print("expected: \nValue(data=2, grad=1) Value(data=3, grad=10) Value(data=10, grad=3) Value(data=32, grad=1) Value(data=14, grad=1) Value(data=46, grad=1) Value(data=46, grad=1)")