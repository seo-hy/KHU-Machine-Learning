from sympy import *

x = Symbol('x')
b = Symbol('b')
v = Symbol('v')
w = Symbol('w')
y = Symbol('y')
loss = (x**2*w +x*v + b - y )**2
res = loss.diff(v)
res2 = loss.diff(w)
print(res)
print(res2)
