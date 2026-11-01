i = 0
while i < 3: i += 1
print(i)
while False:
    print('never')
else:
    print('else runs')
n = 0
while True:
    n += 1
    if n > 2: break
else:
    print('no else')
print(n)
for i in []:
    print('never')
else:
    print('for else')
for i in range(2): print(i); print('same line')
if 1: print('a'); print('b')
if 0: print('c')
else: print('d')
if 0: pass
elif 1: print('e')
x = 1; y = 2; z = x + y; print(z);
def one(): return 1
def two(): x = 2; return x
print(one(), two())
class_ = 1; def_ = 2; print(class_ + def_)
print((1 +
       2), [1,
            2], {'a':
                 1}, f(
    ) if False else 0)
s = 1 + \
    2
print(s)
if (s and
        s):
    print('multi')
pass; pass
