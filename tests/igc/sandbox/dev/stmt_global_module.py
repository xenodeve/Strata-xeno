global g1
g1 = 1
print(g1)
def f():
    global g2, g3
    g2, g3 = 2, 3
f()
print(g2, g3)
