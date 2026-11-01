if (n := 5) > 3:
    print(n)
while (m := n - 1) > 2:
    n = m
print(n, m)
print((y := 1) + y, y)
def f():
    if (a := 1):
        return a
print(f())
