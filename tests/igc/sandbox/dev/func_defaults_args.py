def f(a, b=2, *args, c=3):
    return (a, b, args, c)
print(f(1))
print(f(1, 5))
print(f(1, 5, 6, 7))
print(f(1, c=9))
print(f(b=4, a=8))
def shared(x, acc=[]):
    acc.append(x)
    return acc
print(shared(1), shared(2))
