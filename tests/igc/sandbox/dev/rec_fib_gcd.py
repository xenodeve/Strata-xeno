def fib(n):
    return n if n < 2 else fib(n - 1) + fib(n - 2)
def gcd(a, b):
    return a if b == 0 else gcd(b, a % b)
print(fib(15), gcd(1071, 462))
def hanoi(n, a, b, c):
    return 0 if n == 0 else hanoi(n - 1, a, c, b) + 1 + hanoi(n - 1, c, b, a)
print(hanoi(8, 'a', 'b', 'c'))
