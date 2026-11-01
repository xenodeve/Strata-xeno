def even(n):
    return True if n == 0 else odd(n - 1)
def odd(n):
    return False if n == 0 else even(n - 1)
print(even(50), odd(51), even(51))
