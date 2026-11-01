# expect: RecursionLimit
f = lambda n: f(n + 1)
f(0)
