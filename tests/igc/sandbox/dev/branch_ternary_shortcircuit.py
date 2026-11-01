def t(v):
    print('t', v)
    return v
print(t(0) or t(2) or t(3))
print(t(1) and t(0) and t(3))
print('a' if t(0) else 'b')
print(not t([]))
print(1 < 2 < 3, 3 > 2 > 2, 1 == 1.0 == True)
print(t(1) < t(2) < t(0) < t(5))
