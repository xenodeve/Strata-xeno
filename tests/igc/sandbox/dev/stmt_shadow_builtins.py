list = 5
print(list)
del list
print(list('a'))
def print2(x):
    return x
len = print2
print(len(5))
del len
print(len('ab'))
sum = lambda x: 'mine'
print(sum([1]))
def f(str):
    return str
print(f(1), str(2))
input = 1
print(input)
open = 2
print(open)
def eval(x):
    return x + 1
print(eval(1))
