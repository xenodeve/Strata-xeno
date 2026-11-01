x = 'g'
def a():
    return x
def b():
    x = 'l'
    return x
def c():
    try:
        print(x)
    except UnboundLocalError as e:
        print(e)
    x = 1
def d():
    for x in [1]:
        pass
    return x
def e():
    global x
    x = 'changed'
    return x
print(a(), b()); c(); print(d(), e(), x)
def f():
    del undefined_local
    undefined_local = 1
try:
    f()
except UnboundLocalError as err:
    print(err)
def g():
    try:
        del gone
    except NameError as err:
        print(type(err).__name__, err)
    gone = 1
    del gone
    try:
        gone
    except NameError as err:
        print(type(err).__name__, err)
g()
try:
    del nothing_here
except NameError as err:
    print(err)
print(__name__)
if __name__ == '__main__':
    print('main')
