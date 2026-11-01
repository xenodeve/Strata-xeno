x = 1
def setx():
    global x
    x = 2
def readx():
    return x
print(readx())
setx()
print(readx())
y = 10
def shadow():
    y = 20
    def inner():
        return y
    return inner()
print(shadow(), y)
