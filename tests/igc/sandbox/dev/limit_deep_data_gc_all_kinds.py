a = []
b = {}
c = ()
for i in range(50000):
    a = [a]
    b = {'x': b}
    c = (c,)
a = b = c = None
print('freed all')
