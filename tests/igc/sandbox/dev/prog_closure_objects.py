def account(balance=0):
    history = []
    def deposit(x):
        nonlocal balance
        balance += x
        history.append(('d', x))
        return balance
    def withdraw(x):
        nonlocal balance
        if x > balance:
            raise ValueError(f'insufficient: {balance} < {x}')
        balance -= x
        history.append(('w', x))
        return balance
    return {'deposit': deposit, 'withdraw': withdraw, 'history': lambda: list(history), 'balance': lambda: balance}
a, b = account(10), account()
print(a['deposit'](5), a['withdraw'](3), b['deposit'](1), a['history'](), b['balance']())
try:
    b['withdraw'](9)
except ValueError as e:
    print('error:', e)
ops = [('deposit', 1), ('withdraw', 100), ('deposit', 2)]
for op, x in ops:
    try:
        print(op, a[op](x))
    except ValueError as e:
        print('skip', e)
