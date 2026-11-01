balance, history = 100.0, []
for op, amount in [('dep', 50), ('wd', 30.5), ('wd', 500), ('dep', 0.25)]:
    if op == 'dep':
        balance += amount
    elif amount <= balance:
        balance -= amount
    else:
        history.append(f'declined {amount}')
        continue
    history.append(f'{op} {amount:.2f} -> {balance:.2f}')
print('\n'.join(history))
