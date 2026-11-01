lines = []
for i in range(1, 6):
    lines.append(' ' * (5 - i) + '*' * (2 * i - 1))
print('\n'.join(lines))
print('\n'.join(f'{i:2} x {j} = {i * j:3}' for i in (2, 11) for j in range(1, 3)))
print('|'.join(str(x).rjust(3) for x in [1, 22, 333]), 'abc'.center(9, '-'), '%-5s|%5s|' % ('a', 'b'), '{:^7}'.format('x'), ','.join(f'{x:.1f}' for x in [1, 2.5]))
