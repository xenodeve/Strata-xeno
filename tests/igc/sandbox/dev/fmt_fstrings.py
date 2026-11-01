x, y, s = 3.14159, 42, 'hi'
print(f'{x:.3f} {y:d} {s:>8} {x:08.3f} {y:,} {y:x} {y:b} {x:e}')
print(f'{y!r} {s!r} {s!s:^6}|{1000000:,}|{-5:05d}|{255:#x}')
w = 10
print(f'{x:{w}.2f}|{{literal}}|{y=}')
print(f'{0.5:.0%} {1e10:.2e} {3:<4}| {True} {None}')
