n = 1234567
print(f'{n:d} {n:,} {n:_} {n:x} {n:X} {n:o} {n:b} {n:#x} {n:#o} {n:#b} {n:#X} {n:e} {n:.2e} {n:f} {n:.1f} {n:g} {n:%} {n:c}'.encode if False else f'{n:d} {n:,} {n:_} {n:x} {n:X} {n:o} {n:b} {n:#x} {n:#o} {n:#b} {n:#X} {n:e} {n:.2e} {n:f} {n:.1f} {n:g} {n:%}')
print(f'{5:5d}|{5:<5d}|{5:^5d}|{5:05d}|{-5:05d}|{5:+d}|{-5:+d}|{5: d}|{-5: d}|{5:=+5d}|{5:*>5d}|{5:*<5}|{5:*^5}|{-5:*=5}|{0:d}|{0:x}|{255:08b}|{255:#010b}|{255:#06x}')
print(f'{1000:,}|{1000000:,}|{999:,}|{-1234:,}|{1234:08,}|{1234:09,}|{1234:010,}|{0:,}|{1234:_x}|{65535:_b}|{65:c}|{10**20:,}|{-10**20:x}')
for spec in ['.2', 'q', ',x', ',b', '_c', ',,', ',_', '+c']:
    try:
        print(format(255, spec))
    except ValueError as e:
        print(e)
print(format(True, 'd'), format(True, '>6'), format(False, 'x'), format(True, ''), f'{True}', f'{True:d}', f'{True:5}')
