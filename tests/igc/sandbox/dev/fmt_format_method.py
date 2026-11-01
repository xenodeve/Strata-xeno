print('{}'.format(1), '{} {}'.format(1, 2), '{0} {0}'.format('a'), '{1}{0}'.format('a', 'b'), '{x}'.format(x=1), '{x} {}'.format(2, x=1), '{:>5}'.format('a'), '{:05.1f}'.format(3.14159), '{0:{1}}'.format('a', 5), '{:{}}'.format('a', 5), '{:{w}}'.format('a', w=3), '{!r}'.format('a'), '{!s}'.format(1), '{{}}'.format(), '{{{}}}'.format(1), '{0[0]}'.format([5]), '{0[k]}'.format({'k': 1}), '{a[0]}'.format(a=[7]), '{0.real}'.format(5), '{:,}'.format(1000), '{:%}'.format(0.5), '{:e}'.format(1000), '{:b}'.format(5))
for f, a in [('{', ()), ('}', ()), ('{}', ()), ('{1}', (1,)), ('{x}', ()), ('{} {0}', (1,)), ('{0} {}', (1,)), ('{!x}', (1,)), ('{:q}', (1,)), ('{0.nosuch}', (1,)), ('{0[5]}', ([1],))]:
    try:
        print(f.format(*a))
    except (ValueError, IndexError, KeyError, AttributeError) as e:
        print(type(e).__name__, e)
