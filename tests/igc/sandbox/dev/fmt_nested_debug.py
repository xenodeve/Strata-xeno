w, p, v, name = 12, 3, 3.14159265, 'x'
print(f'{v:{w}.{p}f}|{v:>{w}}|{name:{"*"}^{w}}|{v=}|{v=:.2f}|{v = }|{name=!s}|{w+1=}|{[1,2]=}')
print(f'{{}} {{{w}}} {"a" "b"} {"x" if w else "y"} {w*2} {(lambda q: q + 1)(1)} {{"k": 1}}')
print(f'{ {"k": 1}["k"] } {[i for i in range(3)]} {dict(a=1)} {w!r:>5} {name!r:*^9}')
print(f'''{w}
{p}''' f"{v:.1f}" 'plain' "{}")
print(rf'\n{w}', fr'{w}\t', R'\n', r'\'', 'a' 'b' "c", '\x41\101A\U00000041', 'tab\there', "q\"q", 'line\
cont')
