def shift(s, k):
    out = []
    for c in s:
        if c.isalpha():
            base = ord('a') if c.islower() else ord('A')
            out.append(chr((ord(c) - base + k) % 26 + base))
        else:
            out.append(c)
    return ''.join(out)
e = shift('Hello, World!', 3)
print(e, shift(e, -3), shift('xyz', 30))
print(''.join(sorted('hello')), 'hello'[::-1] == 'olleh', ''.join(reversed('abc')), ' '.join(s.capitalize() for s in 'a bc def'.split()), 'a-b_c'.replace('-', ' ').replace('_', ' ').split())
