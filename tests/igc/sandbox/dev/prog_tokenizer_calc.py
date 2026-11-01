def tokenize(s):
    out, i = [], 0
    while i < len(s):
        c = s[i]
        if c.isspace():
            i += 1
        elif c.isdigit() or c == '.':
            j = i
            while j < len(s) and (s[j].isdigit() or s[j] == '.'):
                j += 1
            out.append(float(s[i:j]))
            i = j
        else:
            out.append(c)
            i += 1
    return out
def parse(tokens):
    pos = 0
    def expr():
        nonlocal pos
        v = term()
        while pos < len(tokens) and tokens[pos] in ('+', '-'):
            op = tokens[pos]
            pos += 1
            v = v + term() if op == '+' else v - term()
        return v
    def term():
        nonlocal pos
        v = atom()
        while pos < len(tokens) and tokens[pos] in ('*', '/'):
            op = tokens[pos]
            pos += 1
            v = v * atom() if op == '*' else v / atom()
        return v
    def atom():
        nonlocal pos
        t = tokens[pos]
        pos += 1
        if t == '(':
            v = expr()
            pos += 1
            return v
        if t == '-':
            return -atom()
        return t
    return expr()
for s in ['1 + 2 * 3', '(1 + 2) * 3', '10 / 4 - 1', '-(2 + 3) * 2.5', '2 * (3 + (4 - 1)) / 3']:
    print(s, '=', parse(tokenize(s)))
