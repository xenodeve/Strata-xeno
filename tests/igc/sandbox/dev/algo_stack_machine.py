def run(tokens):
    st = []
    for t in tokens.split():
        if t in '+-*/':
            b, a = st.pop(), st.pop()
            st.append({'+': a + b, '-': a - b, '*': a * b, '/': a // b}[t])
        else:
            st.append(int(t))
    return st.pop()
print(run('3 4 + 2 * 7 -'), run('10 2 8 * + 3 -'))
