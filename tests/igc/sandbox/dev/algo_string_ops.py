def rle(s):
    out = []
    for c in s:
        if out and out[-1][0] == c:
            out[-1][1] += 1
        else:
            out.append([c, 1])
    return ''.join(f'{c}{n}' for c, n in out)
def is_pal(s):
    t = [c.lower() for c in s if c.isalnum()]
    return t == t[::-1]
print(rle('aaabccdd'), is_pal('A man, a plan, a canal: Panama'), ' '.join(w[::-1] for w in 'hello big world'.split()), sorted('the quick brown fox'.split(), key=lambda w: (len(w), w)))
