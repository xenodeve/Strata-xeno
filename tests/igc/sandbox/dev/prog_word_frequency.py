text = '''the quick brown fox jumps over the lazy dog the fox
barks and the dog runs, the end.'''
words = [w.strip('.,') for w in text.lower().split()]
freq = {}
for w in words:
    freq[w] = freq.get(w, 0) + 1
top = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))[:4]
print(top, len(words), len(freq), max(words, key=len), sorted({len(w) for w in words}))
from collections import Counter
print(Counter(words).most_common(3), Counter(len(w) for w in words).most_common(2), sum(1 for w in words if w.startswith('t')))
