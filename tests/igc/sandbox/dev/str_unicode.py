s = 'ไทย ñ 😀'
print(s, len(s), s[0], s[-1], s[::-1], s.split(), list(s[:2]), s.find('ñ'), s.replace('ñ', 'n'), s.count('ย'), s + '!', s * 2, s.startswith('ไ'), ord(s[-1]), f'{s:>10}|', '{:^10}'.format(s), s.encode if False else 'skip', s.strip('😀'), s.center(11, '*'), [c for c in 'é😀'], sorted('😀aé'), repr('😀'), max(s))
print('ก', '\U0001F600', '\xe9', '\101', 'a\
b', len('ก\U0001F600'))
