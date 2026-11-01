print(any([]), all([]), any([0, '']), all([1, 'a']), any(x for x in [0, 1]), all(x for x in [1, 0]), any({0: 1}), all({0: 1}), any('a'), all(''))
