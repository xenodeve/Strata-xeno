state, count = 'a', 0
for c in 'aabbbcabc':
    if state == 'a' and c == 'b':
        state = 'b'
    elif state == 'b' and c == 'c':
        state = 'a'
        count += 1
print(state, count)
