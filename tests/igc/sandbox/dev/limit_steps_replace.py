# expect: StepLimit
s = 'a' * 100000
while True:
    t = s.replace('a', 'bb')
