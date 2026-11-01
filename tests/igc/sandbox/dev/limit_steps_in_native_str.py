# expect: StepLimit
s = 'a' * 1000000
n = 0
while True:
    n += s.count('b')
