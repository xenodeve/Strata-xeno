# expect: StepLimit
s = 'a' * 1000000
t = 'a' * 500000 + 'b'
print(s.find(t))
