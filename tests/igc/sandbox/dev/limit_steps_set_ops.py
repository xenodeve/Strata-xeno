# expect: StepLimit
a = set(range(10000))
while True:
    b = a | a
