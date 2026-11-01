# expect: StepLimit
a = {i: i for i in range(10000)}
while True:
    b = a.copy()
