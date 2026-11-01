# expect: StepLimit
while True:
    try:
        {}['x']
    except KeyError:
        pass
