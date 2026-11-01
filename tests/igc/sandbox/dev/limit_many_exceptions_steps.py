# expect: StepLimit
while True:
    try:
        raise ValueError
    except ValueError:
        pass
