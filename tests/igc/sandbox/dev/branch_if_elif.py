for x in [-1, 0, 1, 15, 30]:
    if x < 0:
        print('neg')
    elif x == 0:
        print('zero')
    elif x % 15 == 0 and x > 15:
        print('fizzbuzz big')
    elif x % 15 == 0:
        print('fizzbuzz')
    else:
        print('other', x)
