state, credit, log = 'idle', 0, []
for ev, val in [('coin', 25), ('coin', 25), ('select', 75), ('coin', 50), ('select', 75), ('refund', 0), ('select', 10)]:
    if ev == 'coin':
        credit += val
        state = 'has_credit'
    elif ev == 'select':
        if credit >= val:
            credit -= val
            log.append(f'vend@{val}')
            state = 'has_credit' if credit else 'idle'
        else:
            log.append(f'need {val - credit}')
    elif ev == 'refund':
        log.append(f'refund {credit}')
        credit, state = 0, 'idle'
print(state, credit, log)
