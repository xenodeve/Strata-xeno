from collections import Counter
try:
    Counter.fromkeys('ab')
except NotImplementedError as e:
    print(e)
