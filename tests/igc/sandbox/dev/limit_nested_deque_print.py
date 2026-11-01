# expect: RecursionLimit
from collections import deque
d = deque()
for i in range(100000):
    d = deque([d])
print(d)
