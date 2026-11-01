# expect: RecursionLimit
from collections import defaultdict
d = defaultdict(lambda: d[len(d) + 1])
d[0]
