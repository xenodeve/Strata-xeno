# expect: StepLimit
import math
x = math.isqrt(10 ** 100000)
for i in range(100):
    x = math.isqrt(10 ** 100000)
