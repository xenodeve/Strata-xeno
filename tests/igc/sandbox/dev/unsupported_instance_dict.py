# expect: Unsupported
class A:
    def __init__(self):
        self.v = 1
print(A().__dict__)
