# expect: Unsupported
g = (x for x in [1])
g.send(None)
