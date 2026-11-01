# expect: Unsupported
try:
    raise SystemExit(3)
except SystemExit as e:
    print(e.code)
