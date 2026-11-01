print(ord('a'), ord('é'), ord('ก'), ord('😀'), chr(0), repr(chr(10)), chr(233), chr(3585), chr(128512), hex(0), hex(-1), hex(255), oct(0), oct(-8), bin(0), bin(-5), hex(2 ** 64), oct(2 ** 64), bin(2 ** 64), hex(True))
for bad in [lambda: ord('ab'), lambda: ord(''), lambda: ord(1), lambda: chr(-1), lambda: chr(1114112), lambda: chr('a'), lambda: hex(1.5), lambda: bin('a'), lambda: oct(None)]:
    try:
        print(bad())
    except (TypeError, ValueError) as e:
        print(type(e).__name__, e)
