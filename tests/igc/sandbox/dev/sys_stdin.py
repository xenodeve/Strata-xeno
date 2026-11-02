import sys
from sys import maxsize, stdin

print(maxsize, sys.maxsize == 2**63 - 1, maxsize > 10**18)
print(sys.getrecursionlimit())
sys.setrecursionlimit(5000)
print(sys.getrecursionlimit())
print(sys.stdout.write('written\n'))
sys.stdout.flush()
print(repr(sys.stdin.read()), repr(sys.stdin.readline()), sys.stdin.readlines(), repr(stdin.read()))
try:
    input()
except EOFError as err:
    print('EOFError', err)
try:
    input('prompt ')
except EOFError:
    print('EOFError')
read = sys.stdin.readline
print(repr(read()))
data = sys.stdin.read().split()
print(data, len(data))


def main():
    if not data:
        print('no data')
        sys.exit()
    print('never')


main()
print('never')
