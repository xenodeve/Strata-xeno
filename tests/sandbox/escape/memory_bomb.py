"""Breach: take memory until the machine has none. Stops by itself at the ceiling given in MiB. Each block is touched, so
it is really held and not only promised."""
import sys

ceiling, blocks = int(sys.argv[1]), []
try:
    while len(blocks) * 8 < ceiling:
        block = bytearray(8 << 20)
        block[::4096] = b"\1" * len(block[::4096])
        blocks.append(block)
except MemoryError as error:
    print(f"refused: {error!r} at {len(blocks) * 8} MiB", flush=True)
    sys.exit(10)
print(f"held {len(blocks) * 8} MiB")
