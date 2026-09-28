import struct, sys, os, collections
def rd(f, fmt): return struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))
def rstr(f): (n,) = rd(f, "Q"); return f.read(n).decode("utf-8", "replace")
def skip_val(f, t):
    sz = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
    if t in sz: f.read(sz[t]); return
    if t == 8: rstr(f); return
    if t == 9:
        (et, n) = rd(f, "IQ")
        for _ in range(n): skip_val(f, et)
def tensors(path):
    f = open(path, "rb"); assert f.read(4) == b"GGUF"
    ver, nt, nkv = rd(f, "IQQ"); align = 32
    for _ in range(nkv):
        k = rstr(f); (t,) = rd(f, "I")
        if k == "general.alignment": (align,) = rd(f, "I")
        else: skip_val(f, t)
    info = []
    for _ in range(nt):
        name = rstr(f); (nd,) = rd(f, "I"); dims = rd(f, "Q" * nd); (ty, off) = rd(f, "IQ")
        info.append((off, name, ty, dims))
    base = (f.tell() + align - 1) // align * align
    end = os.path.getsize(path) - base
    info.sort()
    out = []
    for i, (off, name, ty, dims) in enumerate(info):
        nxt = info[i + 1][0] if i + 1 < len(info) else end
        out.append((name, ty, dims, nxt - off))
    return out
d = sys.argv[1]
allt = []
for p in sorted(os.listdir(d)):
    if p.endswith(".gguf"): allt += tensors(os.path.join(d, p))
cls = collections.Counter(); per = collections.Counter()
for name, ty, dims, sz in allt:
    kind = "experts" if "_exps" in name else "trunk"
    cls[kind] += sz
    if kind == "trunk":
        key = name.split(".", 2)[-1] if name.startswith("blk.") else name
        per[(key, ty)] += sz
for k, v in cls.items(): print(f"{k:8} {v/2**20:10.1f} MiB")
for (k, ty), v in per.most_common(25): print(f"  {v/2**20:9.1f} MiB  type {ty:3}  {k}")
