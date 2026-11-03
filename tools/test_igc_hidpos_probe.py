"""The hidden-row probe's reading of the engine's STRATA_HIDPOS file, and its cross-validation: python tools/test_igc_hidpos_probe.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from tools.igc.hidpos_probe import auroc, leave_one_group_out, rows

rng = np.random.default_rng(0)
W = 2560
# two requests, the second's rows after the first's in the file; a window whose position is absent gets no row
recs = [(5, rng.normal(size=W)), (9, rng.normal(size=W)), (3, rng.normal(size=W))]
blob = b''.join(np.int64(p).tobytes() + r.astype(np.float32).tobytes() for p, r in recs)
cut = 2 * (8 + 4 * W)
index = [dict(pos=9, file_from=0, file_to=cut), dict(pos=3, file_from=cut, file_to=len(blob)), dict(pos=7, file_from=0, file_to=cut),
         dict(pos=1, file_from=cut, file_to=cut)]
got = rows(index, blob)
assert np.allclose(got[0], recs[1][1].astype(np.float32)) and np.allclose(got[1], recs[2][1].astype(np.float32))
assert got[2] is None and got[3] is None, 'absent position, empty request'

# a direction that carries the label is found from held-out questions; labels that carry nothing are not
n, groups = 120, [f'q{i % 8}' for i in range(120)]
X = rng.normal(size=(n, 16))
y = (X[:, 3] + 0.3 * rng.normal(size=n) > 0).astype(float)
assert auroc(leave_one_group_out(X, y, groups, k=None, l2=1.0), y) > 0.9
noise = rng.integers(0, 2, n).astype(float)
assert abs(auroc(leave_one_group_out(X, noise, groups, k=8, l2=1.0), noise) - 0.5) < 0.15
assert auroc(np.array([0.1, 0.9, 0.4]), np.array([0., 1., 1.])) == 1.0
print('ok')
