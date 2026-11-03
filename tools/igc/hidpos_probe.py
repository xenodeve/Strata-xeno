"""Does the hidden state tell working by hand from other reasoning? (#288) A probe on the rows `external_bench.py --hidpos-from` wrote at
labelled windows, against the same probe on plain text features of the same windows, both left out one question at a time.
hidpos_probe.py <arm dir with hidpos.json and hidpos.bin> <records.json> [k]"""
import json, sys
from pathlib import Path
import numpy as np

HAND = ('hand_case', 'recheck_code')


def rows(index, blob):
    """Each window's row: the (int64 position, F32 row) records between its request's file offsets."""
    out = []
    for w in index:
        seg = blob[w['file_from']:w['file_to']]
        n = len(seg)
        if not n:
            out.append(None)
            continue
        width = None
        for cand in (10240, 2560, 5120, 8192):   # hc x n_embd of the packs in use; the record size has to divide the segment
            if n % (8 + 4 * cand) == 0:
                width = cand
                break
        rec = np.frombuffer(seg, dtype=np.uint8).reshape(-1, 8 + 4 * width)
        pos = rec[:, :8].copy().view(np.int64)[:, 0]
        hit = np.nonzero(pos == w['pos'])[0]
        out.append(rec[hit[0], 8:].copy().view(np.float32) if len(hit) else None)
    return out


def text_features(text):
    """Plain cues of working a case: digits, `=`, lines, and code words, per character."""
    n = max(len(text), 1)
    return np.array([sum(c.isdigit() for c in text) / n, text.count('=') / n, text.count('\n') / n,
                     sum(text.count(w) for w in ('def ', 'return', 'for ', 'print(', 'if ')) / n,
                     sum(text.count(w) for w in ('->', ' -> ', 'so ', 'then')) / n, len(text.split()) / n])


def fit(X, y, l2=1.0, steps=200):
    """L2 logistic regression by Newton steps."""
    X1 = np.hstack([X, np.ones((len(X), 1))])
    w = np.zeros(X1.shape[1])
    reg = l2 * np.eye(X1.shape[1]); reg[-1, -1] = 0
    for _ in range(steps):
        p = 1 / (1 + np.exp(-X1 @ w))
        g = X1.T @ (p - y) + reg @ w
        H = X1.T @ (X1 * (p * (1 - p))[:, None]) + reg + 1e-9 * np.eye(len(w))
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return w


def auroc(score, y):
    pos, neg = score[y == 1], score[y == 0]
    if not len(pos) or not len(neg):
        return float('nan')
    return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg)))


def leave_one_group_out(X, y, groups, k=None, l2=1.0):
    """Out-of-fold scores: standardised on the training fold, optionally its top k principal directions, then the probe."""
    score = np.zeros(len(y))
    for g in sorted(set(groups)):
        te = np.array([x == g for x in groups]); tr = ~te
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        A, B = (X[tr] - mu) / sd, (X[te] - mu) / sd
        if k:
            _, _, vt = np.linalg.svd(A, full_matrices=False)
            A, B = A @ vt[:k].T, B @ vt[:k].T
        w = fit(A, y[tr], l2)
        score[te] = np.hstack([B, np.ones((len(B), 1))]) @ w
    return score


if __name__ == '__main__':
    arm, records = Path(sys.argv[1]), json.loads(Path(sys.argv[2]).read_text(encoding='utf-8'))
    k = int(sys.argv[3]) if len(sys.argv) > 3 else 16
    index = json.loads((arm / 'hidpos.json').read_text(encoding='utf-8'))
    got = rows(index, (arm / 'hidpos.bin').read_bytes())
    keep = [i for i, r in enumerate(got) if r is not None]
    print('windows', len(index), 'with a row', len(keep))
    heads = {r['id']: r['text'].split('</think>')[0] for r in records}
    y = np.array([index[i]['activity'] in HAND for i in keep], dtype=float)
    groups = [index[i]['id'] for i in keep]
    H = np.stack([got[i] for i in keep]).astype(np.float64)
    mean_hc = H.reshape(len(H), -1, 2560).mean(1) if H.shape[1] % 2560 == 0 else H
    T = np.stack([text_features(heads[index[i]['id']][max(0, round(float(index[i]['key'].split('@')[1]) * len(heads[index[i]['id']])) - 800):
                                                   round(float(index[i]['key'].split('@')[1]) * len(heads[index[i]['id']])) + 800]) for i in keep])
    print('positives (by hand)', int(y.sum()), 'of', len(y), 'questions', len(set(groups)))
    for name, X, kk in (('text features', T, None), ('hidden, mean of streams', mean_hc, k), ('hidden, all streams', H, k)):
        for l2 in (0.1, 1.0, 10.0):
            print(f'{name:26} k={kk} l2={l2:5}: AUROC {auroc(leave_one_group_out(X, y, groups, kk, l2), y):.3f}')
