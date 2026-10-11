"""Face verification pipeline: data -> preprocess -> train/test split -> train -> test -> save.

The face models are pretrained, so "training" = picking the similarity thresholds.
Data: LFW pairs (same / different person), downloaded by scikit-learn (~230 MB, once).
  pass   threshold: no false matches among the train "different person" pairs
  review threshold: best overall accuracy on train
  duplicate threshold: one face searched against EVERY account (duplicate accounts) — expected false hits per
                       check = per-pair FAR x accounts, so it is set at FAR 1e-6 over all LFW stranger pairs
LFW is mostly Western celebrities: recalibrate on consented Indian selfie/profile pairs
and report error rates per skin-tone group before trusting these numbers (plan §7.2).
Run from ml/:  python -m face_verification.train
"""
import cv2
import numpy as np
from sklearn.datasets import fetch_lfw_pairs

from common.utils import DATA, save_artifact
from face_verification.face import decide, embed


def similarities(subset):
    lfw = fetch_lfw_pairs(subset=subset, color=True, resize=1.0, slice_=None, data_home=DATA)
    pairs = lfw.pairs if lfw.pairs.max() > 1.5 else lfw.pairs * 255  # sklearn versions differ in scaling
    sims, labels, rejected = [], [], 0
    for (a, b), same in zip(pairs, lfw.target):
        ea, _ = embed(np.ascontiguousarray(a[..., ::-1]).astype(np.uint8))  # RGB -> BGR
        eb, _ = embed(np.ascontiguousarray(b[..., ::-1]).astype(np.uint8))
        if ea is None or eb is None:
            rejected += 1
            continue
        sims.append(float(ea @ eb))
        labels.append(int(same))
    return np.array(sims), np.array(labels), rejected


def duplicate_threshold(max_far=1e-6):
    """Every LFW face against every other person's (~13k faces, ~80M stranger pairs): the lowest similarity
    at which at most max_far of stranger pairs match. Also reports how many same-person pairs it still finds."""
    root = DATA / "lfw_home" / "lfw_funneled"
    embs, people = [], []
    for n, person in enumerate(sorted(p for p in root.iterdir() if p.is_dir())):
        for img in sorted(person.glob("*.jpg")):
            e, _ = embed(cv2.imread(str(img)))
            if e is not None:
                embs.append(e)
                people.append(n)
    E, ids = np.array(embs, np.float32), np.array(people)
    strangers, same, tail = 0, [], []
    for i in range(0, len(E), 1000):  # upper triangle in row blocks: 13k x 13k at once is too much memory
        S = E[i:i + 1000] @ E.T
        r, c = np.nonzero(np.arange(len(E))[None, :] > np.arange(i, i + len(S))[:, None])  # each pair once
        s, diff = S[r, c], ids[i + r] != ids[c]
        strangers += int(diff.sum())
        tail.append(s[diff & (s > 0.3)])
        same.append(s[~diff])
    tail, same = np.sort(np.concatenate(tail))[::-1], np.concatenate(same)
    k = int(max_far * strangers)  # this many stranger pairs may score at or above the threshold
    t = float(tail[k] + 1e-3) if k < len(tail) else 0.3
    print(f"duplicate search: {len(E)} faces, {strangers:,} stranger pairs, {len(same):,} same-person pairs")
    print(f"  threshold {t:.3f}: stranger FAR {(tail >= t).sum() / strangers:.1e}, "
          f"same person found {(same >= t).mean():.1%}  (1:1 review threshold would flag "
          f"{(tail >= 0.358).sum() / strangers:.1e} of strangers)")
    return t


def rates(sims, labels, t):
    far = float(((sims >= t) & (labels == 0)).sum() / max(1, (labels == 0).sum()))  # stranger accepted
    frr = float(((sims < t) & (labels == 1)).sum() / max(1, (labels == 1).sum()))  # real person rejected
    return far, frr


def main():
    # data + preprocess (detect, quality gate, align, embed) for both official LFW splits
    s_tr, y_tr, rej_tr = similarities("train")
    s_te, y_te, rej_te = similarities("test")
    print(f"train pairs {len(s_tr)} (quality-rejected {rej_tr})  test pairs {len(s_te)} (rejected {rej_te})")

    # train: choose thresholds on the train split only
    grid = np.linspace(0, 1, 501)
    review = float(max(grid, key=lambda t: ((s_tr >= t) == y_tr).mean()))
    pass_t = float(s_tr[y_tr == 0].max() + 0.01)
    thresholds = {"pass": max(pass_t, review), "review": review, "duplicate": max(duplicate_threshold(), review)}

    # test
    print(f"{'threshold':10s} {'value':>6s} {'FAR':>7s} {'FRR':>7s}")
    for name, t in thresholds.items():
        far, frr = rates(s_te, y_te, t)
        print(f"{name:10s} {t:6.3f} {far:7.3%} {frr:7.3%}")
    bands = [decide(s, thresholds) for s in s_te]
    for band in ["pass", "review", "fail"]:
        mask = np.array([b == band for b in bands])
        print(f"  {band:6s} {mask.mean():6.1%} of test pairs, {y_te[mask].mean() if mask.any() else 0:.1%} truly same person")

    path = save_artifact({"thresholds": thresholds, "model_version": "yunet2023mar-sface2021dec",
                          "metrics": {"test_far_pass": rates(s_te, y_te, thresholds["pass"])[0]}}, "face_verification")
    print("saved", path)


if __name__ == "__main__":
    main()
