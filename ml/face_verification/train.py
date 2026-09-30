"""Face verification pipeline: data -> preprocess -> train/test split -> train -> test -> save.

The face models are pretrained, so "training" = picking the similarity thresholds.
Data: LFW pairs (same / different person), downloaded by scikit-learn (~230 MB, once).
  pass   threshold: no false matches among the train "different person" pairs
  review threshold: best overall accuracy on train
LFW is mostly Western celebrities: recalibrate on consented Indian selfie/profile pairs
and report error rates per skin-tone group before trusting these numbers (plan §7.2).
Run from ml/:  python -m face_verification.train
"""
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
    thresholds = {"pass": max(pass_t, review), "review": review}

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
