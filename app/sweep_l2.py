# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 MatrixArkAI
"""Pick the L2 strength by measuring it, on one corpus, with everything else held fixed.

Builds the corpus, index and event log ONCE, then trains a fresh model per L2 value. The
arms differ in exactly one thing, which is the only way the comparison means anything.

    python -m app.sweep_l2
"""

import time

import numpy as np

from app.adsys import (
    CtrModel, IVFIndex, TargetingIndex, auc, build_corpus, calibration_curve, gauc,
    generate_events, logloss,
)

# The first range (1e-7 .. 1e-4) moved nothing: AUC identical to four decimals and the
# weight mass from 971.0 to 970.9. With Adagrad's `+1` in the denominator, `l2 * w` is
# negligible beside an error of order 0.1, so the penalty has to be far larger before it is
# a penalty at all. A sweep that does not change the thing it varies measures nothing.
VALUES = [0.0, 1e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]


def main() -> None:
    t = time.perf_counter()
    corpus = build_corpus()
    index = TargetingIndex(corpus)
    ev = generate_events(corpus, index)
    n = len(ev["click"])
    n_train, n_cal = int(n * 0.70), int(n * 0.85)
    train_rows = np.arange(n_train)
    cal_rows = np.arange(n_train, n_cal)
    test_rows = np.arange(n_cal, n)
    y = ev["click"][test_rows].astype(np.int64)
    groups = ev["user"][test_rows]
    print(f"corpus + log ready in {time.perf_counter()-t:.0f}s; "
          f"train {n_train}, test {len(test_rows)}\n")

    print(f"{'L2':>8}  {'AUC':>7}  {'GAUC':>7}  {'logloss':>8}  {'ECE':>7}  {'|w|':>9}  {'s':>4}")
    print("-" * 60)
    for l2 in VALUES:
        t0 = time.perf_counter()
        m = CtrModel()
        m.fit_counts(corpus, ev, train_rows, ev["click"][train_rows])
        m.fit(corpus, ev, train_rows, ev["click"][train_rows], epochs=2, l2=l2)
        m.calibrate(corpus, ev, cal_rows, ev["click"][cal_rows])
        p = np.array([m.predict(*m.features(corpus, int(ev["user"][i]), int(ev["ad"][i])))
                      for i in test_rows])
        g = gauc(y, p, groups)
        print(f"{l2:>8.0e}  {auc(y, p):>7.4f}  {g['gauc']:>7.4f}  {logloss(y, p):>8.5f}  "
              f"{calibration_curve(y, p)[1]:>7.5f}  {float(np.abs(m.w).sum()):>9.1f}  "
              f"{time.perf_counter()-t0:>4.0f}")
    print(f"\nGAUC over {g['groups_used']} requests; "
          f"{g['groups_skipped']} skipped with no click contrast "
          f"({g['rows_skipped']} of {len(test_rows)} rows)")


if __name__ == "__main__":
    main()
