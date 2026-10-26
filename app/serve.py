# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 MatrixArkAI
"""Builds the system once at startup, then serves the portal and a small JSON API.

Every stage is timed, both at build (printed to the container log) and per request (returned
to the portal), because "which stage costs what" is the question an ads pipeline lives or
dies on.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import random
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import numpy as np

from app import adsys
from app.adsys import (
    N_ADS, CtrModel, GbdtModel, IVFIndex, TargetingIndex, auc, build_corpus,
    calibration_curve, eligible_bruteforce, gauc, generate_events, logloss, request_tokens,
    roc_points, true_ctr,
)

HERE = os.path.dirname(os.path.abspath(__file__))
STATE: dict = {}
TIMINGS: list = []

ARTIFACT = os.environ.get("ARTIFACT", "/srv/artifacts/indexes.pkl")


def signature() -> str:
    """What the artifact was built from.

    An index loaded from disk is only usable if the corpus it indexes is the same corpus.
    The scale knobs and the seed decide the ads; the source of the module decides how they
    are generated and indexed. A stale artifact that loads cleanly and answers wrongly is
    worse than no artifact, so the build refuses to reuse one whose signature moved.
    """
    src = open(os.path.join(HERE, "adsys.py"), "rb").read()
    key = f"{adsys.N_ADS}:{adsys.N_USERS}:{adsys.N_SESSION_USERS}:{adsys.EVENTS_PER_USER}:" \
          f"{adsys.DIM}:{adsys.SEED}:{hashlib.sha256(src).hexdigest()[:16]}"
    return key


def save_state() -> None:
    os.makedirs(os.path.dirname(ARTIFACT) or ".", exist_ok=True)
    keep = {}
    for name in ("model", "cvr"):
        m = STATE.get(name)
        if m is not None:              # cached training rows; serving never reads them
            keep[name] = (getattr(m, "_train_feats", None), getattr(m, "row_dense", None))
            m._train_feats = None
            m.row_dense = None
    try:
        tmp = ARTIFACT + ".tmp"
        with open(tmp, "wb") as fh:
            pickle.dump({"signature": signature(),
                         **{k: STATE[k] for k in
                            ("corpus", "index", "ivf", "model", "cvr", "metrics")}},
                        fh, protocol=5)
        os.replace(tmp, ARTIFACT)      # a half-written index must never be loadable
    finally:
        for name, (feats, dense) in keep.items():
            STATE[name]._train_feats = feats
            STATE[name].row_dense = dense
    print(f"[build] wrote {ARTIFACT} ({os.path.getsize(ARTIFACT)/1e6:.1f} MB)", flush=True)


def load_state() -> bool:
    if not os.path.exists(ARTIFACT):
        return False
    t = time.perf_counter()
    try:
        with open(ARTIFACT, "rb") as fh:
            blob = pickle.load(fh)
    except Exception as exc:
        print(f"[build] artifact unreadable ({type(exc).__name__}), rebuilding", flush=True)
        return False
    if blob.get("signature") != signature():
        print("[build] artifact signature does not match this build, rebuilding", flush=True)
        return False
    STATE.update({k: blob[k] for k in ("corpus", "index", "ivf", "model", "cvr", "metrics")})
    STATE["gbdt"] = None
    ms = (time.perf_counter() - t) * 1000
    STATE["metrics"]["load_ms"] = round(ms, 1)
    print(f"[build] loaded prebuilt indexes in {ms:.0f} ms "
          f"({adsys.N_ADS} ads, {STATE['index'].n_postings} postings)", flush=True)
    return True


class step:
    """Times a build stage and records it, so the portal can show where the minute went."""

    def __init__(self, name: str):
        self.name = name

    def __enter__(self):
        self.t = time.perf_counter()
        print(f"  [build] {self.name} ...", flush=True)
        return self

    def __exit__(self, *exc):
        ms = (time.perf_counter() - self.t) * 1000
        TIMINGS.append({"stage": self.name, "ms": round(ms, 1)})
        print(f"  [build] {self.name}: {ms:.0f} ms", flush=True)


# ---------------------------------------------------------------------------
# serving path
# ---------------------------------------------------------------------------
def recommend(uid: int, n_slots: int = 5, nprobe: int = 8, ann_k: int = 300,
              strategy: str = "filtered_ann") -> dict:
    corpus = STATE["corpus"]; index = STATE["index"]; ivf = STATE["ivf"]; model = STATE["model"]
    stages, t0 = [], time.perf_counter()

    def mark(name: str, n: int):
        nonlocal t0
        now = time.perf_counter()
        stages.append({"stage": name, "n": n, "ms": round((now - t0) * 1000, 3)})
        t0 = now

    toks = request_tokens(corpus, uid)
    mark("parse request", len(toks))

    elig_ids = index.eligible_ids(toks)
    mark("eligible (inverted index)", len(elig_ids))
    q = corpus.users["vec"][uid]

    if strategy == "search_then_filter":
        # Retrieve globally, then drop what is not targetable. Cheap, and it is the one
        # that loses ads when targeting is narrow -- the portal plots exactly that.
        ids, _ = ivf.search(q, k=ann_k, nprobe=nprobe)
        keep = np.isin(ids, elig_ids)
        cand = ids[keep]
    elif strategy == "exact_in_eligible":
        cand, _ = ivf.exact(q, k=ann_k, allowed_ids=elig_ids)
    else:
        allowed = np.zeros(N_ADS, dtype=bool)
        allowed[elig_ids] = True
        cand, _ = ivf.search(q, k=ann_k, nprobe=nprobe, allowed=allowed)
    mark(f"ANN candidates ({strategy})", len(cand))

    cand = cand[:ann_k]
    gbdt = STATE.get("gbdt")
    if gbdt is not None and len(cand):
        probs = gbdt.predict_batch(corpus, uid, cand)     # one batched call, not a loop
    else:
        probs = np.array([model.predict(*model.features(corpus, uid, int(a))) for a in cand])
    mark(f"pCTR scoring ({'GBDT' if gbdt is not None else 'linear'})", len(cand))

    # pCVR is only needed for the conversion-objective campaigns, so it is computed for them
    # and not for the rest. A CPC bid is per click and an oCPM bid is per conversion -- two
    # different units, which only become comparable once both are turned into expected
    # revenue per thousand impressions.
    cvr = STATE.get("cvr")
    objective = corpus.ads["objective"]
    ocpm_idx = [i for i in range(len(cand)) if objective[int(cand[i])] == "ocpm"]
    pcvr = np.zeros(len(cand), dtype=np.float64)
    if cvr is not None and ocpm_idx:
        for i in ocpm_idx:
            ad = int(cand[i])
            pcvr[i] = cvr.predict(*cvr.features(corpus, uid, ad))
    mark("pCVR scoring (conversion campaigns)", len(ocpm_idx))

    bid_cpc = corpus.ads["bid_cpc"]
    bid_cpa = corpus.ads["bid_cpa"]
    scored = []
    for i in range(len(cand)):
        ad = int(cand[i])
        p = float(probs[i])
        if objective[ad] == "ocpm":
            bid, ecpm = float(bid_cpa[ad]), float(bid_cpa[ad]) * p * pcvr[i] * 10.0
        else:
            bid, ecpm = float(bid_cpc[ad]), float(bid_cpc[ad]) * p * 10.0
        scored.append((ad, p, bid, ecpm, float(pcvr[i]), objective[ad]))
    scored.sort(key=lambda r: -r[3])

    # Second price: the winner pays what it needed to beat the next eCPM, not its own bid.
    reserve_ecpm = 2.0
    winners = []
    for rank, (ad, p, bid, ecpm, pc, obj) in enumerate(scored[:n_slots]):
        nxt = scored[rank + 1][3] if rank + 1 < len(scored) else reserve_ecpm
        need = max(nxt, reserve_ecpm) / 10.0          # the eCPM this ad had to beat, per imp
        # Divide back out by whatever the advertiser is charged for: a click for CPC, a
        # conversion for oCPM. Same second-price rule, different denominator.
        per = max(p, 1e-6) if obj == "cpc" else max(p * pc, 1e-9)
        price = min(need / per, bid)
        winners.append({
            "ad": ad, "rank": rank + 1,
            "objective": obj,
            "category": corpus.ads["category"][ad],
            "advertiser": corpus.ads["advertiser"][ad],
            "p_ctr": round(p, 5),
            "p_cvr": round(pc, 5) if obj == "ocpm" else None,
            "bid_cpc": round(bid, 2),
            "ecpm": round(ecpm, 2),
            "price_cpc": round(price, 2),
            "clause": index.matched_clause(toks, ad),
        })
    mark("auction (GSP)", len(winners))

    u = corpus.users
    return {
        "user": {
            "uid": uid, "country": u["country"][uid], "city": u["city"][uid],
            "age": u["age"][uid], "gender": u["gender"][uid], "os": u["os"][uid],
            "lang": u["lang"][uid], "interests": u["interests"][uid],
        },
        "stages": stages,
        "total_ms": round(sum(s["ms"] for s in stages), 3),
        "eligible_count": int(len(elig_ids)),
        "selectivity": round(len(elig_ids) / N_ADS, 5),
        "winners": winners,
        "revenue_cpm": round(sum(
            w["price_cpc"] * w["p_ctr"] * (w["p_cvr"] if w["objective"] == "ocpm" else 1.0) * 10.0
            for w in winners), 2),
        "ocpm_winners": sum(1 for w in winners if w["objective"] == "ocpm"),
    }


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------
def build() -> None:
    if os.environ.get("REBUILD") != "1" and load_state():
        return
    t_all = time.perf_counter()
    print("[build] starting", flush=True)

    with step("generate corpus (users, ads, DNF targeting)"):
        corpus = build_corpus()
    with step("build inverted targeting index"):
        index = TargetingIndex(corpus)
    with step("build IVF ANN index"):
        ivf = IVFIndex(corpus.ads["vec"], nlist=64)
    with step("simulate impressions and clicks"):
        ev = generate_events(corpus, index)

    n = len(ev["click"])
    n_train = int(n * 0.70)
    n_cal = int(n * 0.85)

    train_rows = np.arange(n_train)
    cal_rows = np.arange(n_train, n_cal)

    with step("fit counting features (time-shifted)"):
        model = CtrModel()
        model.fit_counts(corpus, ev, train_rows, ev["click"][train_rows])
    with step("train pCTR (logistic regression)"):
        # L2 is implemented and swept (`python -m app.sweep_l2`) and the measurement says
        # zero. From 1e-3 to 1.0 it shrinks the weights 971 -> 204 and costs AUC the whole
        # way (0.5854 -> 0.5562); nothing in the range buys anything back. The overfitting
        # it would have cured was removed earlier by cutting advertiser cardinality, and
        # Adagrad plus two epochs covers the rest. Kept switchable, set to what won.
        model.fit(corpus, ev, train_rows, ev["click"][train_rows], epochs=2,
                  l2=float(os.environ.get("L2", "0.0")))
    with step("calibrate pCTR (Platt)"):
        model.calibrate(corpus, ev, cal_rows, ev["click"][cal_rows])

    # pCVR trains on CLICKED rows only -- a population this system's own ranker chose, which
    # is the selection bias that makes conversion models harder than click models. The counts
    # it builds are conversion rates among clicks, not among impressions.
    with step("train pCVR (clicked rows only)"):
        cvr_train = train_rows[ev["click"][train_rows] == 1]
        cvr_cal = cal_rows[ev["click"][cal_rows] == 1]
        cvr = CtrModel()
        cvr.fit_counts(corpus, ev, cvr_train, ev["convert"][cvr_train])
        cvr.fit(corpus, ev, cvr_train, ev["convert"][cvr_train], epochs=2)
        cvr.calibrate(corpus, ev, cvr_cal, ev["convert"][cvr_cal])
        print(f"  [build] pCVR fitted on {len(cvr_train)} clicks "
              f"({100*len(cvr_train)/max(1,n_train):.1f}% of train rows)", flush=True)

    # Off by default: it loses to the linear model (see gbdt_note) and costs minutes of
    # startup building its feature matrix row by row. Kept, and switchable, because the
    # result is worth being able to reproduce: ENABLE_GBDT=1.
    gbdt = None
    if os.environ.get("ENABLE_GBDT") == "1":
        with step("train gradient-boosted trees"):
            try:
                gbdt = GbdtModel(model)
                gbdt.fit(corpus, ev, n_train, n_cal)
                print(f"  [build] gbdt best iteration: {gbdt.best_iter}", flush=True)
            except Exception as exc:             # the linear model still serves without it
                print(f"  [build] gbdt unavailable ({type(exc).__name__}: {exc})", flush=True)
                gbdt = None

    STATE.update(corpus=corpus, index=index, ivf=ivf, model=model, ev=ev, gbdt=gbdt, cvr=cvr)

    with step("evaluate on held-out"):
        y = ev["click"][n_cal:n].astype(np.int64)
        p_true = ev["p_true"][n_cal:n].astype(np.float64)
        p_lin = np.array([model.predict(*model.features(corpus, int(ev["user"][i]), int(ev["ad"][i])))
                          for i in range(n_cal, n)])
        best, best_name = p_lin, "linear"
        gb = None
        if gbdt is not None:
            X = gbdt.matrix(corpus, ev["user"][n_cal:n], ev["ad"][n_cal:n])
            z = gbdt.platt_a * gbdt._raw(X) + gbdt.platt_b
            p_gb = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
            gcurve, gece = calibration_curve(y, p_gb)
            gb = {
                "auc": round(auc(y, p_gb), 4), "logloss": round(logloss(y, p_gb), 5),
                "ece": round(gece, 5), "mean_pred": round(float(p_gb.mean()), 5),
                "calibration": gcurve, "trees": gbdt.best_iter,
            }
            # Serve whichever actually wins on held-out, rather than whichever is newer.
            # The GBDT does not win here, and the reason is in `gbdt_note` below.
            if gb["auc"] > auc(y, p_lin):
                best, best_name = p_gb, "GBDT"

        curve, ece = calibration_curve(y, best)
        model_metrics = {
            "n_events": int(n), "n_test": int(len(y)),
            "base_ctr": round(float(y.mean()), 5),
            "serving_model": best_name,
            "auc": round(auc(y, best), 4),
            "auc_ceiling": round(auc(y, p_true), 4),
            "logloss": round(logloss(y, best), 5),
            "logloss_ceiling": round(logloss(y, p_true), 5),
            "ece": round(ece, 5),
            "mean_pred": round(float(best.mean()), 5),
            "calibration": curve,
            # Pooled AUC credits ordering across requests, which no ranker is ever asked for.
            # GAUC asks the question the ranker answers: within one request.
            "gauc": gauc(y, best, ev["user"][n_cal:n]),
            "gauc_ceiling": gauc(y, p_true, ev["user"][n_cal:n])["gauc"],
            "roc": roc_points(y, best),
            "roc_ceiling": roc_points(y, p_true),
            "linear": {
                "auc": round(auc(y, p_lin), 4), "logloss": round(logloss(y, p_lin), 5),
                "ece": round(calibration_curve(y, p_lin)[1], 5),
            },
            "gbdt": gb,
            "cvr": cvr_metrics(corpus, ev, cvr, n_cal, n),
            "gbdt_note": (
                "Trees lose here, and the stopping iteration says why: the counting features "
                "are fitted on the same rows the model trains on, so ad_ctr partially encodes "
                "that row's own click. A tree splits on it immediately and the split does not "
                "survive to held-out, which is why early stopping fires after a handful of "
                "rounds. The fix is out-of-fold or time-shifted counts -- compute each row's "
                "ad_ctr from a strictly earlier slice -- not more trees or more depth."
            ),
        }

    # The index is the one component that must be EXACT, so it is checked against the
    # definition rather than trusted. An approximately-correct eligibility set is a
    # billing problem, not a quality one.
    with step("verify index == brute force (exactness guard)"):
        rng = random.Random(11)
        checked, mismatches = 0, []
        for uid in rng.sample(range(adsys.N_USERS), 15):
            toks = request_tokens(corpus, uid)
            a = index.eligible(toks)
            b = eligible_bruteforce(corpus, toks)
            checked += 1
            if a != b:
                mismatches.append({"uid": uid, "only_index": len(a - b), "only_brute": len(b - a)})
        guard = {"users_checked": checked, "mismatches": len(mismatches), "detail": mismatches[:3]}
        print(f"  [build] exactness guard: {checked} users, {len(mismatches)} mismatches", flush=True)

    with step("measure retrieval (selectivity, recall cliff, posting lists)"):
        retrieval = measure_retrieval(corpus, index, ivf)

    STATE["metrics"] = {
        "model": model_metrics,
        "guard": guard,
        "retrieval": retrieval,
        "corpus": {
            "ads": N_ADS, "users": adsys.N_USERS,
            "conjunctions": len(corpus.conjunctions),
            "postings": index.n_postings,
            "distinct_keys": len(index.postings),
            "catch_all": len(index.catch_all),
        },
        "build_timings": TIMINGS,
        "build_total_ms": round((time.perf_counter() - t_all) * 1000, 1),
    }
    print(f"[build] done in {(time.perf_counter() - t_all):.1f}s", flush=True)
    try:
        save_state()
    except Exception as exc:          # serving works without the artifact; only speed is lost
        print(f"[build] could not persist indexes ({type(exc).__name__}: {exc})", flush=True)


def cvr_metrics(corpus, ev, cvr, lo: int, hi: int) -> dict:
    """Evaluated on CLICKED held-out rows, because that is the only place the label exists.

    The positive rate here is conversions among clicks, not among impressions, so these
    numbers are not comparable to the pCTR ones and are reported separately on purpose.
    """
    rows = np.arange(lo, hi)
    rows = rows[ev["click"][rows] == 1]
    if cvr is None or len(rows) < 200:
        return {"n_test": int(len(rows)), "note": "too few clicked rows to evaluate"}
    y = ev["convert"][rows].astype(np.int64)
    p = np.array([cvr.predict(*cvr.features(corpus, int(ev["user"][n]), int(ev["ad"][n])))
                  for n in rows])
    p_true = ev["pcvr_true"][rows].astype(np.float64)
    curve, ece = calibration_curve(y, p)
    return {
        "n_test": int(len(rows)),
        "base_cvr": round(float(y.mean()), 5),
        "auc": round(auc(y, p), 4),
        "auc_ceiling": round(auc(y, p_true), 4),
        "gauc": gauc(y, p, ev["user"][rows]),
        "logloss": round(logloss(y, p), 5),
        "ece": round(ece, 5),
        "mean_pred": round(float(p.mean()), 5),
        "calibration": curve,
    }


def measure_retrieval(corpus, index, ivf) -> dict:
    rng = random.Random(3)
    sample = rng.sample(range(adsys.N_USERS), 90)

    lens = sorted(len(v) for v in index.postings.values())
    longest = sorted(((k[0] + "=" + k[1], len(v)) for k, v in index.postings.items()),
                     key=lambda r: -r[1])[:10]

    rows = []
    for uid in sample:
        toks = request_tokens(corpus, uid)
        elig = index.eligible_ids(toks)
        if len(elig) < 10:
            continue
        q = corpus.users["vec"][uid]
        truth, _ = ivf.exact(q, k=10, allowed_ids=elig)
        truth_set = set(int(x) for x in truth)

        allowed = np.zeros(N_ADS, dtype=bool)
        allowed[elig] = True

        # Retrieve globally, then drop the untargetable. The membership test is a single
        # vectorised lookup -- building the eligible set per candidate makes this O(n^2),
        # which is how this measurement first took 154 seconds.
        g_ids, _ = ivf.search(q, k=300, nprobe=8)
        stf = g_ids[allowed[g_ids]][:10] if len(g_ids) else g_ids

        f_ids, _ = ivf.search(q, k=10, nprobe=8, allowed=allowed)

        rows.append({
            "sel": len(elig) / N_ADS,
            "stf": len(truth_set & set(int(x) for x in stf)) / len(truth_set),
            "fann": len(truth_set & set(int(x) for x in f_ids)) / len(truth_set),
        })

    buckets = [(0, 0.001), (0.001, 0.005), (0.005, 0.02), (0.02, 0.08), (0.08, 1.01)]
    cliff = []
    for lo, hi in buckets:
        sel = [r for r in rows if lo <= r["sel"] < hi]
        if not sel:
            continue
        cliff.append({
            "bucket": f"{lo*100:g}-{hi*100:g}%",
            "n": len(sel),
            "search_then_filter": round(sum(r["stf"] for r in sel) / len(sel), 3),
            "filtered_ann": round(sum(r["fann"] for r in sel) / len(sel), 3),
        })

    # The headline measurement. A global top-K holds about K * selectivity eligible ads, so
    # retrieve-then-filter needs K to grow as 1/selectivity to keep the same recall, while
    # filtering INSIDE the search does not depend on K at all. Sweeping K shows that
    # relationship directly, which is better than hunting for a corpus narrow enough to
    # make the first strategy collapse on its own.
    k_sweep = []
    for K in (25, 50, 100, 300, 1000, 3000):
        stf, fann = [], []
        for uid in sample[:50]:
            toks = request_tokens(corpus, uid)
            elig = index.eligible_ids(toks)
            if len(elig) < 10:
                continue
            q = corpus.users["vec"][uid]
            truth, _ = ivf.exact(q, k=10, allowed_ids=elig)
            tset = set(int(x) for x in truth)
            allowed = np.zeros(N_ADS, dtype=bool)
            allowed[elig] = True
            g_ids, _ = ivf.search(q, k=K, nprobe=8)
            got = g_ids[allowed[g_ids]][:10] if len(g_ids) else g_ids
            stf.append(len(tset & set(int(x) for x in got)) / len(tset))
            f_ids, _ = ivf.search(q, k=10, nprobe=8, allowed=allowed)
            fann.append(len(tset & set(int(x) for x in f_ids)) / len(tset))
        if stf:
            k_sweep.append({
                "k": K,
                "search_then_filter": round(sum(stf) / len(stf), 3),
                "filtered_ann": round(sum(fann) / len(fann), 3),
                "cost_vecs_scored": K,
            })

    nprobe_curve = []
    for npb in (1, 2, 4, 8, 16, 32, 64):
        rec, ms = [], []
        for uid in sample[:25]:
            q = corpus.users["vec"][uid]
            t = time.perf_counter()
            ids, _ = ivf.search(q, k=50, nprobe=npb)
            ms.append((time.perf_counter() - t) * 1000)
            ex, _ = ivf.exact(q, k=50)
            rec.append(len(set(int(x) for x in ids) & set(int(x) for x in ex)) / 50)
        nprobe_curve.append({"nprobe": npb,
                             "recall_at_50": round(sum(rec) / len(rec), 3),
                             "ms": round(sum(ms) / len(ms), 3)})

    sels = sorted(r["sel"] for r in rows)
    return {
        "posting_len": {
            "p50": lens[len(lens) // 2], "p90": lens[int(len(lens) * 0.9)],
            "max": lens[-1], "mean": round(sum(lens) / len(lens), 1),
        },
        "longest_lists": [{"key": k, "len": n} for k, n in longest],
        "selectivity": {
            "p10": round(sels[int(len(sels) * 0.10)], 5),
            "p50": round(sels[len(sels) // 2], 5),
            "p90": round(sels[int(len(sels) * 0.90)], 5),
        },
        "recall_cliff": cliff,
        "k_sweep": k_sweep,
        "nprobe_curve": nprobe_curve,
    }


# ---------------------------------------------------------------------------
# http
# ---------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200):
        self._send(code, json.dumps(obj).encode(), "application/json; charset=utf-8")

    def do_GET(self):
        p = urlparse(self.path)
        qs = parse_qs(p.query)
        try:
            if p.path in ("/", "/index.html"):
                with open(os.path.join(HERE, "portal.html"), "rb") as fh:
                    self._send(200, fh.read(), "text/html; charset=utf-8")
            elif p.path == "/api/metrics":
                self._json(STATE["metrics"])
            elif p.path == "/api/users":
                rng = random.Random(int(qs.get("seed", ["1"])[0]))
                uids = rng.sample(range(adsys.N_USERS), 12)
                u = STATE["corpus"].users
                self._json([{"uid": i, "country": u["country"][i], "age": u["age"][i],
                             "os": u["os"][i], "interests": u["interests"][i]} for i in uids])
            elif p.path == "/api/recommend":
                uid = int(qs.get("uid", ["0"])[0]) % adsys.N_USERS
                self._json(recommend(
                    uid,
                    nprobe=int(qs.get("nprobe", ["8"])[0]),
                    strategy=qs.get("strategy", ["filtered_ann"])[0],
                ))
            else:
                self._json({"error": "not found"}, 404)
        except Exception as exc:                      # a demo that 500s silently is worse
            self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)

    def log_message(self, *a):
        pass


def main():
    build()
    port = int(os.environ.get("PORT", "8080"))
    print(f"[serve] portal on http://localhost:{port}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()


if __name__ == "__main__":
    main()
