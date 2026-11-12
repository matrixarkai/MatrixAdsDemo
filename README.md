# MatrixAdsDemo — inverted targeting, ANN retrieval, calibrated pCTR, auction

**Open source under Apache-2.0, for anyone in the world to use, fork, modify and contribute to.**
Issues and pull requests from outside are welcome — see
[the contribution guide](https://github.com/matrixarkai/MatrixAdsDemo/blob/main/CONTRIBUTING.md).

## Project and community

**On this page:** [Run the demo](#overview) | [Index lifecycle](#the-indexes-are-a-build-artifact) |
[Pipeline](#the-pipeline) | [Portal](#what-the-portal-shows) |
[Conversion objective](#pcvr-and-the-conversion-objective) |
[Optional GBDT](#the-gbdt-attempt-and-why-it-is-off-by-default) |
[Limitations](#deliberately-out-of-scope).

- [Public source](https://github.com/matrixarkai/MatrixAdsDemo),
  [issues](https://github.com/matrixarkai/MatrixAdsDemo/issues), and
  [pull requests](https://github.com/matrixarkai/MatrixAdsDemo/pulls).
- [Apache-2.0 license](https://github.com/matrixarkai/MatrixAdsDemo/blob/main/LICENSE)
  and [notices](https://github.com/matrixarkai/MatrixAdsDemo/blob/main/NOTICE).
- [Code of conduct](https://github.com/matrixarkai/MatrixAdsDemo/blob/main/CODE_OF_CONDUCT.md)
  and [security policy](https://github.com/matrixarkai/MatrixAdsDemo/blob/main/SECURITY.md).

## Overview

A short-video style ads recommender, small enough to read end to end. The point is the shape of the
pipeline and the measurements beside it, not scale: exact eligibility, approximate retrieval,
calibrated prediction, second-price auction — each one measured against a baseline or a ceiling
rather than quoted on its own.

Only dependency on the serving path is numpy. The ANN index and the models are written out by hand,
because the interesting parts of both are the parts a library hides.

```bash
docker build -t ads-demo .          # builds the indexes INTO the image (~90s)
docker run -d --name ads -p 127.0.0.1:8080:8080 ads-demo
# http://localhost:8080   (serving ~4s after start)
```

The host binding above keeps this demonstration local. It does not configure
authentication, TLS, or production traffic handling. After startup, inspect the
same generated corpus through the portal or the read-only JSON endpoints:

```bash
curl -fsS http://127.0.0.1:8080/api/metrics
curl -fsS 'http://127.0.0.1:8080/api/users?seed=1'
curl -fsS 'http://127.0.0.1:8080/api/recommend?uid=0'
curl -fsS 'http://127.0.0.1:8080/api/recommend?uid=0&nprobe=16&strategy=filtered_ann'
docker stop ads
docker rm ads
```

Use a generated `uid` from `/api/users` when comparing requests. The handler
maps numeric IDs into the generated corpus; it is not an external user-identity
lookup. Choose the retrieval strategy explicitly:

| `strategy` | Candidate selection |
|---|---|
| `filtered_ann` | Default: constrain approximate search to eligible ads. |
| `search_then_filter` | Search the global ANN index, then remove ineligible candidates. |
| `exact_in_eligible` | Score the eligible set exactly as a retrieval control. |

`nprobe` defaults to 8 and changes ANN search breadth, not the corpus or model.
It does not turn exact eligible-set scoring into an ANN search. Compare the
same user, built index, and scoring model when changing retrieval settings;
candidate recall and end-to-end auction output are different measurements.
The HTTP handler and parameter defaults are in [the server](app/serve.py).

Stopping or removing this container does not turn its generated measurements
into a durable feature service. For online feature storage, use
[TemporalStore](https://github.com/matrixarkai/TemporalStore) and its Feast
integration; the demo does not provision those services automatically.

## The indexes are a build artifact

`python -m app.build_index` derives the corpus, the inverted targeting index, the IVF ANN
index and the CTR model, and writes them to `/srv/artifacts/indexes.pkl` (18.5 MB). The
Dockerfile runs it, so the image ships with the indexes and a container starts by loading
them — under a second — rather than deriving them.

That is not only a speed point. Deriving them at startup cost 48s on an idle machine and over
three minutes on a busy one, and it was paid again on every restart: a Docker restart took the
demo down and the next start spent minutes rebuilding indexes that had not changed.

The artifact records a signature over the scale knobs, the seed and a hash of `adsys.py`. If
any of those move, the loader rejects it and rebuilds, because an index that loads cleanly and
answers for a different corpus is worse than no index at all. `REBUILD=1` forces a rebuild.

## The pipeline

```
request ─► eligible         inverted index over DNF targeting      EXACT
        ─► candidates       IVF ANN over ad vectors                APPROXIMATE, recall measured
        ─► pCTR             hashed logistic regression + Platt     calibrated, ECE reported
        ─► eCPM = bid×pCTR
        ─► auction          second price
```

The asymmetry on the right is the main design claim. Showing an ad outside its targeting is a
billing and compliance failure, so eligibility is checked against brute-force predicate
evaluation at build time and the portal reports the result. "Relevant" has no exact answer, so
ANN recall is a dial with a measured curve.

## What the portal shows

Portal timings describe this generated corpus and the current retrieval settings.
Keep the built image, user IDs, strategy, and search breadth fixed when comparing
runs. Inspect recall and funnel counts with latency: scoring fewer eligible
candidates can be faster while changing auction results. These measurements do
not establish the throughput or durability of a distributed feature store.

| tab | what is on it |
| --- | --- |
| Serve a request | the live funnel for one user: counts and milliseconds per stage, the auction, and the targeting clause that made each winner eligible |
| Retrieval | the K-sweep below, the ANN recall/latency curve, posting-list statistics |
| Model | AUC against the generator's own ceiling, log loss, ECE, calibration curve |
| Design notes | the six decisions and what was deliberately left out |

## Why filtering happens inside the ANN search

A global top-K contains roughly `K × selectivity` eligible ads, so retrieve-then-filter needs K
to grow as `1/selectivity` to hold recall. Filtering during traversal does not depend on K.
Measured against exact search restricted to the eligible set:

| K | search then filter | filtered ANN |
| --: | --: | --: |
| 25 | 0.342 | **0.906** |
| 50 | 0.590 | 0.906 |
| 100 | 0.802 | 0.906 |
| 300 | 0.906 | 0.906 |

Same answer, thirty times fewer vectors scored.

## Layout

- `app/adsys.py` — generator, inverted targeting index, IVF index, CTR model, metrics
- `app/serve.py` — build, the serving funnel, the measurements, the HTTP API
- `app/portal.html` — the portal (no framework, charts are inline SVG)

`PYTHONHASHSEED` is pinned in the image: the model keys weights by `hash(str)`, which Python
randomises per process, so two runs of the same image would otherwise score differently.

## Deliberately out of scope

Budget pacing and frequency capping; a two-tower encoder trained with in-batch
negatives; a small MLP; real click logs instead of a simulator; counterfactual
evaluation from logged propensities; compressed posting lists; delayed conversion
feedback.

The demo does include [pCVR and conversion pricing](#pcvr-and-the-conversion-objective)
and an [optional GBDT experiment](#the-gbdt-attempt-and-why-it-is-off-by-default).
Those paths use synthetic data, and the GBDT is disabled by default; neither is
a claim of production model quality.

## Counting features are time-shifted

Every training row's `ad_ctr` / `adv_ctr` / `cat_ctr` is a snapshot of the history **before
that row happened**, which is what serving actually has. Fitting them on the same rows the
model trains on puts a row's own click inside its own feature.

Nothing in AUC or log loss announces a leak. It showed up as a model that could not be
improved: the GBDT split on the leaked feature immediately and early stopping fired at
**iteration 3**. After the fix it runs 23 rounds and its ECE falls 0.0438 → 0.0059.

It also cost the linear model a little, which is the honest direction: **0.5976 → 0.5918**.
The earlier figure was partly leak.

K-fold out-of-fold counts are the other standard answer and use the data more efficiently,
but they let a row see its own future, which a live system never does.

## pCVR and the conversion objective

A quarter of campaigns buy conversions rather than clicks. For those the auction prices with
`bid × pCTR × pCVR`; for the rest, `bid × pCTR`. A CPA bid and a CPC bid are different units
and only become comparable as expected revenue per thousand impressions — which is the whole
reason the second model exists.

| | pCTR | pCVR |
| --- | --: | --: |
| AUC | 0.5918 | 0.8032 |
| ceiling | 0.6803 | 0.8596 |
| of available lift | 47% | **84%** |
| ECE | 0.0041 | 0.0315 |
| training rows | 140,000 | **14,402** |

pCVR trains on clicked rows only — 10.3% of the log, a population this system's own ranker
chose. That is the selection bias that makes conversion models harder, and it is why its
calibration is an order of magnitude worse than pCTR's on a tenth of the data.

It scores *better* than pCTR here because the generator makes conversion a simpler function:
category propensity dominates, and category is a feature. Real pCVR is not easier than pCTR.

**Not modelled:** delayed feedback. A conversion can arrive days after the click, so a recent
click labelled "no conversion" may only mean *not yet*. Here conversions are drawn at click
time, so labels are complete and the problem is strictly easier than production.

## The GBDT attempt, and why it is off by default

`ENABLE_GBDT=1` trains LightGBM on the same information the linear model gets. It still
**loses**: 0.555 AUC against 0.592.

I expected the count leakage to be the cause. It was not. Fixing it moved the GBDT's
calibration sharply (ECE 0.0438 → 0.0059) and its stopping point (3 → 23 rounds), and its
ranking **not at all**. The leak was real and was not the binding constraint.

The untested hypothesis for the remainder: this signal is a sum of many *weak* pairwise
interactions — every user attribute against every ad attribute — which explicit hashed crosses
represent with one parameter each, while a tree needs a leaf per conjunction. 23 trees × 63
leaves cannot cover hundreds of interaction cells. If that is right, the way to make trees win
is to hand them target-encoded crosses, at which point they are doing what the linear model
already does.

Serving picks whichever model wins on held-out AUC, so the system serves the linear one and
would switch on its own if that changed.

Two smaller findings from the same session, both measured rather than argued: crossing user
attributes against *advertiser* cost 0.018 AUC at 400 advertisers (≈9 training rows per cell)
and paid at 60; and normalising the category embedding to unit length while the advertiser term
had norm √32 let the advertiser silently dominate the ad vector, costing 0.06 AUC.
