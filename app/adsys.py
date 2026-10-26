# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 MatrixArkAI
"""An ads recommendation system, small enough to read in one sitting.

The pipeline is the point, so it is spelled out in stages rather than hidden behind a framework:

    request -> eligible (inverted targeting)   EXACT, because showing an ad that was not
                                               targeted is a billing and compliance failure
            -> candidates (ANN over ad vectors) APPROXIMATE, because "relevant" has no exact
                                               answer and recall is a tunable
            -> pCTR (hashed logistic regression + calibration)
            -> eCPM = bid * pCTR
            -> second-price auction

Only numpy is imported. The ANN index and the model are written out by hand because the
interesting parts of both are the parts a library hides.
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------------------
# scale knobs. Small enough that the whole thing builds in under a minute, large
# enough that brute force is visibly slower than the indexes.
# ---------------------------------------------------------------------------
N_ADS = 30_000
N_USERS = 15_000
N_SESSION_USERS = 2_000
EVENTS_PER_USER = 100
DIM = 32
HASH_BITS = 18
HASH_SIZE = 1 << HASH_BITS
SEED = 7

COUNTRIES = ["US", "UK", "DE", "JP", "BR", "IN", "FR", "CA"]
CITIES = [f"{c}-{i}" for c in COUNTRIES for i in range(5)]
AGES = ["13-17", "18-24", "25-34", "35-44", "45-54", "55+"]
GENDERS = ["f", "m", "u"]
OSES = ["ios", "android"]
LANGS = ["en", "de", "ja", "pt", "hi", "fr"]
INTERESTS = [
    "dance", "comedy", "beauty", "gaming", "food", "fitness", "pets", "travel",
    "diy", "music", "sports", "fashion", "tech", "cars", "finance", "study",
    "parenting", "outdoors", "art", "news", "anime", "movies", "home", "crypto",
]
CATEGORIES = [
    "ecommerce", "gaming", "finance", "beauty", "food-delivery", "travel",
    "education", "auto", "saas", "entertainment",
]

# Which interests an ad category appeals to. This is the only hand-written semantics in the
# generator, and it is what makes a recommendation inspectable: a user who likes food and
# comedy should be served food-delivery, not an arbitrary cluster that happens to be near.
CATEGORY_INTERESTS = {
    "ecommerce": ["fashion", "home", "diy"],
    "gaming": ["gaming", "anime", "tech"],
    "finance": ["finance", "crypto", "news"],
    "beauty": ["beauty", "fashion", "dance"],
    "food-delivery": ["food", "comedy", "home"],
    "travel": ["travel", "outdoors", "art"],
    "education": ["study", "parenting", "news"],
    "auto": ["cars", "sports", "tech"],
    "saas": ["tech", "finance", "study"],
    "entertainment": ["movies", "music", "comedy", "anime"],
}

# A user is single-valued on these and multi-valued on `interest`. The distinction
# matters in the index: a multi-valued attribute can hit one predicate several times,
# which is why satisfaction is tracked as a bitmask and not a counter.
SINGLE_ATTRS = ["country", "city", "age", "gender", "os", "lang"]


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
@dataclass
class Conjunction:
    """One AND-clause of an ad's targeting, pre-compiled for the index.

    `full_mask` has one bit per positive predicate. A conjunction fires when every bit is
    set by the request. Arity is capped at 8 so the mask is a byte.
    """

    ad: int
    full_mask: int
    arity: int
    positives: list = field(default_factory=list)   # [(attr, frozenset(values))]
    negatives: list = field(default_factory=list)   # [(attr, frozenset(values))]


@dataclass
class Corpus:
    users: dict
    ads: dict
    conjunctions: list


# Advertisers do not target uniformly -- they buy the big segments. That skew is what
# makes request selectivity VARY: a user in a rare country with niche interests matches
# few clauses, and those are exactly the requests that break global retrieval. With
# uniform targeting every user came out eligible for the same 12.6% of the corpus.
def _zipf_weights(n: int, s: float = 1.1) -> list:
    w = [1.0 / ((i + 1) ** s) for i in range(n)]
    t = sum(w)
    return [x / t for x in w]


COUNTRY_W = _zipf_weights(len(COUNTRIES), 1.25)
INTEREST_W = _zipf_weights(len(INTERESTS), 1.0)
AGE_W = _zipf_weights(len(AGES), 0.6)


def _pick_values(rng: random.Random, attr: str, pool: list) -> list:
    """A targeting predicate names one value or a small set of them, drawn by popularity."""
    k = min(rng.choice([1, 1, 1, 2, 3]), len(pool))
    weights = {"country": COUNTRY_W, "interest": INTEREST_W, "age": AGE_W}.get(attr)
    if weights is None or len(weights) != len(pool):
        return rng.sample(pool, k)
    chosen: list = []
    remaining = list(zip(pool, weights))
    for _ in range(k):
        total = sum(w for _, w in remaining)
        r = rng.random() * total
        acc = 0.0
        for i, (v, w) in enumerate(remaining):
            acc += w
            if r <= acc:
                chosen.append(v)
                remaining.pop(i)
                break
    return chosen


def build_corpus(seed: int = SEED) -> Corpus:
    rng = random.Random(seed)
    nrng = np.random.default_rng(seed)

    # One latent space, shared by both sides. A user vector is built from the things a
    # profile actually says, and an ad vector from the things a creative says, so the
    # affinity that drives clicks is REACHABLE from the observable features.
    #
    # Drawing the two sides independently (the first thing I did) makes the click signal
    # unlearnable in principle: the model sat at AUC 0.52 against a 0.67 ceiling, and no
    # amount of feature work could have moved it. A generator has to be able to state
    # which observable feature carries the signal.
    emb = {k: nrng.normal(size=DIM).astype(np.float32)
           for k in INTERESTS + COUNTRIES + AGES + OSES}

    # A category sits where its audience's interests sit, so "which ads suit this user" has a
    # readable answer. With independently drawn category vectors the geometry still worked,
    # but a food-and-comedy user was served five gaming ads -- correct arithmetic over labels
    # that meant nothing, which is indistinguishable from a bug when someone asks why.
    # Scaled to the magnitude of a single draw, not to unit length. Normalising it shrank the
    # category term to 1 against an advertiser term of norm ~sqrt(DIM), so the advertiser
    # dominated the ad vector and AUC fell from 0.598 to 0.538. The mean of k draws is
    # sqrt(k) times smaller than a draw, so a mean needs rescaling before it can be added to
    # one.
    for cat, related in CATEGORY_INTERESTS.items():
        v = np.mean([emb[i] for i in related], axis=0)
        v = v / (np.linalg.norm(v) + 1e-9) * math.sqrt(DIM)
        emb[cat] = v.astype(np.float32)

    users = {
        "country": [], "city": [], "age": [], "gender": [], "os": [], "lang": [],
        "interests": [], "vec": None,
    }
    for _ in range(N_USERS):
        c = rng.choice(COUNTRIES)
        users["country"].append(c)
        users["city"].append(rng.choice([x for x in CITIES if x.startswith(c)]))
        users["age"].append(rng.choice(AGES))
        users["gender"].append(rng.choice(GENDERS))
        users["os"].append(rng.choice(OSES))
        users["lang"].append(rng.choice(LANGS))
        users["interests"].append(rng.sample(INTERESTS, rng.choice([1, 2, 2, 3, 4])))

    uv = np.zeros((N_USERS, DIM), dtype=np.float32)
    for i in range(N_USERS):
        v = np.zeros(DIM, dtype=np.float32)
        for it in users["interests"][i]:
            v += emb[it]
        v /= len(users["interests"][i])
        v += 0.35 * emb[users["country"][i]] + 0.25 * emb[users["age"][i]] + 0.15 * emb[users["os"][i]]
        v += 0.45 * nrng.normal(size=DIM).astype(np.float32)   # the part no feature can see
        uv[i] = v / (np.linalg.norm(v) + 1e-9)
    users["vec"] = uv

    # Ads. Bids are in cents-per-click and deliberately heavy-tailed: a flat bid
    # distribution makes the auction look like a ranking, which hides what GSP does.
    # Fewer, larger advertisers. At 400 the advertiser crosses had ~9 training rows per cell
    # and fitting them cost 0.018 AUC; at 60 there are enough rows per cell to estimate one.
    # The cross that carries signal is still worthless if the data cannot populate it.
    N_ADVERTISERS = 60
    adv_emb = {f"adv-{i}": nrng.normal(size=DIM).astype(np.float32) for i in range(N_ADVERTISERS)}
    # A category's own propensity to convert, so pCVR is learnable from `cat` the way pCTR is.
    cat_cvr = {c: float(nrng.normal(0.0, 1.0)) for c in CATEGORIES}
    ads = {
        "campaign": [], "advertiser": [], "category": [], "bid_cpc": [],
        "quality": [], "vec": None, "targeting": [],
        "cvr_base": [], "objective": [], "bid_cpa": [],
    }
    conjunctions: list = []

    for ad in range(N_ADS):
        ads["advertiser"].append(f"adv-{rng.randrange(N_ADVERTISERS)}")
        ads["campaign"].append(f"camp-{rng.randrange(3000)}")
        ads["category"].append(rng.choice(CATEGORIES))
        ads["bid_cpc"].append(round(float(nrng.lognormal(mean=3.1, sigma=0.7)), 2))
        ads["quality"].append(float(nrng.normal(0.0, 1.0)))
        ads["cvr_base"].append(cat_cvr[ads["category"][-1]] + 0.4 * float(nrng.normal()))
        # A quarter of campaigns buy conversions rather than clicks. Their bid is per
        # CONVERSION, so it is a different unit and an order of magnitude larger -- which is
        # exactly why the auction cannot compare the two numbers without pCVR.
        ocpm = rng.random() < 0.25
        ads["objective"].append("ocpm" if ocpm else "cpc")
        # Scaled so the two objectives actually compete. A CPA bid buys a conversion, which
        # happens on roughly 43% of clicks, so an advertiser indifferent between the two
        # bids about bid_cpc / 0.43 -- a shade over 2x. At lognormal(6.0) the CPA bids were
        # ~10x too large and oCPM won every slot, which looks like a result and is an
        # artefact of the bid distribution.
        ads["bid_cpa"].append(round(float(nrng.lognormal(mean=3.9, sigma=0.6)), 2) if ocpm else 0.0)

        # Targeting as DNF: OR over 1..3 conjunctions. Most advertisers write one
        # clause; a few write several, which is what makes DNF the right shape.
        n_conj = rng.choice([1, 1, 1, 1, 2, 3])
        dnf = []
        for _ in range(n_conj):
            # Explicit breadth tiers, because the SPREAD is the thing that matters. A
            # distribution with no narrow clauses hides the one failure mode worth
            # demonstrating: retrieve-globally-then-filter cannot find a city-targeted ad.
            # An earlier version made a quarter of all clauses unconstrained, and every
            # user came out eligible for 37% of the corpus -- one bucket, no cliff.
            tier = rng.random()
            pos, neg = [], []
            if tier < 0.03:
                pass                                    # "all users" -- rare, as in practice
            else:
                pos.append(("country", frozenset(_pick_values(rng, "country", COUNTRIES))))
                if tier >= 0.25:
                    pos.append(("age", frozenset(_pick_values(rng, "age", AGES))))
                if tier >= 0.50:
                    pos.append(("interest", frozenset(_pick_values(rng, "interest", INTERESTS))))
                if tier >= 0.72:
                    pos.append(("os", frozenset([rng.choice(OSES)])))
                if tier >= 0.88:
                    anchor = sorted(pos[0][1])[0]
                    pool = [x for x in CITIES if x.startswith(anchor)]
                    pos.append(("city", frozenset(_pick_values(rng, "city", pool))))
                    pos.append(("gender", frozenset([rng.choice(["f", "m"])])))
            if rng.random() < 0.12:
                neg.append(("interest", frozenset(rng.sample(INTERESTS, 2))))
            if rng.random() < 0.05:
                neg.append(("country", frozenset(rng.sample(COUNTRIES, 1))))

            pos = pos[:8]
            dnf.append({
                "positives": [(a, sorted(v)) for a, v in pos],
                "negatives": [(a, sorted(v)) for a, v in neg],
            })
            conjunctions.append(Conjunction(
                ad=ad,
                full_mask=(1 << len(pos)) - 1,
                arity=len(pos),
                positives=pos,
                negatives=neg,
            ))
        ads["targeting"].append(dnf)

    av = np.zeros((N_ADS, DIM), dtype=np.float32)
    for i in range(N_ADS):
        v = emb[ads["category"][i]] + 0.45 * adv_emb[ads["advertiser"][i]]
        v += 0.55 * nrng.normal(size=DIM).astype(np.float32)
        av[i] = v / (np.linalg.norm(v) + 1e-9)
    ads["vec"] = av

    ads["cvr_base"] = np.asarray(ads["cvr_base"], dtype=np.float32)
    ads["bid_cpa"] = np.asarray(ads["bid_cpa"], dtype=np.float32)
    ads["bid_cpc"] = np.asarray(ads["bid_cpc"], dtype=np.float32)
    ads["quality"] = np.asarray(ads["quality"], dtype=np.float32)
    return Corpus(users=users, ads=ads, conjunctions=conjunctions)


def request_tokens(corpus: Corpus, uid: int) -> list:
    """The (attribute, value) pairs a request carries. Index lookups are exactly these."""
    u = corpus.users
    toks = [(a, u[a][uid]) for a in SINGLE_ATTRS]
    toks += [("interest", i) for i in u["interests"][uid]]
    return toks


# ---------------------------------------------------------------------------
# 1. inverted targeting index  (exact)
# ---------------------------------------------------------------------------
class TargetingIndex:
    """Boolean-expression index over DNF targeting.

    One posting list per (attribute, value), holding (conjunction_id, predicate_bit). A
    conjunction is eligible once the request has set all of its bits and none of its
    negative predicates matched.

    The alternative -- one posting list per ad and a scan -- is what you get if you treat
    targeting as a filter instead of a retrieval problem, and it costs O(corpus) a request.
    """

    def __init__(self, corpus: Corpus):
        self.corpus = corpus
        self.conj = corpus.conjunctions
        self.catch_all: list = []          # clauses with no positive predicate at all

        pos: dict = {}
        neg: dict = {}
        self.by_ad: dict = {}
        for cid, c in enumerate(self.conj):
            self.by_ad.setdefault(c.ad, []).append(cid)
            if c.arity == 0:
                self.catch_all.append(cid)
            for bit, (attr, values) in enumerate(c.positives):
                for v in values:
                    # (conjunction, which predicate) -- NOT (conjunction, which value).
                    # A user with several interests can hit one predicate more than once,
                    # so satisfaction counts distinct predicates, never postings.
                    pos.setdefault((attr, v), []).append(cid * 8 + bit)
            for attr, values in c.negatives:
                for v in values:
                    neg.setdefault((attr, v), []).append(cid)

        # Posting lists as arrays: a request concatenates a handful of them, so the whole
        # evaluation becomes unique + bincount instead of a Python loop per posting.
        self.postings = {k: np.asarray(v, dtype=np.int64) for k, v in pos.items()}
        self.neg_postings = {k: np.asarray(v, dtype=np.int64) for k, v in neg.items()}
        self.arity = np.asarray([c.arity for c in self.conj], dtype=np.int16)
        self.conj_ad = np.asarray([c.ad for c in self.conj], dtype=np.int64)
        self.n_postings = int(sum(len(v) for v in self.postings.values()))
        # A clause with no positive predicate targets everyone -- but it may still carry a
        # negative one, so it goes through the same disqualification pass as the rest.
        self._catch_cids = np.asarray(self.catch_all, dtype=np.int64)

    def eligible(self, toks: list) -> set:
        return set(int(a) for a in self.eligible_ids(toks))

    def eligible_ids(self, toks: list) -> np.ndarray:
        lists = [self.postings[t] for t in toks if t in self.postings]
        if lists:
            # One entry per (conjunction, predicate) actually satisfied, so a predicate hit
            # twice by a multi-valued attribute still counts once.
            hits = np.unique(np.concatenate(lists))
            cids = hits >> 3
            counts = np.bincount(cids)
            touched = np.flatnonzero(counts)
            ok = touched[counts[touched] == self.arity[touched]]
        else:
            ok = np.empty(0, dtype=np.int64)

        if len(self._catch_cids):
            ok = np.concatenate([ok, self._catch_cids])

        neg = [self.neg_postings[t] for t in toks if t in self.neg_postings]
        if neg and len(ok):
            ok = ok[~np.isin(ok, np.unique(np.concatenate(neg)))]

        return np.unique(self.conj_ad[ok]) if len(ok) else np.empty(0, dtype=np.int64)

    def matched_clause(self, toks: list, ad: int):
        """Which clause let this ad through -- the portal shows it, so 'why' is answerable.

        Keyed by ad. Scanning every conjunction instead cost 29 ms a request, which made
        explaining the answer more expensive than computing it.
        """
        tokset = set(toks)
        for cid in self.by_ad.get(ad, ()):
            c = self.conj[cid]
            if _clause_matches(c, tokset):
                return {
                    "positives": [[a, sorted(v)] for a, v in c.positives],
                    "negatives": [[a, sorted(v)] for a, v in c.negatives],
                }
        return None


def _clause_matches(c: Conjunction, tokset: set) -> bool:
    for attr, values in c.positives:
        if not any((attr, v) in tokset for v in values):
            return False
    for attr, values in c.negatives:
        if any((attr, v) in tokset for v in values):
            return False
    return True


def eligible_bruteforce(corpus: Corpus, toks: list) -> set:
    """The definition, evaluated directly. The index must agree with this exactly."""
    tokset = set(toks)
    out = set()
    for c in corpus.conjunctions:
        if c.ad in out:
            continue
        if _clause_matches(c, tokset):
            out.add(c.ad)
    return out


# ---------------------------------------------------------------------------
# 2. ANN retrieval  (approximate, and the error is measured)
# ---------------------------------------------------------------------------
class IVFIndex:
    """IVF-Flat: k-means coarse quantizer plus inverted lists of ad ids.

    The same idea as the targeting index -- a posting list keyed by something cheap to
    compute -- which is why both live in this file. Vectors are unit-norm, so inner
    product ranks the same as cosine.
    """

    def __init__(self, vecs: np.ndarray, nlist: int = 64, iters: int = 12, seed: int = SEED):
        rng = np.random.default_rng(seed)
        n = vecs.shape[0]
        self.vecs = vecs
        self.nlist = nlist
        centroids = vecs[rng.choice(n, size=nlist, replace=False)].copy()
        for _ in range(iters):
            assign = np.argmax(vecs @ centroids.T, axis=1)
            for j in range(nlist):
                members = vecs[assign == j]
                if len(members):
                    c = members.mean(axis=0)
                    centroids[j] = c / (np.linalg.norm(c) + 1e-9)
        self.centroids = centroids
        self.assign = np.argmax(vecs @ centroids.T, axis=1)
        self.lists = [np.where(self.assign == j)[0] for j in range(nlist)]

    def search(self, q: np.ndarray, k: int = 200, nprobe: int = 8, allowed: np.ndarray | None = None):
        order = np.argsort(-(self.centroids @ q))[:nprobe]
        ids = np.concatenate([self.lists[j] for j in order]) if len(order) else np.empty(0, dtype=int)
        if allowed is not None and len(ids):
            ids = ids[allowed[ids]]
        if not len(ids):
            return np.empty(0, dtype=int), np.empty(0, dtype=np.float32)
        scores = self.vecs[ids] @ q
        if len(ids) > k:
            top = np.argpartition(-scores, k)[:k]
            ids, scores = ids[top], scores[top]
        o = np.argsort(-scores)
        return ids[o], scores[o]

    def exact(self, q: np.ndarray, k: int = 200, allowed_ids: np.ndarray | None = None):
        if allowed_ids is None:
            scores = self.vecs @ q
            ids = np.arange(len(scores))
        else:
            ids = allowed_ids
            scores = self.vecs[ids] @ q
        if len(ids) > k:
            top = np.argpartition(-scores, k)[:k]
            ids, scores = ids[top], scores[top]
        o = np.argsort(-scores)
        return ids[o], scores[o]


# ---------------------------------------------------------------------------
# 3. click model and event log
# ---------------------------------------------------------------------------
def true_ctr(corpus: Corpus, uid: int, ads_idx: np.ndarray) -> np.ndarray:
    """The generative truth. Having it means the model's AUC can be read against a ceiling
    instead of against nothing."""
    u = corpus.users["vec"][uid]
    a = corpus.ads["vec"][ads_idx]
    affinity = a @ u
    q = corpus.ads["quality"][ads_idx]
    logit = -2.6 + 2.4 * affinity + 0.45 * q
    return 1.0 / (1.0 + np.exp(-logit))


def true_cvr(corpus: Corpus, uid: int, ads_idx: np.ndarray) -> np.ndarray:
    """P(convert | click). Deliberately NOT the same function as the click model.

    If conversion were driven by the same affinity as the click, pCVR would be a monotone
    transform of pCTR and a conversion model would be pointless -- you could rank by pCTR and
    be done. The interesting case, and the real one, is that what makes someone click is not
    what makes them buy: here the category's own propensity dominates and the affinity term
    is weaker and differently signed.
    """
    u = corpus.users["vec"][uid]
    a = corpus.ads["vec"][ads_idx]
    affinity = a @ u
    intent = corpus.ads["cvr_base"][ads_idx]
    logit = -1.4 + 1.9 * intent - 0.7 * affinity
    return 1.0 / (1.0 + np.exp(-logit))


def generate_events(corpus: Corpus, index: TargetingIndex, seed: int = SEED):
    """Impressions are drawn from the ELIGIBLE set, so the log carries the same selection
    bias a real one does: an ad can only be clicked where it was allowed to serve.

    Conversions are drawn only where a click happened, which is the shape a conversion log
    really has -- and the reason a pCVR model trains on a population its own ranker chose.
    """
    rng = np.random.default_rng(seed)
    prng = random.Random(seed)
    uids = prng.sample(range(N_USERS), N_SESSION_USERS)

    ev_user, ev_ad, ev_click, ev_p = [], [], [], []
    ev_conv, ev_pc = [], []
    for uid in uids:
        elig = index.eligible_ids(request_tokens(corpus, uid))
        if len(elig) == 0:
            continue
        picks = elig[rng.integers(0, len(elig), size=EVENTS_PER_USER)]
        p = true_ctr(corpus, uid, picks)
        clicks = (rng.random(len(picks)) < p).astype(np.int8)
        pc = true_cvr(corpus, uid, picks)
        convs = ((rng.random(len(picks)) < pc) & (clicks == 1)).astype(np.int8)
        ev_user.append(np.full(len(picks), uid, dtype=np.int64))
        ev_ad.append(picks)
        ev_click.append(clicks)
        ev_p.append(p.astype(np.float32))
        ev_conv.append(convs)
        ev_pc.append(pc.astype(np.float32))

    return {
        "user": np.concatenate(ev_user),
        "ad": np.concatenate(ev_ad),
        "click": np.concatenate(ev_click),
        "p_true": np.concatenate(ev_p),
        "convert": np.concatenate(ev_conv),
        "pcvr_true": np.concatenate(ev_pc),
    }


# ---------------------------------------------------------------------------
# 4. the model: hashed features + logistic regression + calibration
# ---------------------------------------------------------------------------
def _h(s: str) -> int:
    return hash(s) % HASH_SIZE


class CtrModel:
    """Hashing trick, SGD logistic regression, Platt calibration.

    Deliberately the simple choice. What makes it hold up as the ad corpus grows is not the
    loss function, it is that no parameter is per-ad: ad identity enters through hashed
    buckets and through smoothed counting features, so a new ad needs a prior rather than a
    retrain. A per-ad embedding table would need the opposite.
    """

    def __init__(self):
        self.w = np.zeros(HASH_SIZE, dtype=np.float32)
        self.b = 0.0
        self.wn = np.zeros(4, dtype=np.float32)     # weights on the dense counting tail
        self.g = np.zeros(HASH_SIZE, dtype=np.float32)     # adagrad accumulators
        self.gn = np.zeros(4, dtype=np.float32)
        self.gb = 0.0
        self.prior = 0.02
        self.ad_ctr: dict = {}
        self.adv_ctr: dict = {}
        self.cat_ctr: dict = {}
        self.platt_a = 1.0
        self.platt_b = 0.0

    # -- counting features, TIME-SHIFTED: every row sees only the history before it --
    def fit_counts(self, corpus: Corpus, ev: dict, rows, labels, alpha: float = 60.0):
        """Build the serving counts, and a per-training-row snapshot of the counts as they
        stood BEFORE that row happened.

        Fitting the counts on the same rows the model trains on leaks the label: a row's own
        click is inside its own `ad_ctr`. The linear model spread that across many weights
        and was hurt a little; a tree split on it at once and boosting stopped after three
        rounds. Nothing in AUC or log loss says "leak" -- it shows up as a model that cannot
        be improved.

        The snapshot is what serving actually has: the past. K-fold out-of-fold counts are
        the other standard answer and use the data more efficiently, but they let a row see
        its own future, which a live system never does.
        """
        ads = corpus.ads
        imp_ad: dict = {}
        clk_ad: dict = {}
        imp_adv: dict = {}
        clk_adv: dict = {}
        imp_cat: dict = {}
        clk_cat: dict = {}
        rows = np.asarray(rows)
        n_train = len(rows)
        a_arr = ev["ad"][rows]
        c_arr = np.asarray(labels)
        self.prior = float(c_arr.mean()) if n_train else 0.02

        self.row_dense = np.zeros((n_train, 4), dtype=np.float32)
        seen = 0
        clicks = 0
        for i in range(n_train):
            ad = int(a_arr[i])
            adv = ads["advertiser"][ad]
            cat = ads["category"][ad]

            # Snapshot FIRST. The running prior stands in until there is enough history for
            # the empirical rate to mean anything.
            p_now = (clicks + alpha * self.prior) / (seen + alpha)
            ia = imp_ad.get(ad, 0)
            self.row_dense[i, 0] = (clk_ad.get(ad, 0) + alpha * p_now) / (ia + alpha)
            iv = imp_adv.get(adv, 0)
            self.row_dense[i, 1] = (clk_adv.get(adv, 0) + alpha * p_now) / (iv + alpha)
            ic = imp_cat.get(cat, 0)
            self.row_dense[i, 2] = (clk_cat.get(cat, 0) + alpha * p_now) / (ic + alpha)
            self.row_dense[i, 3] = math.log1p(ia) * 0.1

            c = int(c_arr[i])
            imp_ad[ad] = ia + 1
            clk_ad[ad] = clk_ad.get(ad, 0) + c
            imp_adv[adv] = iv + 1
            clk_adv[adv] = clk_adv.get(adv, 0) + c
            imp_cat[cat] = ic + 1
            clk_cat[cat] = clk_cat.get(cat, 0) + c
            seen += 1
            clicks += c

        p = self.prior
        # Beta-smoothed rates. With alpha=60 an ad needs real traffic before its own rate
        # outweighs the prior, which is what stops a 1-impression 1-click ad from winning
        # every auction. These are the SERVING counts: everything known by the end of train.
        self.ad_ctr = {k: (clk_ad[k] + alpha * p) / (v + alpha) for k, v in imp_ad.items()}
        self.adv_ctr = {k: (clk_adv[k] + alpha * p) / (v + alpha) for k, v in imp_adv.items()}
        self.cat_ctr = {k: (clk_cat[k] + alpha * p) / (v + alpha) for k, v in imp_cat.items()}
        self.imp_ad = imp_ad

    def features(self, corpus: Corpus, uid: int, ad: int, row: int | None = None):
        """`row` names a training row, whose counting features come from its own snapshot.
        Without it the serving counts are used, which is right for anything after train."""
        u, ads = corpus.users, corpus.ads
        cat = ads["category"][ad]
        adv = ads["advertiser"][ad]
        country = u["country"][uid]
        age = u["age"][uid]
        os_ = u["os"][uid]
        # A linear model can only use an interaction it is handed, so the cross set is chosen
        # from where the signal is: affinity is a dot product between a user side built from
        # {interests, country, age, os} and an ad side built from {category, advertiser}.
        # Crossing only against category leaves the advertiser half unrepresentable -- that
        # was the whole of the first version's gap to the ceiling, and no amount of training
        # would have closed it.
        idx = [
            _h(f"c={country}"), _h(f"ci={u['city'][uid]}"), _h(f"a={age}"),
            _h(f"g={u['gender'][uid]}"), _h(f"o={os_}"), _h(f"l={u['lang'][uid]}"),
            _h(f"cat={cat}"), _h(f"adv={adv}"), _h(f"camp={ads['campaign'][ad]}"),
            _h(f"c_cat={country}|{cat}"), _h(f"a_cat={age}|{cat}"), _h(f"o_cat={os_}|{cat}"),
            _h(f"c_adv={country}|{adv}"), _h(f"a_adv={age}|{adv}"), _h(f"o_adv={os_}|{adv}"),
        ]
        for it in u["interests"][uid]:
            idx.append(_h(f"i_cat={it}|{cat}"))
            idx.append(_h(f"i_adv={it}|{adv}"))
            idx.append(_h(f"i={it}"))
        if row is not None and getattr(self, "row_dense", None) is not None:
            dense = self.row_dense[row]
        else:
            dense = np.array([
                self.ad_ctr.get(ad, self.prior),
                self.adv_ctr.get(adv, self.prior),
                self.cat_ctr.get(cat, self.prior),
                math.log1p(self.imp_ad.get(ad, 0)) * 0.1,
            ], dtype=np.float32)
        return idx, dense

    def raw_logit(self, idx, dense) -> float:
        return float(self.b + self.w[idx].sum() + float(self.wn @ dense))

    def predict(self, idx, dense) -> float:
        z = self.platt_a * self.raw_logit(idx, dense) + self.platt_b
        return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))

    def fit(self, corpus: Corpus, ev: dict, rows, labels, epochs: int = 2, lr: float = 0.18,
            l2: float = 0.0):
        """`l2` shrinks a weight toward zero in proportion to its size, applied only to the
        features a row touches.

        Without it the only thing limiting a rare cross is how few times it is seen -- and a
        cross seen nine times fits those nine rows exactly. That is what made the 400-
        advertiser crosses cost AUC: 16,000 cells, ~9 rows each, every one memorised.
        """
        rows = np.asarray(rows)
        labels = np.asarray(labels, dtype=np.float64)
        n_train = len(rows)
        order = np.arange(n_train)
        rng = np.random.default_rng(SEED)
        feats = [None] * n_train
        for ep in range(epochs):
            rng.shuffle(order)
            for n in order:
                r = int(rows[n])
                if feats[n] is None:
                    # row=n indexes the snapshot: this row's counts are the history BEFORE it
                    feats[n] = self.features(corpus, int(ev["user"][r]), int(ev["ad"][r]), row=int(n))
                idx, dense = feats[n]
                z = self.b + self.w[idx].sum() + float(self.wn @ dense)
                p = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))
                err = p - float(labels[n])
                self.gb += err * err
                self.b -= lr * err / (math.sqrt(self.gb) + 1.0)
                gw = err + l2 * self.w[idx]       # L2 only on the features this row touches
                self.g[idx] += gw * gw
                self.w[idx] -= lr * gw / (np.sqrt(self.g[idx]) + 1.0)
                gd = err * dense + l2 * self.wn
                self.gn += gd * gd
                self.wn -= lr * gd / (np.sqrt(self.gn) + 1.0)
        self._train_feats = feats

    def calibrate(self, corpus: Corpus, ev: dict, rows, labels, iters: int = 240):
        """Platt scaling on a held-out slice.

        An auction prices with bid * pCTR, so a model that ranks perfectly and is scaled
        wrong still misprices every impression. Ranking metrics cannot see this.
        """
        rows = np.asarray(rows)
        zs = np.array([self.raw_logit(*self.features(corpus, int(ev["user"][n]), int(ev["ad"][n])))
                       for n in rows], dtype=np.float64)
        y = np.asarray(labels, dtype=np.float64)
        a, b = 1.0, 0.0
        for _ in range(iters):
            p = 1.0 / (1.0 + np.exp(-np.clip(a * zs + b, -30, 30)))
            ga = float(((p - y) * zs).mean())
            gb = float((p - y).mean())
            a -= 1.0 * ga
            b -= 1.0 * gb
        self.platt_a, self.platt_b = float(a), float(b)


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
class GbdtModel:
    """Gradient-boosted trees over the same information, for the interactions a linear model
    on hashed crosses cannot reach.

    The linear model needs every interaction handed to it as an explicit cross, and a cross
    is only learnable where the data populates it -- advertiser crosses cost AUC at 400
    advertisers and paid at 60. Trees find interactions by splitting, and they share
    statistical strength across cells instead of giving each its own free parameter, which is
    the property that matters when the ad corpus grows faster than the log does.

    The features are deliberately the SAME information the linear model gets: interests as
    indicator columns, the low-cardinality categoricals, and the smoothed counting features.
    A fair comparison is between the models, not between the feature sets.
    """

    N_DENSE = len(INTERESTS) + 7 + 4

    def __init__(self, ctr: "CtrModel"):
        self.ctr = ctr
        self.booster = None
        self.platt_a, self.platt_b = 1.0, 0.0
        self._ix = {v: i for i, v in enumerate(INTERESTS)}
        self._maps = {
            "country": {v: i for i, v in enumerate(COUNTRIES)},
            "age": {v: i for i, v in enumerate(AGES)},
            "gender": {v: i for i, v in enumerate(GENDERS)},
            "os": {v: i for i, v in enumerate(OSES)},
            "lang": {v: i for i, v in enumerate(LANGS)},
            "category": {v: i for i, v in enumerate(CATEGORIES)},
        }

    def row(self, corpus: Corpus, uid: int, ad: int, out: np.ndarray,
            src: int | None = None) -> np.ndarray:
        """`src` names a training row, so the counts come from before it happened. Trees are
        the model that punishes the leak hardest, so this is the one that most needs it."""
        u, ads = corpus.users, corpus.ads
        out[:] = 0.0
        for it in u["interests"][uid]:
            out[self._ix[it]] = 1.0
        b = len(INTERESTS)
        cat = ads["category"][ad]
        adv = ads["advertiser"][ad]
        out[b + 0] = self._maps["country"][u["country"][uid]]
        out[b + 1] = self._maps["age"][u["age"][uid]]
        out[b + 2] = self._maps["gender"][u["gender"][uid]]
        out[b + 3] = self._maps["os"][u["os"][uid]]
        out[b + 4] = self._maps["lang"][u["lang"][uid]]
        out[b + 5] = self._maps["category"][cat]
        out[b + 6] = int(adv.split("-")[1])
        if src is not None and getattr(self.ctr, "row_dense", None) is not None:
            out[b + 7:b + 11] = self.ctr.row_dense[src]
        else:
            out[b + 7] = self.ctr.ad_ctr.get(ad, self.ctr.prior)
            out[b + 8] = self.ctr.adv_ctr.get(adv, self.ctr.prior)
            out[b + 9] = self.ctr.cat_ctr.get(cat, self.ctr.prior)
            out[b + 10] = math.log1p(self.ctr.imp_ad.get(ad, 0)) * 0.1
        return out

    def matrix(self, corpus: Corpus, uids, adids, srcs=None) -> np.ndarray:
        X = np.zeros((len(uids), self.N_DENSE), dtype=np.float32)
        for i in range(len(uids)):
            self.row(corpus, int(uids[i]), int(adids[i]), X[i],
                     src=None if srcs is None else int(srcs[i]))
        return X

    @property
    def categorical(self):
        b = len(INTERESTS)
        return list(range(b, b + 7))

    def fit(self, corpus: Corpus, ev: dict, n_train: int, n_cal: int, rounds: int = 320,
            max_rows: int = 60_000):
        import lightgbm as lgb

        # Capped: building the matrix is a Python loop per row and dominated startup at
        # 156 s for the full log. The model stops after a handful of rounds anyway.
        # Cap the FIT rows only. Reassigning n_train also moved the validation slice's lower
        # bound, which grew it from 30k rows to 110k and made the whole thing slower rather
        # than faster -- the opposite of what the cap was for.
        n_fit = min(n_train, max_rows)
        Xtr = self.matrix(corpus, ev["user"][:n_fit], ev["ad"][:n_fit], srcs=np.arange(n_fit))
        ytr = ev["click"][:n_fit].astype(np.float32)
        Xva = self.matrix(corpus, ev["user"][n_train:n_cal], ev["ad"][n_train:n_cal])
        yva = ev["click"][n_train:n_cal].astype(np.float32)

        train = lgb.Dataset(Xtr, label=ytr, categorical_feature=self.categorical,
                            free_raw_data=False)
        valid = lgb.Dataset(Xva, label=yva, categorical_feature=self.categorical,
                            reference=train, free_raw_data=False)
        params = {
            "objective": "binary", "metric": "binary_logloss",
            "learning_rate": 0.06, "num_leaves": 63, "min_data_in_leaf": 120,
            "feature_fraction": 0.85, "bagging_fraction": 0.85, "bagging_freq": 1,
            "lambda_l2": 2.0, "verbosity": -1, "num_threads": 0, "seed": SEED,
        }
        self.booster = lgb.train(
            params, train, num_boost_round=rounds, valid_sets=[valid],
            callbacks=[lgb.early_stopping(30, verbose=False)],
        )
        self.best_iter = int(self.booster.best_iteration or rounds)

        # Trees are trained on log loss and come out close to calibrated, but "close" is not
        # a price. The same Platt step the linear model gets, on the same held-out slice.
        z = self._raw(Xva)
        a, b = 1.0, 0.0
        for _ in range(240):
            p = 1.0 / (1.0 + np.exp(-np.clip(a * z + b, -30, 30)))
            a -= 1.0 * float(((p - yva) * z).mean())
            b -= 1.0 * float((p - yva).mean())
        self.platt_a, self.platt_b = float(a), float(b)

    def _raw(self, X: np.ndarray) -> np.ndarray:
        p = np.clip(self.booster.predict(X, num_iteration=self.best_iter), 1e-7, 1 - 1e-7)
        return np.log(p / (1 - p))

    def predict_batch(self, corpus: Corpus, uid: int, adids) -> np.ndarray:
        X = self.matrix(corpus, np.full(len(adids), uid), adids)
        z = self.platt_a * self._raw(X) + self.platt_b
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def auc(y: np.ndarray, s: np.ndarray) -> float:
    y = np.asarray(y); s = np.asarray(s)
    pos, neg = y.sum(), len(y) - y.sum()
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(s)
    ranks = np.empty(len(s), dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1)
    # average ranks within ties, or AUC drifts when many scores collide
    _, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts)); np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def roc_points(y: np.ndarray, s: np.ndarray, max_points: int = 60) -> list:
    """The ROC curve itself: (FPR, TPR) as the threshold sweeps from +inf down to -inf.

    Built the way the definition is built -- sort by score descending, then walk. Each step
    onto a POSITIVE moves the curve up by 1/n_pos; each step onto a NEGATIVE moves it right
    by 1/n_neg. That is the whole construction, and it is why the area under it equals the
    probability that a random positive outscores a random negative: every rightward step
    sweeps a rectangle whose height is the fraction of positives already above that negative.

    Thinned for transport; the AUC reported elsewhere is computed on all rows, not on these.
    """
    order = np.argsort(-np.asarray(s, dtype=np.float64))
    yy = np.asarray(y)[order]
    tp = np.cumsum(yy)
    fp = np.cumsum(1 - yy)
    n_pos, n_neg = float(tp[-1]), float(fp[-1])
    if n_pos == 0 or n_neg == 0:
        return []
    idx = np.unique(np.linspace(0, len(yy) - 1, max_points).astype(int))
    pts = [{"fpr": 0.0, "tpr": 0.0}]
    pts += [{"fpr": round(float(fp[i] / n_neg), 4), "tpr": round(float(tp[i] / n_pos), 4)}
            for i in idx]
    return pts


def gauc(y: np.ndarray, s: np.ndarray, groups: np.ndarray) -> dict:
    """AUC computed WITHIN each request, then averaged by impressions.

    Pooled AUC gets credit for ordering impressions that are never ranked against each other
    -- telling a high-CTR user from a low-CTR one is real signal and useless signal, because
    you only ever choose among the ads eligible for ONE request. GAUC asks the question the
    ranker is actually asked, and is almost always lower.

    A group whose rows are all clicks or all non-clicks has no AUC at all. Those rows are
    dropped, and the count is returned rather than hidden: a GAUC computed over a tenth of
    the traffic is a different claim from one computed over all of it.
    """
    y = np.asarray(y)
    s = np.asarray(s)
    groups = np.asarray(groups)
    total, weight, used, skipped, skipped_groups = 0.0, 0, 0, 0, 0
    for g in np.unique(groups):
        m = groups == g
        yg = y[m]
        pos = int(yg.sum())
        if pos == 0 or pos == len(yg):
            skipped += int(len(yg))
            skipped_groups += 1
            continue
        total += auc(yg, s[m]) * len(yg)
        weight += int(len(yg))
        used += 1
    return {
        "gauc": round(total / weight, 4) if weight else float("nan"),
        "groups_used": used,
        "groups_skipped": skipped_groups,
        "rows_used": weight,
        "rows_skipped": skipped,
    }


def logloss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def calibration_curve(y: np.ndarray, p: np.ndarray, bins: int = 12):
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    edges = np.unique(edges)
    out, ece = [], 0.0
    for i in range(len(edges) - 1):
        m = (p >= edges[i]) & (p <= edges[i + 1] if i == len(edges) - 2 else p < edges[i + 1])
        if m.sum() < 5:
            continue
        pred, obs = float(p[m].mean()), float(y[m].mean())
        out.append({"pred": pred, "obs": obs, "n": int(m.sum())})
        ece += (m.sum() / len(y)) * abs(pred - obs)
    return out, float(ece)
