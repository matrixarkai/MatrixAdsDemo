# Contributing

This project is Apache-2.0 and open to anyone in the world. Fork it, change it, open issues, send
pull requests. Outside contributions are the point, not an exception to manage.

## License and security

Unless explicitly stated otherwise, contributions intentionally submitted for
inclusion are licensed under
[Apache-2.0](https://github.com/matrixarkai/MatrixAdsDemo/blob/main/LICENSE),
without additional terms or conditions.
For vulnerabilities, follow the
[security policy](https://github.com/matrixarkai/MatrixAdsDemo/blob/main/SECURITY.md)
instead of posting exploit details in a public issue.

## Run it

```bash
docker build -t ads-demo .
docker run -d --name ads -p 8080:8080 ads-demo
# http://localhost:8080
```

Or locally, with Python 3.12:

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
PYTHONHASHSEED=0 ARTIFACT=./artifacts/indexes.pkl .venv/bin/python -m app.build_index
PYTHONHASHSEED=0 ARTIFACT=./artifacts/indexes.pkl PORT=8081 .venv/bin/python -m app.serve
```

`PYTHONHASHSEED=0` is not optional. The model keys weights by `hash(str)`, which Python randomises
per process, so without it two runs of the same code score differently.

## What a good change looks like

**Measure it, and say what you measured against.** Every number in this repo is reported next to a
baseline or a ceiling, because a bare figure cannot be judged. The click simulator is a known
function, so the AUC of its own probabilities is the best score achievable on this data — that is
the denominator, not 1.0.

**A guard must be able to fail.** If you add a check, show it failing on the defect it is for. The
targeting index is verified against brute-force predicate evaluation at build time for exactly this
reason: eligibility must be exact, so it is checked rather than trusted.

**Say what you did not do.** The README's list of what is deliberately out of scope is as useful as
the code. If your change leaves something unresolved, write that down rather than letting the next
reader rediscover it.

## Before you open a pull request

- `python -m app.build_index` completes and the exactness guard reports zero mismatches.
- `python -m app.serve` serves the portal and a request returns winners.
- New source files carry `# SPDX-License-Identifier: Apache-2.0`. Run `python tools/add_spdx.py`.
- Dependencies in `requirements.txt` match the Dockerfile, so a local run and a container run are
  the same program.

## Scope

This is a teaching-sized system: it demonstrates the shape of ad serving — exact eligibility,
approximate retrieval, calibrated prediction, auction — on synthetic data. It is not a production
ad server and does not pretend to be. Changes that make the mechanism clearer or the measurement
more honest are welcome. Changes that add scale without adding insight are probably not.
