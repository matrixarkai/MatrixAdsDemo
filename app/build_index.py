# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 MatrixArkAI
"""Build the targeting index and the ANN index for the ad corpus, and write them to disk.

Run at image build time so a container starts by LOADING indexes rather than deriving them.
Deriving them took 48 s on an idle machine and over three minutes on a busy one, and every
restart paid it again — including the one that followed a Docker restart and left the demo
down.

    python -m app.build_index          # writes $ARTIFACT (default /srv/artifacts/indexes.pkl)
    REBUILD=1 python -m app.build_index

What lands in the artifact: the corpus, the inverted targeting index, the IVF ANN index, the
CTR model, and the measurements taken over them. The event log does not — it is what the
model was fitted on, not something serving reads.
"""

from app.serve import ARTIFACT, build

if __name__ == "__main__":
    import os

    os.environ.setdefault("REBUILD", "1")     # an explicit build never reuses an old artifact
    build()
    print(f"[build_index] artifact at {ARTIFACT}")
