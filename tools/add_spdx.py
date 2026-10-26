# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 MatrixArkAI
"""Put an SPDX header on every source file, idempotently.

A repo can be Apache-2.0 at the root and still leave every individual file unmarked, which is
what a downstream consumer actually reads when a single file is vendored out of context. The
header travels with the file; the LICENSE at the root does not.
"""

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
PY = "# SPDX-License-Identifier: Apache-2.0\n# Copyright 2026 MatrixArkAI\n"
HTML = "<!-- SPDX-License-Identifier: Apache-2.0 -->\n<!-- Copyright 2026 MatrixArkAI -->\n"

SKIP = {".venv", ".git", "artifacts", "__pycache__"}

added, already, scanned = [], [], 0
for path in sorted(ROOT.rglob("*")):
    if any(part in SKIP for part in path.parts):
        continue
    if path.suffix not in (".py", ".html"):
        continue
    scanned += 1
    text = path.read_text(encoding="utf-8")
    # Only the first few lines count. Searching the whole file makes THIS file report itself
    # as marked, because it carries the token in a string constant -- a check that passes by
    # matching its own definition rather than the thing it is checking.
    if "SPDX-License-Identifier" in "".join(text.splitlines(keepends=True)[:4]):
        already.append(path.name)
        continue
    header = HTML if path.suffix == ".html" else PY
    if path.suffix == ".html" and text.lstrip().lower().startswith("<!doctype"):
        # the doctype must stay the first thing the parser sees
        i = text.index("\n") + 1
        path.write_text(text[:i] + header + text[i:], encoding="utf-8")
    else:
        path.write_text(header + text, encoding="utf-8")
    added.append(path.name)

print(f"scanned {scanned} source files")
print(f"header added  : {len(added)}  {added}")
print(f"already marked: {len(already)}  {already}")
if scanned == 0:
    sys.exit("scanned nothing -- the walk is wrong, not the tree")
