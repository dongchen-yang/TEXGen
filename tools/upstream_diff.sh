#!/bin/bash
# The size of the fork's delta against upstream TEXGen, file by file (informational). Run from anywhere;
# `git fetch upstream` first if upstream/main is stale.
set -euo pipefail
cd "$(dirname "$0")/.." || exit 1
git rev-parse -q --verify upstream/main >/dev/null || {
    echo "no upstream/main: git remote add upstream https://github.com/CVMI-Lab/TEXGen.git && git fetch upstream" >&2
    exit 1
}
echo "== launch.py, spuv/, requirements.txt and .gitignore at HEAD against upstream/main"
git diff --stat upstream/main HEAD -- launch.py spuv/ requirements.txt .gitignore
echo "== upstream files renamed texgen_emission_* and edited in place, each against its upstream source"
echo "   (above, git pairs only texgen_network.py by itself: texgen_base.py is still a file, the re-export, and texgen_test.py shows as deleted)"
while read -r old new; do
    git diff --stat "upstream/main:$old" "HEAD:$new" | { head -1; cat >/dev/null; }
done <<'PAIRS'
spuv/systems/texgen_base.py spuv/systems/texgen_emission_base.py
spuv/systems/texgen_test.py spuv/systems/texgen_emission_test.py
spuv/models/sparse_networks/texgen_network.py spuv/models/sparse_networks/texgen_emission_network.py
PAIRS
