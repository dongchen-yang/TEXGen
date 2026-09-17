#!/bin/bash
# The size of the fork's delta against upstream TEXGen, file by file (informational). Run from anywhere;
# `git fetch upstream` first if upstream/main is stale.
cd "$(dirname "$0")/.." || exit 1
echo "== every path under launch.py, spuv/, requirements.txt and .gitignore against upstream/main"
git diff --stat upstream/main -- launch.py spuv/ requirements.txt .gitignore
echo "== upstream files renamed texgen_emission_* and edited in place, each against its upstream source"
echo "   (git's rename detection does not pair files edited this much, so the pairs are named here)"
while read -r old new; do
    git diff --stat "upstream/main:$old" "HEAD:$new" | head -1
done <<'PAIRS'
spuv/systems/texgen_base.py spuv/systems/texgen_emission_base.py
spuv/systems/texgen_test.py spuv/systems/texgen_emission_test.py
spuv/models/sparse_networks/texgen_network.py spuv/models/sparse_networks/texgen_emission_network.py
PAIRS
