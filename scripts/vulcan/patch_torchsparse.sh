#!/bin/bash
# Patch torchsparse v2.1.0 (07f021b) for torch>=2.9: master (385f5ce) replaced the
# deprecated Tensor.type() with scalar_type(). Applied by hand because master
# also adds a Rust/maturin build dep that cannot build on a compute node.
cd /scratch/dya78/lightgen_repo/build_src/torchsparse
for f in $(grep -rl '\.type(), "' torchsparse/backend/); do
  sed -i 's/\.type(), "/.scalar_type(), "/g' "$f"
  echo "patched: $f"
done
echo "remaining: $(grep -rn '\.type(), "' torchsparse/backend/ | wc -l)"
