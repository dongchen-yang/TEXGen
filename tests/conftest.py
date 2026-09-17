import os
import sys

# The submodule root, so `spuv` and `inference_specific_samples` import the way launch.py sees them.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
