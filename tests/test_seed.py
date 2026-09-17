import importlib.util
import os

import pytest

from spuv.utils.seed import sample_seed

PARENT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "evaluation", "newdata_eval", "seedutil.py"))


@pytest.mark.skipif(not os.path.exists(PARENT), reason="the parent repo's seedutil.py is not beside this checkout")
def test_sample_seed_matches_the_parent_canonical_copy():
    spec = importlib.util.spec_from_file_location("seedutil", PARENT)
    seedutil = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(seedutil)
    for sha, seed in [("0001c49868a04c7180e5abe3d6a9334c", 0), ("ffff", 4), ("a", 123456)]:
        assert sample_seed(sha, seed) == seedutil.sample_seed(sha, seed)


def test_sample_seed_is_pinned_order_independent_and_full_width():
    assert sample_seed("abc", 0) == 6860934752652134519
    assert sample_seed("a", 0) != sample_seed("b", 0)
    assert 0 <= sample_seed("a", 0) < 2 ** 64
    # the value the parent's evaluation/newdata_eval/tests/test_seedutil.py pins; its top bit is
    # set, so a copy that masked the digest to int63 would return 4652262302224825907 here
    v = sample_seed("014276d502484e46b234cd1b73d05a8c", 3)
    assert v == 13875634339079601715 and v >= 2 ** 63
