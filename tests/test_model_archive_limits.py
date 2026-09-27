"""The expansion policy for an imported model archive (audit SEC-5).

The ratio guard alone scales with the upload: at the default 200 MB cap it permits
`20 × 200 MB = 4 GiB` of declared expansion, and `unpack` returns every member's bytes, so
that lands in the serving process's RAM on top of the upload buffer. An absolute ceiling is
what makes the guard a bound rather than a multiplier.

Tested as arithmetic because that is what it is. Reaching the ceiling through a real archive
would need a ~75 MB upload just to get past the ratio guard first; the end-to-end proof that a
bomb is refused before extraction stays in
`tests/test_training.py::test_import_rejects_zip_bomb`.
"""

import pytest

from app.model_archive import (
    _DECOMPRESSION_FLOOR_BYTES,
    _MAX_DECOMPRESSION_RATIO,
    _MAX_UNCOMPRESSED_BYTES,
    _refuse_if_overexpanded,
)
from app.registry import UnsafeModelError

MiB = 1024 * 1024


def test_a_small_archive_that_deflates_well_is_allowed():
    """The floor exists for exactly this: a tiny bundle is mostly JSON and compresses ~16x."""
    _refuse_if_overexpanded(total_uncompressed=60 * MiB, compressed=1 * MiB)


def test_a_small_archive_past_the_floor_is_refused():
    with pytest.raises(UnsafeModelError, match="possible zip bomb"):
        _refuse_if_overexpanded(total_uncompressed=_DECOMPRESSION_FLOOR_BYTES + 1, compressed=1)


def test_a_large_legitimate_bundle_is_allowed():
    """A big model is mostly `head.skops`, which skops stores uncompressed, so a real large
    bundle expands barely at all — 500 MiB from a 200 MB upload is generous already."""
    _refuse_if_overexpanded(total_uncompressed=500 * MiB, compressed=200 * MiB)


def test_the_absolute_ceiling_refuses_what_the_ratio_alone_would_allow():
    """The finding itself: the ratio scales with the upload, so it stops bounding anything.

    A 200 MB upload buys `20 x 200 MB = 4 GiB` of ratio allowance, and every byte of it would
    be read into memory. The ceiling is what turns the guard back into a bound.
    """
    ratio_allowance = _MAX_DECOMPRESSION_RATIO * 200 * MiB
    assert ratio_allowance > _MAX_UNCOMPRESSED_BYTES, "the ratio must be the looser of the two"

    with pytest.raises(UnsafeModelError, match="possible zip bomb"):
        _refuse_if_overexpanded(total_uncompressed=_MAX_UNCOMPRESSED_BYTES + 1,
                                compressed=200 * MiB)


def test_the_ceiling_leaves_room_for_any_plausible_bundle():
    """A guard that rejects real work is worse than none, so the headroom is stated.

    The largest bundle this project describes is a 300-label model at 200k features: the head
    is `n_labels x n_features x 4` bytes = ~240 MB, plus a vocabulary of ~3 MB and the
    vectorizer. The ceiling has to sit well above that and well below the 4 GiB the ratio
    alone would wave through.
    """
    assert 1024 * MiB <= _MAX_UNCOMPRESSED_BYTES
    assert 2048 * MiB >= _MAX_UNCOMPRESSED_BYTES
