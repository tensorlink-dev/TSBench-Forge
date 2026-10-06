"""Reference forecasts round-trip: export contexts, replay returned quantiles."""

from types import SimpleNamespace

import numpy as np
import pytest

import reference_io as rio

LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def _challenges(n=5):
    rng = np.random.default_rng(0)
    return [SimpleNamespace(context=rng.normal(size=50 + i), truth=np.zeros(8 + 4 * (i % 2)), meta={})
            for i in range(n)]


def _write(path, challenges, fingerprint):
    hmax = max(len(ch.truth) for ch in challenges)
    q = np.full((len(challenges), len(LEVELS), hmax), np.nan)
    for i, ch in enumerate(challenges):
        h = len(ch.truth)
        q[i, :, :h] = np.asarray(LEVELS)[:, None] + i  # monotone, row-identifiable
    np.savez_compressed(path, quantiles=q, levels=np.asarray(LEVELS), fingerprint=np.asarray(fingerprint))


def test_export_never_includes_the_truth(tmp_path):
    chs = _challenges()
    rio.export_contexts(chs, tmp_path / "ctx.npz")
    z = np.load(tmp_path / "ctx.npz")
    assert set(z.files) == {"context_flat", "context_offsets", "horizons", "levels", "fingerprint"}
    assert list(z["horizons"]) == [len(ch.truth) for ch in chs]


def test_replayed_forecasts_line_up_with_their_challenge(tmp_path):
    chs = _challenges()
    fp = rio.export_contexts(chs, tmp_path / "ctx.npz")
    (tmp_path / "ref").mkdir()
    _write(tmp_path / "ref" / "timesfm3.npz", chs, fp)
    fcs, errors = rio.load_reference_forecasters(tmp_path / "ref", chs)
    assert errors == {}
    out = fcs["timesfm3"](chs[3].context)
    assert out.mean.shape == (len(chs[3].truth),)
    assert out.mean[0] == pytest.approx(0.5 + 3)
    assert out.quantiles[0.9][0] == pytest.approx(0.9 + 3)


def test_forecasts_for_another_challenge_set_are_refused(tmp_path):
    chs = _challenges()
    (tmp_path / "ref").mkdir()
    _write(tmp_path / "ref" / "toto2-2.5b.npz", chs, "some-other-round")
    fcs, errors = rio.load_reference_forecasters(tmp_path / "ref", chs)
    assert fcs == {} and "different challenge set" in errors["toto2-2.5b"]
