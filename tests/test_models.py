import numpy as np
import pandas as pd
import pytest

from losslens.models import evaluate, split
from losslens.pipeline import tail_analysis


def test_policy_split_has_no_overlap():
    df = pd.DataFrame({"IDpol": np.arange(100), "PaidClaimNb": np.arange(100) % 2})
    a, b, c = split(df)
    assert len(a) + len(b) + len(c) == 100
    assert not set(a.IDpol) & set(c.IDpol)


def test_exposure_scales_expected_loss():
    df = pd.DataFrame({"Exposure": [0.5, 2.0], "ClaimAmount": [5.0, 20.0], "PurePremium": [10.0, 10.0]})
    metric = evaluate(df, np.array([10.0, 10.0]), "perfect", "test")
    assert metric["expected_loss"] == 25
    assert metric["observed_expected_ratio"] == 1


def test_empirical_layer_curve_is_monotone_and_bounded():
    _, curve = tail_analysis(np.random.default_rng(1).lognormal(7, 1, 1000))
    for _, group in curve.groupby("deductible_eur"):
        assert np.all(np.diff(group.expected_layer_payment_eur) >= 0)
        assert group.fraction_uncapped_mean.between(0, 1).all()
    zero = curve[curve.deductible_eur == 0]
    deductible = curve[curve.deductible_eur == 5000]
    assert np.all(
        deductible.expected_layer_payment_eur.to_numpy() <= zero.expected_layer_payment_eur.to_numpy()
    )


def test_duplicate_ids_fail_split_guard():
    df = pd.DataFrame({"IDpol": [1] * 100, "PaidClaimNb": np.arange(100) % 2})
    with pytest.raises(ValueError, match="leakage"):
        split(df)
