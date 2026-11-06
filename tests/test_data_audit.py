import pandas as pd
import pytest

from losslens import data


def test_payment_count_target_reconciles_and_orphans_are_audited(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "ROOT", tmp_path)
    freq = pd.DataFrame(
        {
            "IDpol": [1.0, 2.0],
            "Exposure": [0.5, 1.5],
            "ClaimNb": [3, 0],
            "Density": [100, 200],
            "Area": ["A", "B"],
            "VehBrand": ["B1", "B2"],
            "VehGas": ["Regular", "Diesel"],
            "Region": ["R1", "R2"],
        }
    )
    severity = pd.DataFrame({"IDpol": [1.0, 1.0, 999.0], "ClaimAmount": [10.0, 20.0, 100.0]})
    f, s = tmp_path / "frequency.parquet", tmp_path / "severity.parquet"
    freq.to_parquet(f)
    severity.to_parquet(s)
    monkeypatch.setattr(data, "fetch", lambda url, path, refresh: f if "41214" in url else s)
    result, audit = data.prepare()
    assert result.PaidClaimNb.tolist() == [2, 0]
    assert audit["unmatched_severity_rows"] == 1
    assert audit["unmatched_severity_amount"] == 100
    assert audit["exposure_over_one_year"] == 1
    positive = result.PaidClaimNb > 0
    calculated = (
        result.loc[positive, "Exposure"]
        * result.loc[positive, "Frequency"]
        * result.loc[positive, "AvgClaimAmount"]
    )
    assert calculated.sum() == 30
    assert result.ClaimAmount.sum() == 30


def test_duplicate_frequency_policy_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "ROOT", tmp_path)
    f, s = tmp_path / "frequency.parquet", tmp_path / "severity.parquet"
    pd.DataFrame({"IDpol": [1, 1]}).to_parquet(f)
    pd.DataFrame({"IDpol": [1], "ClaimAmount": [10.0]}).to_parquet(s)
    monkeypatch.setattr(data, "fetch", lambda url, path, refresh: f if "41214" in url else s)
    with pytest.raises(ValueError, match="one row"):
        data.prepare()
