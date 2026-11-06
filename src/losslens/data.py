"""Public motor claims inputs and explicit target reconciliation."""

import pandas as pd

from .io import ROOT, dump, fetch

SOURCES = {
    "frequency": "https://data.openml.org/datasets/0004/41214/dataset_41214.pq",
    "severity": "https://data.openml.org/datasets/0004/41215/dataset_41215.pq",
}


def prepare(refresh=False):
    tables = {
        name: pd.read_parquet(fetch(url, ROOT / f"data/raw/{name}.parquet", refresh))
        for name, url in SOURCES.items()
    }
    f, s = tables["frequency"], tables["severity"]
    if f.IDpol.duplicated().any():
        raise ValueError("Frequency data must contain one row per policy")
    if (f.Exposure <= 0).any() or (s.ClaimAmount <= 0).any():
        raise ValueError("Invalid exposure or nonpositive severity; investigate source before fitting")
    joined = s[s.IDpol.isin(f.IDpol)]
    aggregate = joined.groupby("IDpol").ClaimAmount.agg(PaidClaimNb="count", ClaimAmount="sum")
    df = f.merge(aggregate, how="left", left_on="IDpol", right_index=True, validate="one_to_one")
    df[["PaidClaimNb", "ClaimAmount"]] = df[["PaidClaimNb", "ClaimAmount"]].fillna(0)
    df["AvgClaimAmount"] = df.ClaimAmount / df.PaidClaimNb.where(df.PaidClaimNb > 0)
    df["Frequency"] = df.PaidClaimNb / df.Exposure
    df["PurePremium"] = df.ClaimAmount / df.Exposure
    df["LogDensity"] = __import__("numpy").log1p(df.Density)
    for col in ["Area", "VehBrand", "VehGas", "Region"]:
        df[col] = df[col].astype(str)
    audit = dict(
        frequency_rows=len(f),
        severity_rows=len(s),
        unmatched_severity_rows=int((~s.IDpol.isin(f.IDpol)).sum()),
        unmatched_severity_amount=float(s.loc[~s.IDpol.isin(f.IDpol), "ClaimAmount"].sum()),
        duplicate_frequency_ids=int(f.IDpol.duplicated().sum()),
        missing_values=int(f.isna().sum().sum()),
        exposure_over_one_year=int((f.Exposure > 1).sum()),
        notified_claims=int(f.ClaimNb.sum()),
        matched_positive_payment_claims=int(df.PaidClaimNb.sum()),
        claim_count_disagreements=int((df.PaidClaimNb != df.ClaimNb).sum()),
        matched_paid_loss=float(df.ClaimAmount.sum()),
        target="positive-payment claim count and observed matched payment amounts; not ultimate notified claims",
        treatment="all positive exposures and claim amounts retained; orphan severity rows excluded and quantified; no winsorization",
    )
    dump(ROOT / "reports/data_audit.json", audit)
    out = ROOT / "data/processed"
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / "policies.parquet", index=False)
    dump(
        ROOT / "data/provenance.json",
        dict(
            kind="public historical French motor third-party liability",
            sources=SOURCES,
            attribution="C. Dutang and A. Charpentier (2018), CASdatasets; OpenML datasets 41214 and 41215, CC0",
            geography="France",
            units="EUR; exposure years",
            target=audit["target"],
        ),
    )
    return df, audit
