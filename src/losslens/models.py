"""Exposure-aware models and policy-level validation."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import GammaRegressor, PoissonRegressor, TweedieRegressor
from sklearn.metrics import mean_tweedie_deviance
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, SplineTransformer, StandardScaler

NUMERIC = ["VehPower", "VehAge", "DrivAge", "BonusMalus", "LogDensity"]
CATEGORICAL = ["Area", "VehBrand", "VehGas", "Region"]
FEATURES = NUMERIC + CATEGORICAL


def split(df):
    train, rest = train_test_split(df, test_size=0.4, random_state=26, stratify=df.PaidClaimNb > 0)
    validation, test = train_test_split(rest, test_size=0.5, random_state=27, stratify=rest.PaidClaimNb > 0)
    if set(train.IDpol) & (set(validation.IDpol) | set(test.IDpol)) or set(validation.IDpol) & set(
        test.IDpol
    ):
        raise ValueError("Policy leakage across splits")
    return train.copy(), validation.copy(), test.copy()


def preprocess(boost=False):
    numerical = (
        StandardScaler()
        if boost
        else Pipeline(
            [
                ("spline", SplineTransformer(n_knots=4, degree=2, extrapolation="linear")),
                ("scale", StandardScaler()),
            ]
        )
    )
    categories = (
        OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        if boost
        else OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    )
    return ColumnTransformer([("numeric", numerical, NUMERIC), ("categorical", categories, CATEGORICAL)])


def fit_models(train):
    x = train[FEATURES]
    freq = Pipeline([("features", preprocess()), ("model", PoissonRegressor(alpha=0.01, max_iter=1000))])
    freq.fit(x, train.Frequency, model__sample_weight=train.Exposure)
    positive = train.PaidClaimNb > 0
    severity = Pipeline([("features", preprocess()), ("model", GammaRegressor(alpha=1.0, max_iter=1000))])
    severity.fit(
        x.loc[positive],
        train.loc[positive, "AvgClaimAmount"],
        model__sample_weight=train.loc[positive, "PaidClaimNb"],
    )
    boost = Pipeline(
        [
            ("features", preprocess(True)),
            (
                "model",
                HistGradientBoostingRegressor(
                    loss="poisson",
                    learning_rate=0.06,
                    max_iter=120,
                    max_leaf_nodes=15,
                    min_samples_leaf=100,
                    l2_regularization=10,
                    random_state=26,
                ),
            ),
        ]
    )
    boost.fit(x, train.Frequency, model__sample_weight=train.Exposure)
    tweedie = Pipeline(
        [("features", preprocess()), ("model", TweedieRegressor(power=1.5, alpha=0.1, max_iter=1000))]
    )
    tweedie.fit(x, train.PurePremium, model__sample_weight=train.Exposure)
    return dict(
        frequency=freq,
        severity=severity,
        boosted_frequency=boost,
        tweedie=tweedie,
        constant_pure_premium=float(train.ClaimAmount.sum() / train.Exposure.sum()),
    )


def predictions(models, df):
    x = df[FEATURES]
    severity = models["severity"].predict(x)
    return {
        "constant": np.full(len(df), models["constant_pure_premium"]),
        "glm_product": models["frequency"].predict(x) * severity,
        "boosted_product": models["boosted_frequency"].predict(x) * severity,
        "tweedie": models["tweedie"].predict(x),
    }


def evaluate(df, prediction, name, subset):
    expected = prediction * df.Exposure.to_numpy()
    return dict(
        model=name,
        subset=subset,
        tweedie_deviance=float(
            mean_tweedie_deviance(df.PurePremium, prediction, sample_weight=df.Exposure, power=1.5)
        ),
        observed_loss=float(df.ClaimAmount.sum()),
        expected_loss=float(expected.sum()),
        observed_expected_ratio=float(df.ClaimAmount.sum() / expected.sum()),
        policies=len(df),
        exposure_years=float(df.Exposure.sum()),
    )


def bootstrap(df, predictions_by_name, repetitions=200, seed=61):
    rng = np.random.default_rng(seed)
    actual = df.ClaimAmount.to_numpy()
    exposure = df.Exposure.to_numpy()
    names = list(predictions_by_name)
    samples = {n: [] for n in names}
    for _ in range(repetitions):
        idx = rng.integers(0, len(df), len(df))
        for name, prediction in predictions_by_name.items():
            samples[name].append(float(actual[idx].sum() / (prediction[idx] * exposure[idx]).sum()))
    return {
        name: dict(
            oe_2_5=float(np.quantile(values, 0.025)),
            oe_97_5=float(np.quantile(values, 0.975)),
            repetitions=repetitions,
        )
        for name, values in samples.items()
    }


def calibration(df, prediction, group="risk_band"):
    data = df[["Exposure", "ClaimAmount", "PaidClaimNb", "DrivAge", "Region"]].copy()
    data["expected"] = prediction * data.Exposure
    if group == "risk_band":
        data[group] = pd.qcut(prediction, 10, duplicates="drop").astype(str)
    elif group == "driver_age":
        data[group] = pd.cut(data.DrivAge, [0, 25, 35, 50, 65, 120], right=False).astype(str)
    else:
        data[group] = data.Region
    result = (
        data.groupby(group)
        .agg(
            policies=("Exposure", "size"),
            exposure=("Exposure", "sum"),
            observed=("ClaimAmount", "sum"),
            expected=("expected", "sum"),
            paid_claims=("PaidClaimNb", "sum"),
        )
        .reset_index()
    )
    result["observed_expected"] = result.observed / result.expected
    result["sparse"] = result.paid_claims < 100
    return result
