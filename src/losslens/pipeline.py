import joblib
import numpy as np
import pandas as pd
import plotly.express as px
from scipy.stats import genpareto
from sklearn.metrics import mean_gamma_deviance, mean_poisson_deviance

from .data import prepare
from .io import ROOT, chart, report
from .models import FEATURES, bootstrap, calibration, evaluate, fit_models, predictions, split


def tail_analysis(amounts):
    amounts = np.asarray(amounts, dtype=float)
    rows = []
    for quantile in [0.9, 0.95, 0.975]:
        threshold = float(np.quantile(amounts, quantile))
        excess = amounts[amounts > threshold] - threshold
        shape, _, scale = genpareto.fit(excess, floc=0)
        rows.append(
            dict(
                threshold_quantile=quantile,
                threshold_eur=threshold,
                excess_claims=len(excess),
                gpd_shape=float(shape),
                gpd_scale=float(scale),
                finite_mean=bool(shape < 1),
            )
        )
    curves = []
    for deductible in [0, 500, 1000, 5000]:
        for limit in [1000, 5000, 10000, 50000, 100000, 500000]:
            limited = np.minimum(np.maximum(amounts - deductible, 0), limit).mean()
            curves.append(
                dict(
                    deductible_eur=deductible,
                    layer_limit_eur=limit,
                    expected_layer_payment_eur=float(limited),
                    fraction_uncapped_mean=float(limited / amounts.mean()),
                )
            )
    return pd.DataFrame(rows), pd.DataFrame(curves)


def run(refresh=False, max_policies=None):
    df, audit = prepare(refresh)
    if max_policies:
        df = df.sample(min(max_policies, len(df)), random_state=52)
    train, validation, test = split(df)
    models = fit_models(train)
    all_metrics = []
    holdout = None
    for name, subset in [("train", train), ("validation", validation), ("test", test)]:
        preds = predictions(models, subset)
        all_metrics += [evaluate(subset, p, model, name) for model, p in preds.items()]
        if name == "test":
            holdout = preds
    scores = pd.DataFrame(all_metrics)
    chosen = scores[scores.subset == "validation"].sort_values("tweedie_deviance").iloc[0].model
    scores.to_csv(ROOT / "reports/model_comparison.csv", index=False)
    models_dir = ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    joblib.dump(models, models_dir / "pricing.joblib")
    id_split = pd.concat(
        [
            subset[["IDpol"]].assign(split=name)
            for name, subset in [("train", train), ("validation", validation), ("test", test)]
        ]
    )
    id_split.to_csv(ROOT / "reports/policy_splits.csv", index=False)
    test_out = test[["IDpol", "Exposure", "ClaimAmount", "PaidClaimNb", "Region", "DrivAge"]].copy()
    for name, prediction in holdout.items():
        test_out[f"{name}_annual_loss_cost"] = prediction
    test_out.to_parquet(ROOT / "data/processed/holdout_predictions.parquet", index=False)
    intervals = bootstrap(test, holdout)
    calib = calibration(test, holdout[chosen])
    calib.to_csv(ROOT / "reports/risk_calibration.csv", index=False)
    age = calibration(test, holdout[chosen], "driver_age")
    age.to_csv(ROOT / "reports/age_calibration.csv", index=False)
    region = calibration(test, holdout[chosen], "region")
    region.to_csv(ROOT / "reports/region_calibration.csv", index=False)
    # Actual claim-level amounts, rather than policy averages, are required for layer calculations.
    severity = pd.read_parquet(ROOT / "data/raw/severity.parquet")
    severity = severity[severity.IDpol.isin(train.IDpol)]
    tail, curves = tail_analysis(severity.ClaimAmount)
    tail.to_csv(ROOT / "reports/tail_sensitivity.csv", index=False)
    curves.to_csv(ROOT / "reports/limited_loss_curves.csv", index=False)
    scenario = []
    pred = holdout[chosen]
    for inflation in [0, 0.05, 0.1, 0.2]:
        for expenses in [0.15, 0.25, 0.35]:
            expected = float((pred * test.Exposure).sum() * (1 + inflation))
            scenario.append(
                dict(
                    claim_cost_inflation=inflation,
                    expense_ratio=expenses,
                    expected_claims_eur=expected,
                    indicative_required_premium_eur=expected / (1 - expenses),
                )
            )
    scenario = pd.DataFrame(scenario)
    scenario.to_csv(ROOT / "reports/portfolio_scenarios.csv", index=False)
    positive = test.PaidClaimNb > 0
    diagnostic = dict(
        frequency_poisson_deviance=float(
            mean_poisson_deviance(
                test.Frequency, models["frequency"].predict(test[FEATURES]), sample_weight=test.Exposure
            )
        ),
        severity_gamma_deviance=float(
            mean_gamma_deviance(
                test.loc[positive, "AvgClaimAmount"],
                models["severity"].predict(test.loc[positive, FEATURES]),
                sample_weight=test.loc[positive, "PaidClaimNb"],
            )
        ),
    )
    metrics = dict(
        policies_modelled=len(df),
        full_dataset=max_policies is None,
        train=len(train),
        validation=len(validation),
        test=len(test),
        selected_on_validation=chosen,
        split="policy-disjoint random split; no temporal claim",
        intervals=intervals,
        component_diagnostics=diagnostic,
        comparisons=all_metrics,
        target=audit["target"],
    )
    chart(
        px.bar(
            calib,
            x="risk_band",
            y=["observed", "expected"],
            barmode="group",
            title=f"Holdout observed and expected payments — {chosen}",
        ),
        "calibration",
    )
    chart(
        px.line(
            curves,
            x="layer_limit_eur",
            y="fraction_uncapped_mean",
            color="deductible_eur",
            log_x=True,
            title="Empirical limited-loss curves — training claims only",
        ),
        "limited_loss_curves",
    )
    with pd.ExcelWriter(ROOT / "reports/pricing_review.xlsx", engine="xlsxwriter") as writer:
        for name, frame in [
            ("Models", scores),
            ("Calibration", calib),
            ("Driver age", age),
            ("Regions", region),
            ("Tail", tail),
            ("Layers", curves),
            ("Scenarios", scenario),
        ]:
            frame.to_excel(writer, sheet_name=name, index=False)
            writer.sheets[name].set_column(0, len(frame.columns) - 1, 24)
            writer.sheets[name].freeze_panes(1, 0)
    chosen_score = scores[(scores.model == chosen) & (scores.subset == "test")].iloc[0]
    report(
        "LossLens",
        f"""# Motor loss-cost study

Public historical French motor third-party liability data, sourced through OpenML from CASdatasets by Dutang and Charpentier. Amounts are EUR and exposures are years. These are observed payment proxies, not an Indian tariff or an estimate of ultimate notified losses.

The frequency source contains {audit["frequency_rows"]:,} policies and the severity source {audit["severity_rows"]:,} claim rows. There are {audit["unmatched_severity_rows"]} orphan severity rows, excluded and quantified in data_audit.json. {audit["claim_count_disagreements"]:,} policies have differing notified and matched positive-payment counts. The target deliberately uses matched positive-payment claim counts so that frequency times severity reconciles to observed matched losses. All positive exposures, including {audit["exposure_over_one_year"]:,} exposures over one year, and all large claim amounts are retained.

Models: constant loss-cost baseline, spline-feature Poisson/Gamma product, boosted-frequency/Gamma product, and direct Tweedie. Training, validation and test policies are disjoint. All transformations fit on training data. Fixed candidate specifications are compared using validation Tweedie deviance at power 1.5. The selected model is {chosen}.

On {len(test):,} holdout policies, selected-model observed/expected loss is {chosen_score.observed_expected_ratio:.3f}. Its 95% policy-resampling interval is [{intervals[chosen]["oe_2_5"]:.3f}, {intervals[chosen]["oe_97_5"]:.3f}]. This interval describes held-out portfolio sampling variability with fixed fitted models; it does not include refitting or future structural shifts. Full baseline comparisons are in model_comparison.csv.

Recommendation: use the validation-selected model as the reference for further investigation, then review sparse region and driver-age segments before applying any pricing action. Calibration tables flag groups with fewer than 100 paid claims. Large losses are retained, so ranking and total-loss calibration should be assessed separately.

Tail fits and empirical layer-payment curves use training claim-level amounts only. GPD threshold stability is shown at three thresholds, including whether the fitted tail has a finite mean. Curves are descriptive estimates from observed claims and have substantial tail sampling uncertainty. Expense ratios and claim-cost inflation in the scenario tool are explicit assumptions.

Limitations: no credible policy dates for temporal backtesting, no claim development adjustment, unobserved claims missing payment amounts, French historical risk mix, no causal interpretation of model features, and no real underwriting or regulatory approval. Higher predictive complexity is retained only if its validation result warrants it. No commercial premium savings are claimed.
""",
        metrics,
    )
    return {k: v for k, v in metrics.items() if k != "comparisons"}
