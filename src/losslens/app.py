import json

import pandas as pd
import plotly.express as px
import streamlit as st

from .io import ROOT, require, theme


def main():
    theme("LossLens", "Motor claim frequency, severity, calibration, and portfolio scenarios")
    require(ROOT / "reports/metrics.json", "losslens run")
    require(ROOT / "data/processed/holdout_predictions.parquet", "losslens run")
    metrics = json.loads((ROOT / "reports/metrics.json").read_text())
    st.caption("Historical French motor data · EUR · observed matched positive-payment claims")
    cols = st.columns(3)
    cols[0].metric("Policies modelled", f"{metrics['policies_modelled']:,}")
    cols[1].metric("Holdout policies", f"{metrics['test']:,}")
    cols[2].metric("Validation-selected model", metrics["selected_on_validation"])
    tabs = st.tabs(["Model comparison", "Calibration", "Portfolio scenarios", "Large losses", "Data audit"])
    with tabs[0]:
        scores = pd.read_csv(ROOT / "reports/model_comparison.csv")
        st.dataframe(scores, hide_index=True)
        st.plotly_chart(
            px.bar(
                scores[scores.subset != "train"],
                x="model",
                y="tweedie_deviance",
                color="subset",
                barmode="group",
            ),
            width="stretch",
        )
    with tabs[1]:
        group = st.selectbox("Calibration group", ["risk", "age", "region"])
        c = pd.read_csv(ROOT / f"reports/{group}_calibration.csv")
        st.plotly_chart(
            px.bar(c, x=c.columns[0], y=["observed", "expected"], barmode="group"), width="stretch"
        )
        st.dataframe(c, hide_index=True)
    with tabs[2]:
        predictions = pd.read_parquet(ROOT / "data/processed/holdout_predictions.parquet")
        regions = st.multiselect(
            "Portfolio regions",
            sorted(predictions.Region.unique()),
            default=sorted(predictions.Region.unique()),
        )
        inflation = st.slider("Assumed claim-cost inflation", 0.0, 0.4, 0.05, 0.01)
        expense = st.slider("Assumed expense ratio", 0.0, 0.5, 0.25, 0.01)
        exposure_multiplier = st.slider("Exposure multiplier", 0.25, 3.0, 1.0, 0.05)
        selected = predictions[predictions.Region.isin(regions)]
        expected = (
            (selected[f"{metrics['selected_on_validation']}_annual_loss_cost"] * selected.Exposure).sum()
            * (1 + inflation)
            * exposure_multiplier
        )
        st.metric("Expected observed-payment proxy", f"€{expected:,.0f}")
        st.metric("Indicative premium including assumed expenses", f"€{expected / (1 - expense):,.0f}")
        st.caption(
            "Mix, inflation, exposure and expense assumptions; no claim-development, profit or capital loading."
        )
    with tabs[3]:
        curves = pd.read_csv(ROOT / "reports/limited_loss_curves.csv")
        st.plotly_chart(
            px.line(
                curves, x="layer_limit_eur", y="fraction_uncapped_mean", color="deductible_eur", log_x=True
            ),
            width="stretch",
        )
        st.dataframe(pd.read_csv(ROOT / "reports/tail_sensitivity.csv"), hide_index=True)
    with tabs[4]:
        st.json(json.loads((ROOT / "reports/data_audit.json").read_text()))
    st.download_button(
        "Download pricing review", (ROOT / "reports/pricing_review.xlsx").read_bytes(), "pricing_review.xlsx"
    )
