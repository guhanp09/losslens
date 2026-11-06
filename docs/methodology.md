# LossLens methodology

Each frequency row is a policy. Severity rows are aggregated by policy ID, preserving the count of matched positive payments and total payment amount. Orphan severity rows are excluded and quantified. All positive exposures and large losses are retained. Matched paid count is used for frequency; notified counts remain in the audit. This gives the identity exposure × frequency × average paid severity = observed matched payments.

Poisson regression fits annualized paid frequency with exposure weights. Gamma regression fits policy-average positive severity with paid-count weights. The product estimates annual loss cost. Numeric spline features and categorical encodings fit on training data only. The boosted challenger changes the frequency model while retaining the Gamma severity model. A direct Tweedie model uses power 1.5 and exposure weights. The validation comparison uses the same power for all candidates.

Policies are stratified by positive payment and split 60/20/20 with fixed seeds. There are no reliable policy dates for temporal validation. Model selection uses validation deviance; test results are then reported without refitting. Observed/expected ratios and group calibration complement deviance. Two hundred policy bootstrap samples estimate holdout portfolio sampling variability with fixed fitted models, excluding refit and structural uncertainty.

Tail fits use claim-level training amounts, not policy-average severities or test losses. GPD thresholds are 90%, 95% and 97.5%; fitted shape, scale and finite-mean status are shown. Empirical layer loss is mean(min(max(claim − deductible, 0), limit)). Limited-loss curves are monotone but have material uncertainty in sparse tails. Commercial expense and inflation factors are scenario assumptions. No regulatory adequacy, profitability, causal feature effect or ultimate-loss sufficiency is claimed.

## Data flow

Public frequency + severity parquet → audited policy/payment join → policy partitions → training-only feature pipelines → candidate models → validation selection → holdout and tail diagnostics → scenario review.
