# Motor loss-cost study

Public historical French motor third-party liability data, sourced through OpenML from CASdatasets by Dutang and Charpentier. Amounts are EUR and exposures are years. These are observed payment proxies, not an Indian tariff or an estimate of ultimate notified losses.

The frequency source contains 678,013 policies and the severity source 26,639 claim rows. There are 195 orphan severity rows, excluded and quantified in data_audit.json. 9,117 policies have differing notified and matched positive-payment counts. The target deliberately uses matched positive-payment claim counts so that frequency times severity reconciles to observed matched losses. All positive exposures, including 1,224 exposures over one year, and all large claim amounts are retained.

Models: constant loss-cost baseline, spline-feature Poisson/Gamma product, boosted-frequency/Gamma product, and direct Tweedie. Training, validation and test policies are disjoint. All transformations fit on training data. Fixed candidate specifications are compared using validation Tweedie deviance at power 1.5. The selected model is boosted_product.

On 135,603 holdout policies, selected-model observed/expected loss is 1.005. Its 95% policy-resampling interval is [0.830, 1.324]. This interval describes held-out portfolio sampling variability with fixed fitted models; it does not include refitting or future structural shifts. Full baseline comparisons are in model_comparison.csv.

Recommendation: use the validation-selected model as the reference for further investigation, then review sparse region and driver-age segments before applying any pricing action. Calibration tables flag groups with fewer than 100 paid claims. Large losses are retained, so ranking and total-loss calibration should be assessed separately.

Tail fits and empirical layer-payment curves use training claim-level amounts only. GPD threshold stability is shown at three thresholds, including whether the fitted tail has a finite mean. Curves are descriptive estimates from observed claims and have substantial tail sampling uncertainty. Expense ratios and claim-cost inflation in the scenario tool are explicit assumptions.

Limitations: no credible policy dates for temporal backtesting, no claim development adjustment, unobserved claims missing payment amounts, French historical risk mix, no causal interpretation of model features, and no real underwriting or regulatory approval. Higher predictive complexity is retained only if its validation result warrants it. No commercial premium savings are claimed.
