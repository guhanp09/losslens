# Engineering notes

The frequency input has one row per policy. Positive-payment records are joined by policy
identifier, with unmatched records quantified separately. Matched paid-claim count and mean
paid severity reconcile to total matched payment amount. Large payments remain uncapped.

Policies are separated into training, validation and test partitions. Feature transformations
fit on training data only. Exposure weights determine annualized frequency and loss-cost
fitting; paid-count weights determine the severity fit. Candidate selection uses validation
Tweedie deviance. Test calibration and bootstrap intervals are reported after selection.

Fixtures check policy isolation, exposure scaling, target reconciliation, orphan records and
limited-loss curve properties. Tail fits use training claim-level amounts. Data scope and
uncertainty assumptions are documented in methodology.md and the generated pricing review.

Tested dependency versions are recorded in constraints.txt and environment.json.
