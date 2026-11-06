# Data dictionary and provenance

OpenML 41214 (freMTPL2freq) and 41215 (freMTPL2sev), originating from CASdatasets by C. Dutang and A. Charpentier. OpenML lists CC0. Historical French motor third-party liability; amounts EUR, exposure years. The local frequency snapshot has 678,013 policy rows. The target is observed matched positive-payment losses, not ultimate notified claims or an Indian insurance tariff.

| Field | Meaning | Unit/treatment |
| --- | --- | --- |
| IDpol | Policy identifier | Unique in frequency input |
| Exposure | Observed coverage duration | Years; positive values retained, including >1 |
| ClaimNb | Original reported claim count | Audit only; can differ from linked payments |
| PaidClaimNb | Number of linked positive severity rows | Modelling frequency target |
| ClaimAmount | Sum of linked payment amounts | EUR |
| AvgClaimAmount | ClaimAmount / PaidClaimNb | EUR per matched paid claim; positive policies only |
| Frequency | PaidClaimNb / Exposure | Annual matched paid-claim rate |
| PurePremium | ClaimAmount / Exposure | Annual observed matched loss cost |
| VehPower, VehAge, DrivAge, BonusMalus | Original risk features | Source units/categories |
| Area, VehBrand, VehGas, Region | Original risk categories | Encoded on training data only |
| LogDensity | log(1 + Density) | Derived source feature |

Models do not use notified count, payment amount, exposure, frequency or pure-premium
targets as predictive features. Exposure supplies fitting weights and expected-loss scaling.
Orphan severity rows are quantified separately. No losses are capped or winsorized.
