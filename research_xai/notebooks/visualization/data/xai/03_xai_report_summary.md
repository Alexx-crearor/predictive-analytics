# DeepKoopFormer XAI — Research Results Summary

## Experimental Protocol

- Models: M0, M1, M2, M3, M4
- Datasets: Electricity, Financial, CMIP6 Wind, CMIP6 Pressure, Lorenz
- XAI configuration: lookback=100, horizon=8, seed=7
- Attribution target: output channel 0, forecast step 8 (zero-based index 7)
- XAI samples: 8 selected test windows per dataset
- IG integration steps: 32
- Occlusion block size: 10 time steps
- Faithfulness: 20% block-channel masking; 10 random selections
- Robustness: Gaussian noise with alpha 0.005, 0.01, 0.02; 3 repeats
- Full-test MSE is taken from the training evaluation, not estimated using 8 XAI windows.

## Full-test MSE — M0 to M4

| dataset | M0 | M1 | M2 | M3 | M4 |
| --- | --- | --- | --- | --- | --- |
| Electricity | 0.03072941 | 0.02876955 | 0.03071698 | 0.02194518 | 0.02570613 |
| Financial | 0.00234537 | 0.00217975 | 0.00203047 | 0.00312794 | 0.00845691 |
| CMIP6 Wind | 0.03506190 | 0.03723591 | 0.03188637 | 0.02909119 | 0.02996535 |
| CMIP6 Pressure | 0.00826280 | 0.00997398 | 0.00902528 | 0.00751516 | 0.00817099 |
| Lorenz | 0.00007680 | 0.00010584 | 0.00007509 | 0.00070041 | 0.00062434 |

## Best Forecast Model Among M0–M4

| dataset | best_model_among_M0_M4 | best_full_test_mse | m4_vs_m1_mse_improvement_pct |
| --- | --- | --- | --- |
| Electricity | M3 | 0.02194518 | +10.65 |
| Financial | M2 | 0.00203047 | -287.98 |
| CMIP6 Wind | M3 | 0.02909119 | +19.53 |
| CMIP6 Pressure | M3 | 0.00751516 | +18.08 |
| Lorenz | M2 | 0.00007509 | -489.92 |

## Faithfulness: Top-IG Beats Random (%)

| dataset | M0 | M1 | M2 | M3 | M4 |
| --- | --- | --- | --- | --- | --- |
| Electricity | 50.0 | 75.0 | 75.0 | 62.5 | 62.5 |
| Financial | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| CMIP6 Wind | 50.0 | 62.5 | 75.0 | 62.5 | 37.5 |
| CMIP6 Pressure | 50.0 | 75.0 | 62.5 | 100.0 | 87.5 |
| Lorenz | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |

## Robustness: Relative L1 Change at Alpha=0.02 (%)

| dataset | M0 | M1 | M2 | M3 | M4 |
| --- | --- | --- | --- | --- | --- |
| Electricity | 2.776 | 2.443 | 2.743 | 2.464 | 2.391 |
| Financial | 2.665 | 2.772 | 2.670 | 2.470 | 2.891 |
| CMIP6 Wind | 2.458 | 2.568 | 2.467 | 2.443 | 2.300 |
| CMIP6 Pressure | 2.567 | 2.482 | 2.619 | 2.353 | 2.328 |
| Lorenz | 2.131 | 2.178 | 2.175 | 2.184 | 2.121 |

## Robustness: Top-K Overlap at Alpha=0.02 (%)

| dataset | M0 | M1 | M2 | M3 | M4 |
| --- | --- | --- | --- | --- | --- |
| Electricity | 100.00 | 100.00 | 99.65 | 100.00 | 99.31 |
| Financial | 100.00 | 100.00 | 96.88 | 98.96 | 97.92 |
| CMIP6 Wind | 99.58 | 100.00 | 97.08 | 98.33 | 99.58 |
| CMIP6 Pressure | 100.00 | 100.00 | 97.50 | 99.58 | 98.33 |
| Lorenz | 99.31 | 100.00 | 99.31 | 99.31 | 100.00 |

## Interpretation Limits

- Numerical IG convergence does not prove faithfulness.
- Attribution agreement does not prove causality.
- Robustness to tested Gaussian noise does not imply robustness to arbitrary perturbations.
- The 8 selected windows are not the full test set.
- The three noise repeats are not independent test windows.
- Faithfulness measures prediction changes, not forecast error changes.
- Spectral stability does not guarantee forecast accuracy.
- M1 uses a single Koopman transition, while M2/M3/M4 use temporal rollouts.
- Shift gap is based on student re-encoding; it is not the EMA teacher training loss.
- All conclusions are conditional on the tested configuration and seed.
- No statistical significance is established by this visualization notebook.

## Output Locations

- Figures: C:/Users/Win 11 New/Desktop/pa gk/research_xai/notebooks/visualization/figures/03_xai
- Summary CSV and validation: C:/Users/Win 11 New/Desktop/pa gk/research_xai/notebooks/visualization/data/xai
- Original XAI evidence: C:/Users/Win 11 New/Desktop/pa gk/research_xai/results/reduced_grid_225_v1/xai/p100_h8