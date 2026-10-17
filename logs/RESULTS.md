# workflow-en-v4 experiment results

Started (KST): 20261017_010400

Validation, generated fixture check, unit tests, and 780 scripted replays passed.
Pilot: README selections (shopping only), held separate from the main series.

| Model | Condition | Success | Trials | Invalid |
|---|---|---:|---:|---:|
| self-FP16 | free | 42 | 180 | 0 |
| self-FP16 | normal_restart | 35 | 120 | 0 |
| self-FP16 | error_restart | 34 | 240 | 0 |
| self-FP16 | signaled_error_restart | 10 | 240 | 0 |
| self-Q8 | free | 40 | 180 | 0 |
| self-Q8 | normal_restart | 38 | 120 | 0 |
| self-Q8 | error_restart | 30 | 240 | 0 |
| self-Q8 | signaled_error_restart | 6 | 240 | 0 |
| self-Q4 | free | 28 | 180 | 0 |
| self-Q4 | normal_restart | 33 | 120 | 0 |
| self-Q4 | error_restart | 52 | 240 | 0 |
| self-Q4 | signaled_error_restart | 49 | 240 | 0 |

Error families and horizons are separated in main/summary.csv.
Precision effects and adjusted p-values are in main/summary_precision.csv,
main/summary_interaction.csv, and main/summary_penalty.csv.
GPU memory values measure the whole device, not process-exclusive VRAM.
Earlier workflow-en-v2 results are not included.
