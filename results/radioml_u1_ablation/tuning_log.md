# Tuning Log - RadioML 2018.01A Modulation Classification

Random-search sweep following `docs/tuning_budget.md`: shared trial samples across all families, seeded per trial, selection by mean validation accuracy.

- Trials per family: `16`
- Seeds per trial: `[0, 1, 2, 3, 4]`
- Sweep seed: `20260503`
- Search space: see `trials.json`

## Selected configuration per family

| family | trial | val acc | test acc | test std | hyperparameters |
| --- | ---: | ---: | ---: | ---: | --- |
| `complex` | 5 | 0.4442 | 0.4389 | 0.0123 | batch_size=256, hidden_features=64, learning_rate=0.02364, steps=400 |
| `real_equivariant` | 5 | 0.4545 | 0.4477 | 0.0072 | batch_size=256, hidden_features=64, learning_rate=0.02364, steps=400 |
| `real_matched_params` | 5 | 0.3390 | 0.3369 | 0.0210 | batch_size=256, hidden_features=64, learning_rate=0.02364, steps=400 |
| `real_stacked` | 5 | 0.3311 | 0.3321 | 0.0301 | batch_size=256, hidden_features=64, learning_rate=0.02364, steps=400 |
| `real_polar` | 5 | 0.4349 | 0.4327 | 0.0172 | batch_size=256, hidden_features=64, learning_rate=0.02364, steps=400 |
| `real_magnitude` | 5 | 0.3984 | 0.3952 | 0.0160 | batch_size=256, hidden_features=64, learning_rate=0.02364, steps=400 |

## All trials (mean across seeds)

| family | trial | val acc | test acc | hyperparameters |
| --- | ---: | ---: | ---: | --- |
| `complex` | 0 | 0.3891 | 0.3888 | batch_size=512, hidden_features=32, learning_rate=0.002444, steps=800 |
| `complex` | 1 | 0.3773 | 0.3846 | batch_size=512, hidden_features=16, learning_rate=0.007904, steps=800 |
| `complex` | 2 | 0.3413 | 0.3502 | batch_size=256, hidden_features=16, learning_rate=0.02782, steps=200 |
| `complex` | 3 | 0.3890 | 0.3853 | batch_size=512, hidden_features=64, learning_rate=0.03977, steps=200 |
| `complex` | 4 | 0.3481 | 0.3496 | batch_size=128, hidden_features=32, learning_rate=0.001134, steps=800 |
| `complex` | 5 | 0.4442 | 0.4389 | batch_size=256, hidden_features=64, learning_rate=0.02364, steps=400 |
| `complex` | 6 | 0.3461 | 0.3530 | batch_size=256, hidden_features=32, learning_rate=0.01289, steps=200 |
| `complex` | 7 | 0.3294 | 0.3362 | batch_size=256, hidden_features=16, learning_rate=0.002315, steps=200 |
| `complex` | 8 | 0.3239 | 0.3295 | batch_size=256, hidden_features=16, learning_rate=0.002142, steps=200 |
| `complex` | 9 | 0.3204 | 0.3184 | batch_size=128, hidden_features=16, learning_rate=0.001647, steps=200 |
| `complex` | 10 | 0.3431 | 0.3514 | batch_size=128, hidden_features=32, learning_rate=0.004728, steps=400 |
| `complex` | 11 | 0.3861 | 0.3942 | batch_size=256, hidden_features=64, learning_rate=0.02619, steps=200 |
| `complex` | 12 | 0.3719 | 0.3792 | batch_size=128, hidden_features=64, learning_rate=0.008588, steps=400 |
| `complex` | 13 | 0.3570 | 0.3621 | batch_size=512, hidden_features=32, learning_rate=0.001191, steps=800 |
| `complex` | 14 | 0.3497 | 0.3563 | batch_size=256, hidden_features=32, learning_rate=0.003545, steps=400 |
| `complex` | 15 | 0.3325 | 0.3393 | batch_size=128, hidden_features=16, learning_rate=0.002283, steps=400 |
| `real_equivariant` | 0 | 0.4009 | 0.3979 | batch_size=512, hidden_features=32, learning_rate=0.002444, steps=800 |
| `real_equivariant` | 1 | 0.4381 | 0.4334 | batch_size=512, hidden_features=16, learning_rate=0.007904, steps=800 |
| `real_equivariant` | 2 | 0.3748 | 0.3758 | batch_size=256, hidden_features=16, learning_rate=0.02782, steps=200 |
| `real_equivariant` | 3 | 0.4337 | 0.4288 | batch_size=512, hidden_features=64, learning_rate=0.03977, steps=200 |
| `real_equivariant` | 4 | 0.3444 | 0.3394 | batch_size=128, hidden_features=32, learning_rate=0.001134, steps=800 |
| `real_equivariant` | 5 | 0.4545 | 0.4477 | batch_size=256, hidden_features=64, learning_rate=0.02364, steps=400 |
| `real_equivariant` | 6 | 0.3620 | 0.3636 | batch_size=256, hidden_features=32, learning_rate=0.01289, steps=200 |
| `real_equivariant` | 7 | 0.2753 | 0.2795 | batch_size=256, hidden_features=16, learning_rate=0.002315, steps=200 |
| `real_equivariant` | 8 | 0.2743 | 0.2761 | batch_size=256, hidden_features=16, learning_rate=0.002142, steps=200 |
| `real_equivariant` | 9 | 0.2419 | 0.2423 | batch_size=128, hidden_features=16, learning_rate=0.001647, steps=200 |
| `real_equivariant` | 10 | 0.3454 | 0.3500 | batch_size=128, hidden_features=32, learning_rate=0.004728, steps=400 |
| `real_equivariant` | 11 | 0.4057 | 0.4059 | batch_size=256, hidden_features=64, learning_rate=0.02619, steps=200 |
| `real_equivariant` | 12 | 0.3763 | 0.3827 | batch_size=128, hidden_features=64, learning_rate=0.008588, steps=400 |
| `real_equivariant` | 13 | 0.3567 | 0.3614 | batch_size=512, hidden_features=32, learning_rate=0.001191, steps=800 |
| `real_equivariant` | 14 | 0.3525 | 0.3520 | batch_size=256, hidden_features=32, learning_rate=0.003545, steps=400 |
| `real_equivariant` | 15 | 0.2973 | 0.2984 | batch_size=128, hidden_features=16, learning_rate=0.002283, steps=400 |
| `real_matched_params` | 0 | 0.4173 | 0.4142 | batch_size=512, hidden_features=32, learning_rate=0.002444, steps=800 |
| `real_matched_params` | 1 | 0.4201 | 0.4236 | batch_size=512, hidden_features=16, learning_rate=0.007904, steps=800 |
| `real_matched_params` | 2 | 0.3061 | 0.3083 | batch_size=256, hidden_features=16, learning_rate=0.02782, steps=200 |
| `real_matched_params` | 3 | 0.3465 | 0.3513 | batch_size=512, hidden_features=64, learning_rate=0.03977, steps=200 |
| `real_matched_params` | 4 | 0.3419 | 0.3407 | batch_size=128, hidden_features=32, learning_rate=0.001134, steps=800 |
| `real_matched_params` | 5 | 0.3390 | 0.3369 | batch_size=256, hidden_features=64, learning_rate=0.02364, steps=400 |
| `real_matched_params` | 6 | 0.3462 | 0.3508 | batch_size=256, hidden_features=32, learning_rate=0.01289, steps=200 |
| `real_matched_params` | 7 | 0.2688 | 0.2678 | batch_size=256, hidden_features=16, learning_rate=0.002315, steps=200 |
| `real_matched_params` | 8 | 0.2711 | 0.2680 | batch_size=256, hidden_features=16, learning_rate=0.002142, steps=200 |
| `real_matched_params` | 9 | 0.2486 | 0.2488 | batch_size=128, hidden_features=16, learning_rate=0.001647, steps=200 |
| `real_matched_params` | 10 | 0.3653 | 0.3612 | batch_size=128, hidden_features=32, learning_rate=0.004728, steps=400 |
| `real_matched_params` | 11 | 0.2975 | 0.2992 | batch_size=256, hidden_features=64, learning_rate=0.02619, steps=200 |
| `real_matched_params` | 12 | 0.3356 | 0.3324 | batch_size=128, hidden_features=64, learning_rate=0.008588, steps=400 |
| `real_matched_params` | 13 | 0.3875 | 0.3869 | batch_size=512, hidden_features=32, learning_rate=0.001191, steps=800 |
| `real_matched_params` | 14 | 0.3718 | 0.3684 | batch_size=256, hidden_features=32, learning_rate=0.003545, steps=400 |
| `real_matched_params` | 15 | 0.3077 | 0.3113 | batch_size=128, hidden_features=16, learning_rate=0.002283, steps=400 |
| `real_stacked` | 0 | 0.3991 | 0.4021 | batch_size=512, hidden_features=32, learning_rate=0.002444, steps=800 |
| `real_stacked` | 1 | 0.4002 | 0.4016 | batch_size=512, hidden_features=16, learning_rate=0.007904, steps=800 |
| `real_stacked` | 2 | 0.3002 | 0.2995 | batch_size=256, hidden_features=16, learning_rate=0.02782, steps=200 |
| `real_stacked` | 3 | 0.3556 | 0.3558 | batch_size=512, hidden_features=64, learning_rate=0.03977, steps=200 |
| `real_stacked` | 4 | 0.3182 | 0.3227 | batch_size=128, hidden_features=32, learning_rate=0.001134, steps=800 |
| `real_stacked` | 5 | 0.3311 | 0.3321 | batch_size=256, hidden_features=64, learning_rate=0.02364, steps=400 |
| `real_stacked` | 6 | 0.3310 | 0.3288 | batch_size=256, hidden_features=32, learning_rate=0.01289, steps=200 |
| `real_stacked` | 7 | 0.2539 | 0.2518 | batch_size=256, hidden_features=16, learning_rate=0.002315, steps=200 |
| `real_stacked` | 8 | 0.2530 | 0.2500 | batch_size=256, hidden_features=16, learning_rate=0.002142, steps=200 |
| `real_stacked` | 9 | 0.2325 | 0.2329 | batch_size=128, hidden_features=16, learning_rate=0.001647, steps=200 |
| `real_stacked` | 10 | 0.3472 | 0.3405 | batch_size=128, hidden_features=32, learning_rate=0.004728, steps=400 |
| `real_stacked` | 11 | 0.2823 | 0.2864 | batch_size=256, hidden_features=64, learning_rate=0.02619, steps=200 |
| `real_stacked` | 12 | 0.3787 | 0.3817 | batch_size=128, hidden_features=64, learning_rate=0.008588, steps=400 |
| `real_stacked` | 13 | 0.3637 | 0.3671 | batch_size=512, hidden_features=32, learning_rate=0.001191, steps=800 |
| `real_stacked` | 14 | 0.3516 | 0.3603 | batch_size=256, hidden_features=32, learning_rate=0.003545, steps=400 |
| `real_stacked` | 15 | 0.2554 | 0.2508 | batch_size=128, hidden_features=16, learning_rate=0.002283, steps=400 |
| `real_polar` | 0 | 0.4247 | 0.4308 | batch_size=512, hidden_features=32, learning_rate=0.002444, steps=800 |
| `real_polar` | 1 | 0.4341 | 0.4380 | batch_size=512, hidden_features=16, learning_rate=0.007904, steps=800 |
| `real_polar` | 2 | 0.3998 | 0.3919 | batch_size=256, hidden_features=16, learning_rate=0.02782, steps=200 |
| `real_polar` | 3 | 0.4103 | 0.4086 | batch_size=512, hidden_features=64, learning_rate=0.03977, steps=200 |
| `real_polar` | 4 | 0.3755 | 0.3797 | batch_size=128, hidden_features=32, learning_rate=0.001134, steps=800 |
| `real_polar` | 5 | 0.4349 | 0.4327 | batch_size=256, hidden_features=64, learning_rate=0.02364, steps=400 |
| `real_polar` | 6 | 0.3786 | 0.3846 | batch_size=256, hidden_features=32, learning_rate=0.01289, steps=200 |
| `real_polar` | 7 | 0.2935 | 0.2947 | batch_size=256, hidden_features=16, learning_rate=0.002315, steps=200 |
| `real_polar` | 8 | 0.2893 | 0.2867 | batch_size=256, hidden_features=16, learning_rate=0.002142, steps=200 |
| `real_polar` | 9 | 0.2633 | 0.2615 | batch_size=128, hidden_features=16, learning_rate=0.001647, steps=200 |
| `real_polar` | 10 | 0.3888 | 0.3910 | batch_size=128, hidden_features=32, learning_rate=0.004728, steps=400 |
| `real_polar` | 11 | 0.3884 | 0.3841 | batch_size=256, hidden_features=64, learning_rate=0.02619, steps=200 |
| `real_polar` | 12 | 0.4389 | 0.4427 | batch_size=128, hidden_features=64, learning_rate=0.008588, steps=400 |
| `real_polar` | 13 | 0.4096 | 0.4137 | batch_size=512, hidden_features=32, learning_rate=0.001191, steps=800 |
| `real_polar` | 14 | 0.4100 | 0.4074 | batch_size=256, hidden_features=32, learning_rate=0.003545, steps=400 |
| `real_polar` | 15 | 0.3258 | 0.3308 | batch_size=128, hidden_features=16, learning_rate=0.002283, steps=400 |
| `real_magnitude` | 0 | 0.4006 | 0.3941 | batch_size=512, hidden_features=32, learning_rate=0.002444, steps=800 |
| `real_magnitude` | 1 | 0.3942 | 0.3887 | batch_size=512, hidden_features=16, learning_rate=0.007904, steps=800 |
| `real_magnitude` | 2 | 0.3267 | 0.3302 | batch_size=256, hidden_features=16, learning_rate=0.02782, steps=200 |
| `real_magnitude` | 3 | 0.3143 | 0.3179 | batch_size=512, hidden_features=64, learning_rate=0.03977, steps=200 |
| `real_magnitude` | 4 | 0.3388 | 0.3409 | batch_size=128, hidden_features=32, learning_rate=0.001134, steps=800 |
| `real_magnitude` | 5 | 0.3984 | 0.3952 | batch_size=256, hidden_features=64, learning_rate=0.02364, steps=400 |
| `real_magnitude` | 6 | 0.3630 | 0.3608 | batch_size=256, hidden_features=32, learning_rate=0.01289, steps=200 |
| `real_magnitude` | 7 | 0.2791 | 0.2761 | batch_size=256, hidden_features=16, learning_rate=0.002315, steps=200 |
| `real_magnitude` | 8 | 0.2746 | 0.2705 | batch_size=256, hidden_features=16, learning_rate=0.002142, steps=200 |
| `real_magnitude` | 9 | 0.2448 | 0.2476 | batch_size=128, hidden_features=16, learning_rate=0.001647, steps=200 |
| `real_magnitude` | 10 | 0.3450 | 0.3443 | batch_size=128, hidden_features=32, learning_rate=0.004728, steps=400 |
| `real_magnitude` | 11 | 0.3784 | 0.3721 | batch_size=256, hidden_features=64, learning_rate=0.02619, steps=200 |
| `real_magnitude` | 12 | 0.3958 | 0.3941 | batch_size=128, hidden_features=64, learning_rate=0.008588, steps=400 |
| `real_magnitude` | 13 | 0.3722 | 0.3753 | batch_size=512, hidden_features=32, learning_rate=0.001191, steps=800 |
| `real_magnitude` | 14 | 0.3619 | 0.3604 | batch_size=256, hidden_features=32, learning_rate=0.003545, steps=400 |
| `real_magnitude` | 15 | 0.2960 | 0.2944 | batch_size=128, hidden_features=16, learning_rate=0.002283, steps=400 |
