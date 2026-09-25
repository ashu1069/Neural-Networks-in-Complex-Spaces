# Tuning Log - RadioML 2018.01A Modulation Classification

Random-search sweep following `docs/tuning_budget.md`: shared trial samples across all families, seeded per trial, selection by mean validation accuracy.

- Trials per family: `16`
- Seeds per trial: `[0, 1, 2, 3, 4]`
- Sweep seed: `20260503`
- Search space: see `trials.json`

## Selected configuration per family

| family | trial | val acc | test acc | test std | hyperparameters |
| --- | ---: | ---: | ---: | ---: | --- |
| `complex` | 5 | 0.4374 | 0.4369 | 0.0131 | batch_size=256, hidden_features=64, learning_rate=0.02364, steps=400 |
| `real_stacked` | 5 | 0.3384 | 0.3363 | 0.0180 | batch_size=256, hidden_features=64, learning_rate=0.02364, steps=400 |
| `real_matched_params` | 5 | 0.3166 | 0.3197 | 0.0446 | batch_size=256, hidden_features=64, learning_rate=0.02364, steps=400 |
| `real_matched_flops` | 5 | 0.3499 | 0.3576 | 0.0534 | batch_size=256, hidden_features=64, learning_rate=0.02364, steps=400 |
| `real_polar` | 5 | 0.4327 | 0.4298 | 0.0207 | batch_size=256, hidden_features=64, learning_rate=0.02364, steps=400 |
| `real_phase` | 5 | 0.2143 | 0.2176 | 0.0754 | batch_size=256, hidden_features=64, learning_rate=0.02364, steps=400 |
| `real_magnitude` | 5 | 0.4016 | 0.3980 | 0.0125 | batch_size=256, hidden_features=64, learning_rate=0.02364, steps=400 |

## All trials (mean across seeds)

| family | trial | val acc | test acc | hyperparameters |
| --- | ---: | ---: | ---: | --- |
| `complex` | 0 | 0.3872 | 0.3865 | batch_size=512, hidden_features=32, learning_rate=0.002444, steps=800 |
| `complex` | 1 | 0.3773 | 0.3846 | batch_size=512, hidden_features=16, learning_rate=0.007904, steps=800 |
| `complex` | 2 | 0.3413 | 0.3502 | batch_size=256, hidden_features=16, learning_rate=0.02782, steps=200 |
| `complex` | 3 | 0.4144 | 0.4085 | batch_size=512, hidden_features=64, learning_rate=0.03977, steps=200 |
| `complex` | 4 | 0.3481 | 0.3496 | batch_size=128, hidden_features=32, learning_rate=0.001134, steps=800 |
| `complex` | 5 | 0.4374 | 0.4369 | batch_size=256, hidden_features=64, learning_rate=0.02364, steps=400 |
| `complex` | 6 | 0.3436 | 0.3498 | batch_size=256, hidden_features=32, learning_rate=0.01289, steps=200 |
| `complex` | 7 | 0.3294 | 0.3362 | batch_size=256, hidden_features=16, learning_rate=0.002315, steps=200 |
| `complex` | 8 | 0.3239 | 0.3295 | batch_size=256, hidden_features=16, learning_rate=0.002142, steps=200 |
| `complex` | 9 | 0.3204 | 0.3184 | batch_size=128, hidden_features=16, learning_rate=0.001647, steps=200 |
| `complex` | 10 | 0.3431 | 0.3514 | batch_size=128, hidden_features=32, learning_rate=0.004728, steps=400 |
| `complex` | 11 | 0.3864 | 0.3882 | batch_size=256, hidden_features=64, learning_rate=0.02619, steps=200 |
| `complex` | 12 | 0.3719 | 0.3792 | batch_size=128, hidden_features=64, learning_rate=0.008588, steps=400 |
| `complex` | 13 | 0.3574 | 0.3620 | batch_size=512, hidden_features=32, learning_rate=0.001191, steps=800 |
| `complex` | 14 | 0.3472 | 0.3554 | batch_size=256, hidden_features=32, learning_rate=0.003545, steps=400 |
| `complex` | 15 | 0.3325 | 0.3393 | batch_size=128, hidden_features=16, learning_rate=0.002283, steps=400 |
| `real_stacked` | 0 | 0.3990 | 0.4029 | batch_size=512, hidden_features=32, learning_rate=0.002444, steps=800 |
| `real_stacked` | 1 | 0.4002 | 0.4016 | batch_size=512, hidden_features=16, learning_rate=0.007904, steps=800 |
| `real_stacked` | 2 | 0.3002 | 0.2995 | batch_size=256, hidden_features=16, learning_rate=0.02782, steps=200 |
| `real_stacked` | 3 | 0.3541 | 0.3493 | batch_size=512, hidden_features=64, learning_rate=0.03977, steps=200 |
| `real_stacked` | 4 | 0.3182 | 0.3227 | batch_size=128, hidden_features=32, learning_rate=0.001134, steps=800 |
| `real_stacked` | 5 | 0.3384 | 0.3363 | batch_size=256, hidden_features=64, learning_rate=0.02364, steps=400 |
| `real_stacked` | 6 | 0.3339 | 0.3325 | batch_size=256, hidden_features=32, learning_rate=0.01289, steps=200 |
| `real_stacked` | 7 | 0.2539 | 0.2518 | batch_size=256, hidden_features=16, learning_rate=0.002315, steps=200 |
| `real_stacked` | 8 | 0.2530 | 0.2500 | batch_size=256, hidden_features=16, learning_rate=0.002142, steps=200 |
| `real_stacked` | 9 | 0.2325 | 0.2329 | batch_size=128, hidden_features=16, learning_rate=0.001647, steps=200 |
| `real_stacked` | 10 | 0.3472 | 0.3405 | batch_size=128, hidden_features=32, learning_rate=0.004728, steps=400 |
| `real_stacked` | 11 | 0.2862 | 0.2857 | batch_size=256, hidden_features=64, learning_rate=0.02619, steps=200 |
| `real_stacked` | 12 | 0.3787 | 0.3817 | batch_size=128, hidden_features=64, learning_rate=0.008588, steps=400 |
| `real_stacked` | 13 | 0.3630 | 0.3674 | batch_size=512, hidden_features=32, learning_rate=0.001191, steps=800 |
| `real_stacked` | 14 | 0.3500 | 0.3585 | batch_size=256, hidden_features=32, learning_rate=0.003545, steps=400 |
| `real_stacked` | 15 | 0.2554 | 0.2508 | batch_size=128, hidden_features=16, learning_rate=0.002283, steps=400 |
| `real_matched_params` | 0 | 0.4184 | 0.4141 | batch_size=512, hidden_features=32, learning_rate=0.002444, steps=800 |
| `real_matched_params` | 1 | 0.4202 | 0.4233 | batch_size=512, hidden_features=16, learning_rate=0.007904, steps=800 |
| `real_matched_params` | 2 | 0.3051 | 0.3079 | batch_size=256, hidden_features=16, learning_rate=0.02782, steps=200 |
| `real_matched_params` | 3 | 0.3380 | 0.3433 | batch_size=512, hidden_features=64, learning_rate=0.03977, steps=200 |
| `real_matched_params` | 4 | 0.3419 | 0.3407 | batch_size=128, hidden_features=32, learning_rate=0.001134, steps=800 |
| `real_matched_params` | 5 | 0.3166 | 0.3197 | batch_size=256, hidden_features=64, learning_rate=0.02364, steps=400 |
| `real_matched_params` | 6 | 0.3454 | 0.3517 | batch_size=256, hidden_features=32, learning_rate=0.01289, steps=200 |
| `real_matched_params` | 7 | 0.2716 | 0.2707 | batch_size=256, hidden_features=16, learning_rate=0.002315, steps=200 |
| `real_matched_params` | 8 | 0.2682 | 0.2644 | batch_size=256, hidden_features=16, learning_rate=0.002142, steps=200 |
| `real_matched_params` | 9 | 0.2486 | 0.2488 | batch_size=128, hidden_features=16, learning_rate=0.001647, steps=200 |
| `real_matched_params` | 10 | 0.3653 | 0.3612 | batch_size=128, hidden_features=32, learning_rate=0.004728, steps=400 |
| `real_matched_params` | 11 | 0.2866 | 0.2892 | batch_size=256, hidden_features=64, learning_rate=0.02619, steps=200 |
| `real_matched_params` | 12 | 0.3367 | 0.3376 | batch_size=128, hidden_features=64, learning_rate=0.008588, steps=400 |
| `real_matched_params` | 13 | 0.3876 | 0.3900 | batch_size=512, hidden_features=32, learning_rate=0.001191, steps=800 |
| `real_matched_params` | 14 | 0.3714 | 0.3694 | batch_size=256, hidden_features=32, learning_rate=0.003545, steps=400 |
| `real_matched_params` | 15 | 0.3077 | 0.3113 | batch_size=128, hidden_features=16, learning_rate=0.002283, steps=400 |
| `real_matched_flops` | 0 | 0.4253 | 0.4259 | batch_size=512, hidden_features=32, learning_rate=0.002444, steps=800 |
| `real_matched_flops` | 1 | 0.4352 | 0.4352 | batch_size=512, hidden_features=16, learning_rate=0.007904, steps=800 |
| `real_matched_flops` | 2 | 0.3009 | 0.3013 | batch_size=256, hidden_features=16, learning_rate=0.02782, steps=200 |
| `real_matched_flops` | 3 | 0.3399 | 0.3358 | batch_size=512, hidden_features=64, learning_rate=0.03977, steps=200 |
| `real_matched_flops` | 4 | 0.3616 | 0.3581 | batch_size=128, hidden_features=32, learning_rate=0.001134, steps=800 |
| `real_matched_flops` | 5 | 0.3499 | 0.3576 | batch_size=256, hidden_features=64, learning_rate=0.02364, steps=400 |
| `real_matched_flops` | 6 | 0.3440 | 0.3437 | batch_size=256, hidden_features=32, learning_rate=0.01289, steps=200 |
| `real_matched_flops` | 7 | 0.2872 | 0.2860 | batch_size=256, hidden_features=16, learning_rate=0.002315, steps=200 |
| `real_matched_flops` | 8 | 0.2835 | 0.2802 | batch_size=256, hidden_features=16, learning_rate=0.002142, steps=200 |
| `real_matched_flops` | 9 | 0.2424 | 0.2430 | batch_size=128, hidden_features=16, learning_rate=0.001647, steps=200 |
| `real_matched_flops` | 10 | 0.3777 | 0.3786 | batch_size=128, hidden_features=32, learning_rate=0.004728, steps=400 |
| `real_matched_flops` | 11 | 0.3321 | 0.3335 | batch_size=256, hidden_features=64, learning_rate=0.02619, steps=200 |
| `real_matched_flops` | 12 | 0.3245 | 0.3190 | batch_size=128, hidden_features=64, learning_rate=0.008588, steps=400 |
| `real_matched_flops` | 13 | 0.3906 | 0.3956 | batch_size=512, hidden_features=32, learning_rate=0.001191, steps=800 |
| `real_matched_flops` | 14 | 0.3889 | 0.3906 | batch_size=256, hidden_features=32, learning_rate=0.003545, steps=400 |
| `real_matched_flops` | 15 | 0.3074 | 0.3089 | batch_size=128, hidden_features=16, learning_rate=0.002283, steps=400 |
| `real_polar` | 0 | 0.4263 | 0.4311 | batch_size=512, hidden_features=32, learning_rate=0.002444, steps=800 |
| `real_polar` | 1 | 0.4341 | 0.4380 | batch_size=512, hidden_features=16, learning_rate=0.007904, steps=800 |
| `real_polar` | 2 | 0.3998 | 0.3919 | batch_size=256, hidden_features=16, learning_rate=0.02782, steps=200 |
| `real_polar` | 3 | 0.3951 | 0.3937 | batch_size=512, hidden_features=64, learning_rate=0.03977, steps=200 |
| `real_polar` | 4 | 0.3755 | 0.3797 | batch_size=128, hidden_features=32, learning_rate=0.001134, steps=800 |
| `real_polar` | 5 | 0.4327 | 0.4298 | batch_size=256, hidden_features=64, learning_rate=0.02364, steps=400 |
| `real_polar` | 6 | 0.3809 | 0.3868 | batch_size=256, hidden_features=32, learning_rate=0.01289, steps=200 |
| `real_polar` | 7 | 0.2935 | 0.2947 | batch_size=256, hidden_features=16, learning_rate=0.002315, steps=200 |
| `real_polar` | 8 | 0.2893 | 0.2867 | batch_size=256, hidden_features=16, learning_rate=0.002142, steps=200 |
| `real_polar` | 9 | 0.2633 | 0.2615 | batch_size=128, hidden_features=16, learning_rate=0.001647, steps=200 |
| `real_polar` | 10 | 0.3888 | 0.3910 | batch_size=128, hidden_features=32, learning_rate=0.004728, steps=400 |
| `real_polar` | 11 | 0.3988 | 0.3915 | batch_size=256, hidden_features=64, learning_rate=0.02619, steps=200 |
| `real_polar` | 12 | 0.4389 | 0.4427 | batch_size=128, hidden_features=64, learning_rate=0.008588, steps=400 |
| `real_polar` | 13 | 0.4121 | 0.4164 | batch_size=512, hidden_features=32, learning_rate=0.001191, steps=800 |
| `real_polar` | 14 | 0.4107 | 0.4085 | batch_size=256, hidden_features=32, learning_rate=0.003545, steps=400 |
| `real_polar` | 15 | 0.3258 | 0.3308 | batch_size=128, hidden_features=16, learning_rate=0.002283, steps=400 |
| `real_phase` | 0 | 0.3410 | 0.3411 | batch_size=512, hidden_features=32, learning_rate=0.002444, steps=800 |
| `real_phase` | 1 | 0.3516 | 0.3564 | batch_size=512, hidden_features=16, learning_rate=0.007904, steps=800 |
| `real_phase` | 2 | 0.2795 | 0.2844 | batch_size=256, hidden_features=16, learning_rate=0.02782, steps=200 |
| `real_phase` | 3 | 0.1681 | 0.1633 | batch_size=512, hidden_features=64, learning_rate=0.03977, steps=200 |
| `real_phase` | 4 | 0.3184 | 0.3210 | batch_size=128, hidden_features=32, learning_rate=0.001134, steps=800 |
| `real_phase` | 5 | 0.2143 | 0.2176 | batch_size=256, hidden_features=64, learning_rate=0.02364, steps=400 |
| `real_phase` | 6 | 0.3091 | 0.3070 | batch_size=256, hidden_features=32, learning_rate=0.01289, steps=200 |
| `real_phase` | 7 | 0.2668 | 0.2708 | batch_size=256, hidden_features=16, learning_rate=0.002315, steps=200 |
| `real_phase` | 8 | 0.2550 | 0.2643 | batch_size=256, hidden_features=16, learning_rate=0.002142, steps=200 |
| `real_phase` | 9 | 0.2289 | 0.2295 | batch_size=128, hidden_features=16, learning_rate=0.001647, steps=200 |
| `real_phase` | 10 | 0.3187 | 0.3205 | batch_size=128, hidden_features=32, learning_rate=0.004728, steps=400 |
| `real_phase` | 11 | 0.1833 | 0.1906 | batch_size=256, hidden_features=64, learning_rate=0.02619, steps=200 |
| `real_phase` | 12 | 0.3279 | 0.3333 | batch_size=128, hidden_features=64, learning_rate=0.008588, steps=400 |
| `real_phase` | 13 | 0.3328 | 0.3321 | batch_size=512, hidden_features=32, learning_rate=0.001191, steps=800 |
| `real_phase` | 14 | 0.3225 | 0.3206 | batch_size=256, hidden_features=32, learning_rate=0.003545, steps=400 |
| `real_phase` | 15 | 0.3036 | 0.3068 | batch_size=128, hidden_features=16, learning_rate=0.002283, steps=400 |
| `real_magnitude` | 0 | 0.3997 | 0.3943 | batch_size=512, hidden_features=32, learning_rate=0.002444, steps=800 |
| `real_magnitude` | 1 | 0.3942 | 0.3887 | batch_size=512, hidden_features=16, learning_rate=0.007904, steps=800 |
| `real_magnitude` | 2 | 0.3267 | 0.3302 | batch_size=256, hidden_features=16, learning_rate=0.02782, steps=200 |
| `real_magnitude` | 3 | 0.3128 | 0.3204 | batch_size=512, hidden_features=64, learning_rate=0.03977, steps=200 |
| `real_magnitude` | 4 | 0.3388 | 0.3409 | batch_size=128, hidden_features=32, learning_rate=0.001134, steps=800 |
| `real_magnitude` | 5 | 0.4016 | 0.3980 | batch_size=256, hidden_features=64, learning_rate=0.02364, steps=400 |
| `real_magnitude` | 6 | 0.3706 | 0.3676 | batch_size=256, hidden_features=32, learning_rate=0.01289, steps=200 |
| `real_magnitude` | 7 | 0.2791 | 0.2761 | batch_size=256, hidden_features=16, learning_rate=0.002315, steps=200 |
| `real_magnitude` | 8 | 0.2746 | 0.2705 | batch_size=256, hidden_features=16, learning_rate=0.002142, steps=200 |
| `real_magnitude` | 9 | 0.2448 | 0.2476 | batch_size=128, hidden_features=16, learning_rate=0.001647, steps=200 |
| `real_magnitude` | 10 | 0.3450 | 0.3443 | batch_size=128, hidden_features=32, learning_rate=0.004728, steps=400 |
| `real_magnitude` | 11 | 0.3743 | 0.3694 | batch_size=256, hidden_features=64, learning_rate=0.02619, steps=200 |
| `real_magnitude` | 12 | 0.3958 | 0.3941 | batch_size=128, hidden_features=64, learning_rate=0.008588, steps=400 |
| `real_magnitude` | 13 | 0.3750 | 0.3786 | batch_size=512, hidden_features=32, learning_rate=0.001191, steps=800 |
| `real_magnitude` | 14 | 0.3623 | 0.3611 | batch_size=256, hidden_features=32, learning_rate=0.003545, steps=400 |
| `real_magnitude` | 15 | 0.2960 | 0.2944 | batch_size=128, hidden_features=16, learning_rate=0.002283, steps=400 |
