# Tuning Log - RadioML 2018.01A Modulation Classification

Random-search sweep following `docs/tuning_budget.md`: shared trial samples across all families, seeded per trial, selection by mean validation accuracy.

- Trials per family: `16`
- Seeds per trial: `[0, 1, 2, 3, 4]`
- Sweep seed: `20260503`
- Search space: see `trials.json`

## Selected configuration per family

| family | trial | val acc | test acc | test std | hyperparameters |
| --- | ---: | ---: | ---: | ---: | --- |
| `complex` | 5 | 0.4191 | 0.4098 | 0.0116 | batch_size=256, hidden_features=64, learning_rate=0.02364, steps=400 |
| `real_stacked` | 5 | 0.3292 | 0.3287 | 0.0221 | batch_size=256, hidden_features=64, learning_rate=0.02364, steps=400 |
| `real_matched_params` | 5 | 0.3302 | 0.3299 | 0.0305 | batch_size=256, hidden_features=64, learning_rate=0.02364, steps=400 |
| `real_matched_flops` | 5 | 0.3466 | 0.3465 | 0.0523 | batch_size=256, hidden_features=64, learning_rate=0.02364, steps=400 |
| `real_polar` | 5 | 0.4236 | 0.4220 | 0.0253 | batch_size=256, hidden_features=64, learning_rate=0.02364, steps=400 |
| `real_phase` | 5 | 0.2142 | 0.2203 | 0.0788 | batch_size=256, hidden_features=64, learning_rate=0.02364, steps=400 |
| `real_magnitude` | 5 | 0.4005 | 0.3991 | 0.0138 | batch_size=256, hidden_features=64, learning_rate=0.02364, steps=400 |

## All trials (mean across seeds)

| family | trial | val acc | test acc | hyperparameters |
| --- | ---: | ---: | ---: | --- |
| `complex` | 0 | 0.4019 | 0.4065 | batch_size=512, hidden_features=32, learning_rate=0.002444, steps=800 |
| `complex` | 1 | 0.3935 | 0.3940 | batch_size=512, hidden_features=16, learning_rate=0.007904, steps=800 |
| `complex` | 2 | 0.3290 | 0.3346 | batch_size=256, hidden_features=16, learning_rate=0.02782, steps=200 |
| `complex` | 3 | 0.3837 | 0.3780 | batch_size=512, hidden_features=64, learning_rate=0.03977, steps=200 |
| `complex` | 4 | 0.3490 | 0.3495 | batch_size=128, hidden_features=32, learning_rate=0.001134, steps=800 |
| `complex` | 5 | 0.4191 | 0.4098 | batch_size=256, hidden_features=64, learning_rate=0.02364, steps=400 |
| `complex` | 6 | 0.3500 | 0.3549 | batch_size=256, hidden_features=32, learning_rate=0.01289, steps=200 |
| `complex` | 7 | 0.3230 | 0.3310 | batch_size=256, hidden_features=16, learning_rate=0.002315, steps=200 |
| `complex` | 8 | 0.3202 | 0.3280 | batch_size=256, hidden_features=16, learning_rate=0.002142, steps=200 |
| `complex` | 9 | 0.3158 | 0.3165 | batch_size=128, hidden_features=16, learning_rate=0.001647, steps=200 |
| `complex` | 10 | 0.3574 | 0.3627 | batch_size=128, hidden_features=32, learning_rate=0.004728, steps=400 |
| `complex` | 11 | 0.3663 | 0.3623 | batch_size=256, hidden_features=64, learning_rate=0.02619, steps=200 |
| `complex` | 12 | 0.3705 | 0.3735 | batch_size=128, hidden_features=64, learning_rate=0.008588, steps=400 |
| `complex` | 13 | 0.3788 | 0.3814 | batch_size=512, hidden_features=32, learning_rate=0.001191, steps=800 |
| `complex` | 14 | 0.3665 | 0.3640 | batch_size=256, hidden_features=32, learning_rate=0.003545, steps=400 |
| `complex` | 15 | 0.3325 | 0.3380 | batch_size=128, hidden_features=16, learning_rate=0.002283, steps=400 |
| `real_stacked` | 0 | 0.3998 | 0.4043 | batch_size=512, hidden_features=32, learning_rate=0.002444, steps=800 |
| `real_stacked` | 1 | 0.4002 | 0.4016 | batch_size=512, hidden_features=16, learning_rate=0.007904, steps=800 |
| `real_stacked` | 2 | 0.3002 | 0.2995 | batch_size=256, hidden_features=16, learning_rate=0.02782, steps=200 |
| `real_stacked` | 3 | 0.3387 | 0.3391 | batch_size=512, hidden_features=64, learning_rate=0.03977, steps=200 |
| `real_stacked` | 4 | 0.3182 | 0.3227 | batch_size=128, hidden_features=32, learning_rate=0.001134, steps=800 |
| `real_stacked` | 5 | 0.3292 | 0.3287 | batch_size=256, hidden_features=64, learning_rate=0.02364, steps=400 |
| `real_stacked` | 6 | 0.3302 | 0.3299 | batch_size=256, hidden_features=32, learning_rate=0.01289, steps=200 |
| `real_stacked` | 7 | 0.2539 | 0.2518 | batch_size=256, hidden_features=16, learning_rate=0.002315, steps=200 |
| `real_stacked` | 8 | 0.2530 | 0.2500 | batch_size=256, hidden_features=16, learning_rate=0.002142, steps=200 |
| `real_stacked` | 9 | 0.2325 | 0.2329 | batch_size=128, hidden_features=16, learning_rate=0.001647, steps=200 |
| `real_stacked` | 10 | 0.3472 | 0.3405 | batch_size=128, hidden_features=32, learning_rate=0.004728, steps=400 |
| `real_stacked` | 11 | 0.2817 | 0.2906 | batch_size=256, hidden_features=64, learning_rate=0.02619, steps=200 |
| `real_stacked` | 12 | 0.3787 | 0.3817 | batch_size=128, hidden_features=64, learning_rate=0.008588, steps=400 |
| `real_stacked` | 13 | 0.3616 | 0.3653 | batch_size=512, hidden_features=32, learning_rate=0.001191, steps=800 |
| `real_stacked` | 14 | 0.3500 | 0.3588 | batch_size=256, hidden_features=32, learning_rate=0.003545, steps=400 |
| `real_stacked` | 15 | 0.2554 | 0.2508 | batch_size=128, hidden_features=16, learning_rate=0.002283, steps=400 |
| `real_matched_params` | 0 | 0.4177 | 0.4137 | batch_size=512, hidden_features=32, learning_rate=0.002444, steps=800 |
| `real_matched_params` | 1 | 0.4211 | 0.4238 | batch_size=512, hidden_features=16, learning_rate=0.007904, steps=800 |
| `real_matched_params` | 2 | 0.3098 | 0.3164 | batch_size=256, hidden_features=16, learning_rate=0.02782, steps=200 |
| `real_matched_params` | 3 | 0.3496 | 0.3605 | batch_size=512, hidden_features=64, learning_rate=0.03977, steps=200 |
| `real_matched_params` | 4 | 0.3419 | 0.3407 | batch_size=128, hidden_features=32, learning_rate=0.001134, steps=800 |
| `real_matched_params` | 5 | 0.3302 | 0.3299 | batch_size=256, hidden_features=64, learning_rate=0.02364, steps=400 |
| `real_matched_params` | 6 | 0.3472 | 0.3502 | batch_size=256, hidden_features=32, learning_rate=0.01289, steps=200 |
| `real_matched_params` | 7 | 0.2723 | 0.2712 | batch_size=256, hidden_features=16, learning_rate=0.002315, steps=200 |
| `real_matched_params` | 8 | 0.2700 | 0.2661 | batch_size=256, hidden_features=16, learning_rate=0.002142, steps=200 |
| `real_matched_params` | 9 | 0.2486 | 0.2488 | batch_size=128, hidden_features=16, learning_rate=0.001647, steps=200 |
| `real_matched_params` | 10 | 0.3653 | 0.3612 | batch_size=128, hidden_features=32, learning_rate=0.004728, steps=400 |
| `real_matched_params` | 11 | 0.2767 | 0.2801 | batch_size=256, hidden_features=64, learning_rate=0.02619, steps=200 |
| `real_matched_params` | 12 | 0.3383 | 0.3381 | batch_size=128, hidden_features=64, learning_rate=0.008588, steps=400 |
| `real_matched_params` | 13 | 0.3891 | 0.3913 | batch_size=512, hidden_features=32, learning_rate=0.001191, steps=800 |
| `real_matched_params` | 14 | 0.3714 | 0.3690 | batch_size=256, hidden_features=32, learning_rate=0.003545, steps=400 |
| `real_matched_params` | 15 | 0.3077 | 0.3113 | batch_size=128, hidden_features=16, learning_rate=0.002283, steps=400 |
| `real_matched_flops` | 0 | 0.4277 | 0.4264 | batch_size=512, hidden_features=32, learning_rate=0.002444, steps=800 |
| `real_matched_flops` | 1 | 0.4354 | 0.4355 | batch_size=512, hidden_features=16, learning_rate=0.007904, steps=800 |
| `real_matched_flops` | 2 | 0.3093 | 0.3107 | batch_size=256, hidden_features=16, learning_rate=0.02782, steps=200 |
| `real_matched_flops` | 3 | 0.3188 | 0.3263 | batch_size=512, hidden_features=64, learning_rate=0.03977, steps=200 |
| `real_matched_flops` | 4 | 0.3616 | 0.3581 | batch_size=128, hidden_features=32, learning_rate=0.001134, steps=800 |
| `real_matched_flops` | 5 | 0.3466 | 0.3465 | batch_size=256, hidden_features=64, learning_rate=0.02364, steps=400 |
| `real_matched_flops` | 6 | 0.3409 | 0.3399 | batch_size=256, hidden_features=32, learning_rate=0.01289, steps=200 |
| `real_matched_flops` | 7 | 0.2870 | 0.2864 | batch_size=256, hidden_features=16, learning_rate=0.002315, steps=200 |
| `real_matched_flops` | 8 | 0.2835 | 0.2802 | batch_size=256, hidden_features=16, learning_rate=0.002142, steps=200 |
| `real_matched_flops` | 9 | 0.2424 | 0.2430 | batch_size=128, hidden_features=16, learning_rate=0.001647, steps=200 |
| `real_matched_flops` | 10 | 0.3777 | 0.3786 | batch_size=128, hidden_features=32, learning_rate=0.004728, steps=400 |
| `real_matched_flops` | 11 | 0.3318 | 0.3317 | batch_size=256, hidden_features=64, learning_rate=0.02619, steps=200 |
| `real_matched_flops` | 12 | 0.3148 | 0.3162 | batch_size=128, hidden_features=64, learning_rate=0.008588, steps=400 |
| `real_matched_flops` | 13 | 0.3908 | 0.3962 | batch_size=512, hidden_features=32, learning_rate=0.001191, steps=800 |
| `real_matched_flops` | 14 | 0.3869 | 0.3896 | batch_size=256, hidden_features=32, learning_rate=0.003545, steps=400 |
| `real_matched_flops` | 15 | 0.3074 | 0.3089 | batch_size=128, hidden_features=16, learning_rate=0.002283, steps=400 |
| `real_polar` | 0 | 0.4259 | 0.4314 | batch_size=512, hidden_features=32, learning_rate=0.002444, steps=800 |
| `real_polar` | 1 | 0.4341 | 0.4380 | batch_size=512, hidden_features=16, learning_rate=0.007904, steps=800 |
| `real_polar` | 2 | 0.3998 | 0.3919 | batch_size=256, hidden_features=16, learning_rate=0.02782, steps=200 |
| `real_polar` | 3 | 0.4085 | 0.4051 | batch_size=512, hidden_features=64, learning_rate=0.03977, steps=200 |
| `real_polar` | 4 | 0.3755 | 0.3797 | batch_size=128, hidden_features=32, learning_rate=0.001134, steps=800 |
| `real_polar` | 5 | 0.4236 | 0.4220 | batch_size=256, hidden_features=64, learning_rate=0.02364, steps=400 |
| `real_polar` | 6 | 0.3797 | 0.3856 | batch_size=256, hidden_features=32, learning_rate=0.01289, steps=200 |
| `real_polar` | 7 | 0.2935 | 0.2947 | batch_size=256, hidden_features=16, learning_rate=0.002315, steps=200 |
| `real_polar` | 8 | 0.2893 | 0.2867 | batch_size=256, hidden_features=16, learning_rate=0.002142, steps=200 |
| `real_polar` | 9 | 0.2633 | 0.2615 | batch_size=128, hidden_features=16, learning_rate=0.001647, steps=200 |
| `real_polar` | 10 | 0.3888 | 0.3910 | batch_size=128, hidden_features=32, learning_rate=0.004728, steps=400 |
| `real_polar` | 11 | 0.3929 | 0.3889 | batch_size=256, hidden_features=64, learning_rate=0.02619, steps=200 |
| `real_polar` | 12 | 0.4389 | 0.4427 | batch_size=128, hidden_features=64, learning_rate=0.008588, steps=400 |
| `real_polar` | 13 | 0.4116 | 0.4136 | batch_size=512, hidden_features=32, learning_rate=0.001191, steps=800 |
| `real_polar` | 14 | 0.4108 | 0.4077 | batch_size=256, hidden_features=32, learning_rate=0.003545, steps=400 |
| `real_polar` | 15 | 0.3258 | 0.3308 | batch_size=128, hidden_features=16, learning_rate=0.002283, steps=400 |
| `real_phase` | 0 | 0.3405 | 0.3419 | batch_size=512, hidden_features=32, learning_rate=0.002444, steps=800 |
| `real_phase` | 1 | 0.3516 | 0.3564 | batch_size=512, hidden_features=16, learning_rate=0.007904, steps=800 |
| `real_phase` | 2 | 0.2795 | 0.2844 | batch_size=256, hidden_features=16, learning_rate=0.02782, steps=200 |
| `real_phase` | 3 | 0.1604 | 0.1584 | batch_size=512, hidden_features=64, learning_rate=0.03977, steps=200 |
| `real_phase` | 4 | 0.3184 | 0.3210 | batch_size=128, hidden_features=32, learning_rate=0.001134, steps=800 |
| `real_phase` | 5 | 0.2142 | 0.2203 | batch_size=256, hidden_features=64, learning_rate=0.02364, steps=400 |
| `real_phase` | 6 | 0.3096 | 0.3085 | batch_size=256, hidden_features=32, learning_rate=0.01289, steps=200 |
| `real_phase` | 7 | 0.2668 | 0.2708 | batch_size=256, hidden_features=16, learning_rate=0.002315, steps=200 |
| `real_phase` | 8 | 0.2550 | 0.2643 | batch_size=256, hidden_features=16, learning_rate=0.002142, steps=200 |
| `real_phase` | 9 | 0.2289 | 0.2295 | batch_size=128, hidden_features=16, learning_rate=0.001647, steps=200 |
| `real_phase` | 10 | 0.3187 | 0.3205 | batch_size=128, hidden_features=32, learning_rate=0.004728, steps=400 |
| `real_phase` | 11 | 0.1760 | 0.1831 | batch_size=256, hidden_features=64, learning_rate=0.02619, steps=200 |
| `real_phase` | 12 | 0.3279 | 0.3333 | batch_size=128, hidden_features=64, learning_rate=0.008588, steps=400 |
| `real_phase` | 13 | 0.3313 | 0.3313 | batch_size=512, hidden_features=32, learning_rate=0.001191, steps=800 |
| `real_phase` | 14 | 0.3237 | 0.3209 | batch_size=256, hidden_features=32, learning_rate=0.003545, steps=400 |
| `real_phase` | 15 | 0.3036 | 0.3068 | batch_size=128, hidden_features=16, learning_rate=0.002283, steps=400 |
| `real_magnitude` | 0 | 0.3991 | 0.3927 | batch_size=512, hidden_features=32, learning_rate=0.002444, steps=800 |
| `real_magnitude` | 1 | 0.3942 | 0.3887 | batch_size=512, hidden_features=16, learning_rate=0.007904, steps=800 |
| `real_magnitude` | 2 | 0.3267 | 0.3302 | batch_size=256, hidden_features=16, learning_rate=0.02782, steps=200 |
| `real_magnitude` | 3 | 0.3176 | 0.3189 | batch_size=512, hidden_features=64, learning_rate=0.03977, steps=200 |
| `real_magnitude` | 4 | 0.3388 | 0.3409 | batch_size=128, hidden_features=32, learning_rate=0.001134, steps=800 |
| `real_magnitude` | 5 | 0.4005 | 0.3991 | batch_size=256, hidden_features=64, learning_rate=0.02364, steps=400 |
| `real_magnitude` | 6 | 0.3625 | 0.3600 | batch_size=256, hidden_features=32, learning_rate=0.01289, steps=200 |
| `real_magnitude` | 7 | 0.2791 | 0.2761 | batch_size=256, hidden_features=16, learning_rate=0.002315, steps=200 |
| `real_magnitude` | 8 | 0.2746 | 0.2705 | batch_size=256, hidden_features=16, learning_rate=0.002142, steps=200 |
| `real_magnitude` | 9 | 0.2448 | 0.2476 | batch_size=128, hidden_features=16, learning_rate=0.001647, steps=200 |
| `real_magnitude` | 10 | 0.3450 | 0.3443 | batch_size=128, hidden_features=32, learning_rate=0.004728, steps=400 |
| `real_magnitude` | 11 | 0.3898 | 0.3830 | batch_size=256, hidden_features=64, learning_rate=0.02619, steps=200 |
| `real_magnitude` | 12 | 0.3958 | 0.3941 | batch_size=128, hidden_features=64, learning_rate=0.008588, steps=400 |
| `real_magnitude` | 13 | 0.3749 | 0.3786 | batch_size=512, hidden_features=32, learning_rate=0.001191, steps=800 |
| `real_magnitude` | 14 | 0.3603 | 0.3600 | batch_size=256, hidden_features=32, learning_rate=0.003545, steps=400 |
| `real_magnitude` | 15 | 0.2960 | 0.2944 | batch_size=128, hidden_features=16, learning_rate=0.002283, steps=400 |
