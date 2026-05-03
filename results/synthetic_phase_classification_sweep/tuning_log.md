# Tuning Log - Synthetic Phase Classification

Random-search sweep following `docs/tuning_budget.md`: shared trial samples across all families, seeded per trial, selection by mean validation accuracy.

- Trials per family: `16`
- Seeds per trial: `[0, 1, 2]`
- Sweep seed: `20260503`
- Search space: see `trials.json`

## Selected configuration per family

| family | trial | val acc | test acc | test std | hyperparameters |
| --- | ---: | ---: | ---: | ---: | --- |
| `complex` | 4 | 0.7870 | 0.7656 | 0.0145 | hidden_features=8, learning_rate=0.005517, steps=400 |
| `real_stacked` | 8 | 0.7854 | 0.7643 | 0.0156 | hidden_features=16, learning_rate=0.01693, steps=200 |
| `real_matched_params` | 7 | 0.7821 | 0.7630 | 0.0006 | hidden_features=32, learning_rate=0.03542, steps=200 |
| `real_matched_flops` | 11 | 0.7837 | 0.7650 | 0.0153 | hidden_features=8, learning_rate=0.007326, steps=200 |

## All trials (mean across seeds)

| family | trial | val acc | test acc | hyperparameters |
| --- | ---: | ---: | ---: | --- |
| `complex` | 0 | 0.7659 | 0.7552 | hidden_features=64, learning_rate=0.009913, steps=800 |
| `complex` | 1 | 0.7659 | 0.7565 | hidden_features=8, learning_rate=0.0272, steps=800 |
| `complex` | 2 | 0.7724 | 0.7607 | hidden_features=8, learning_rate=0.04144, steps=200 |
| `complex` | 3 | 0.7805 | 0.7604 | hidden_features=8, learning_rate=0.02208, steps=800 |
| `complex` | 4 | 0.7870 | 0.7656 | hidden_features=8, learning_rate=0.005517, steps=400 |
| `complex` | 5 | 0.7626 | 0.7562 | hidden_features=16, learning_rate=0.06592, steps=400 |
| `complex` | 6 | 0.7626 | 0.7493 | hidden_features=64, learning_rate=0.05634, steps=400 |
| `complex` | 7 | 0.7724 | 0.7620 | hidden_features=32, learning_rate=0.03542, steps=200 |
| `complex` | 8 | 0.7772 | 0.7604 | hidden_features=16, learning_rate=0.01693, steps=200 |
| `complex` | 9 | 0.7837 | 0.7617 | hidden_features=16, learning_rate=0.00972, steps=400 |
| `complex` | 10 | 0.7854 | 0.7650 | hidden_features=8, learning_rate=0.005719, steps=400 |
| `complex` | 11 | 0.7740 | 0.7663 | hidden_features=8, learning_rate=0.007326, steps=200 |
| `complex` | 12 | 0.7805 | 0.7627 | hidden_features=16, learning_rate=0.007251, steps=400 |
| `complex` | 13 | 0.7821 | 0.7611 | hidden_features=8, learning_rate=0.02007, steps=400 |
| `complex` | 14 | 0.7675 | 0.7614 | hidden_features=32, learning_rate=0.02595, steps=200 |
| `complex` | 15 | 0.7740 | 0.7591 | hidden_features=64, learning_rate=0.005715, steps=800 |
| `real_stacked` | 0 | 0.7740 | 0.7552 | hidden_features=64, learning_rate=0.009913, steps=800 |
| `real_stacked` | 1 | 0.7740 | 0.7640 | hidden_features=8, learning_rate=0.0272, steps=800 |
| `real_stacked` | 2 | 0.7772 | 0.7660 | hidden_features=8, learning_rate=0.04144, steps=200 |
| `real_stacked` | 3 | 0.7854 | 0.7650 | hidden_features=8, learning_rate=0.02208, steps=800 |
| `real_stacked` | 4 | 0.7772 | 0.7630 | hidden_features=8, learning_rate=0.005517, steps=400 |
| `real_stacked` | 5 | 0.7642 | 0.7565 | hidden_features=16, learning_rate=0.06592, steps=400 |
| `real_stacked` | 6 | 0.7675 | 0.7529 | hidden_features=64, learning_rate=0.05634, steps=400 |
| `real_stacked` | 7 | 0.7756 | 0.7578 | hidden_features=32, learning_rate=0.03542, steps=200 |
| `real_stacked` | 8 | 0.7854 | 0.7643 | hidden_features=16, learning_rate=0.01693, steps=200 |
| `real_stacked` | 9 | 0.7837 | 0.7650 | hidden_features=16, learning_rate=0.00972, steps=400 |
| `real_stacked` | 10 | 0.7756 | 0.7627 | hidden_features=8, learning_rate=0.005719, steps=400 |
| `real_stacked` | 11 | 0.7756 | 0.7565 | hidden_features=8, learning_rate=0.007326, steps=200 |
| `real_stacked` | 12 | 0.7854 | 0.7633 | hidden_features=16, learning_rate=0.007251, steps=400 |
| `real_stacked` | 13 | 0.7805 | 0.7666 | hidden_features=8, learning_rate=0.02007, steps=400 |
| `real_stacked` | 14 | 0.7724 | 0.7572 | hidden_features=32, learning_rate=0.02595, steps=200 |
| `real_stacked` | 15 | 0.7821 | 0.7594 | hidden_features=64, learning_rate=0.005715, steps=800 |
| `real_matched_params` | 0 | 0.7724 | 0.7523 | hidden_features=64, learning_rate=0.009913, steps=800 |
| `real_matched_params` | 1 | 0.7691 | 0.7611 | hidden_features=8, learning_rate=0.0272, steps=800 |
| `real_matched_params` | 2 | 0.7789 | 0.7640 | hidden_features=8, learning_rate=0.04144, steps=200 |
| `real_matched_params` | 3 | 0.7805 | 0.7578 | hidden_features=8, learning_rate=0.02208, steps=800 |
| `real_matched_params` | 4 | 0.7805 | 0.7666 | hidden_features=8, learning_rate=0.005517, steps=400 |
| `real_matched_params` | 5 | 0.7610 | 0.7516 | hidden_features=16, learning_rate=0.06592, steps=400 |
| `real_matched_params` | 6 | 0.7642 | 0.7529 | hidden_features=64, learning_rate=0.05634, steps=400 |
| `real_matched_params` | 7 | 0.7821 | 0.7630 | hidden_features=32, learning_rate=0.03542, steps=200 |
| `real_matched_params` | 8 | 0.7789 | 0.7633 | hidden_features=16, learning_rate=0.01693, steps=200 |
| `real_matched_params` | 9 | 0.7805 | 0.7617 | hidden_features=16, learning_rate=0.00972, steps=400 |
| `real_matched_params` | 10 | 0.7805 | 0.7653 | hidden_features=8, learning_rate=0.005719, steps=400 |
| `real_matched_params` | 11 | 0.7756 | 0.7656 | hidden_features=8, learning_rate=0.007326, steps=200 |
| `real_matched_params` | 12 | 0.7740 | 0.7646 | hidden_features=16, learning_rate=0.007251, steps=400 |
| `real_matched_params` | 13 | 0.7789 | 0.7646 | hidden_features=8, learning_rate=0.02007, steps=400 |
| `real_matched_params` | 14 | 0.7772 | 0.7604 | hidden_features=32, learning_rate=0.02595, steps=200 |
| `real_matched_params` | 15 | 0.7691 | 0.7555 | hidden_features=64, learning_rate=0.005715, steps=800 |
| `real_matched_flops` | 0 | 0.7512 | 0.7451 | hidden_features=64, learning_rate=0.009913, steps=800 |
| `real_matched_flops` | 1 | 0.7642 | 0.7562 | hidden_features=8, learning_rate=0.0272, steps=800 |
| `real_matched_flops` | 2 | 0.7772 | 0.7614 | hidden_features=8, learning_rate=0.04144, steps=200 |
| `real_matched_flops` | 3 | 0.7724 | 0.7594 | hidden_features=8, learning_rate=0.02208, steps=800 |
| `real_matched_flops` | 4 | 0.7821 | 0.7620 | hidden_features=8, learning_rate=0.005517, steps=400 |
| `real_matched_flops` | 5 | 0.7659 | 0.7510 | hidden_features=16, learning_rate=0.06592, steps=400 |
| `real_matched_flops` | 6 | 0.7626 | 0.7451 | hidden_features=64, learning_rate=0.05634, steps=400 |
| `real_matched_flops` | 7 | 0.7691 | 0.7572 | hidden_features=32, learning_rate=0.03542, steps=200 |
| `real_matched_flops` | 8 | 0.7724 | 0.7620 | hidden_features=16, learning_rate=0.01693, steps=200 |
| `real_matched_flops` | 9 | 0.7805 | 0.7620 | hidden_features=16, learning_rate=0.00972, steps=400 |
| `real_matched_flops` | 10 | 0.7821 | 0.7617 | hidden_features=8, learning_rate=0.005719, steps=400 |
| `real_matched_flops` | 11 | 0.7837 | 0.7650 | hidden_features=8, learning_rate=0.007326, steps=200 |
| `real_matched_flops` | 12 | 0.7756 | 0.7624 | hidden_features=16, learning_rate=0.007251, steps=400 |
| `real_matched_flops` | 13 | 0.7805 | 0.7581 | hidden_features=8, learning_rate=0.02007, steps=400 |
| `real_matched_flops` | 14 | 0.7691 | 0.7591 | hidden_features=32, learning_rate=0.02595, steps=200 |
| `real_matched_flops` | 15 | 0.7740 | 0.7513 | hidden_features=64, learning_rate=0.005715, steps=800 |
