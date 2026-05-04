# Tuning Log - RadioML 2018.01A Modulation Classification

Random-search sweep following `docs/tuning_budget.md`: shared trial samples across all families, seeded per trial, selection by mean validation accuracy.

- Trials per family: `16`
- Seeds per trial: `[0, 1, 2, 3, 4, 5]`
- Sweep seed: `20260503`
- Search space: see `trials.json`

## Selected configuration per family

| family | trial | val acc | test acc | test std | hyperparameters |
| --- | ---: | ---: | ---: | ---: | --- |
| `complex` | 5 | 0.7095 | 0.7217 | 0.0229 | batch_size=256, hidden_features=64, learning_rate=0.02364, steps=400 |
| `real_stacked` | 5 | 0.4976 | 0.5112 | 0.1383 | batch_size=256, hidden_features=64, learning_rate=0.02364, steps=400 |
| `real_matched_params` | 5 | 0.4255 | 0.4239 | 0.1411 | batch_size=256, hidden_features=64, learning_rate=0.02364, steps=400 |
| `real_matched_flops` | 5 | 0.4490 | 0.4458 | 0.1233 | batch_size=256, hidden_features=64, learning_rate=0.02364, steps=400 |

## All trials (mean across seeds)

| family | trial | val acc | test acc | hyperparameters |
| --- | ---: | ---: | ---: | --- |
| `complex` | 0 | 0.6452 | 0.6426 | batch_size=512, hidden_features=32, learning_rate=0.002444, steps=800 |
| `complex` | 1 | 0.6459 | 0.6509 | batch_size=512, hidden_features=16, learning_rate=0.007904, steps=800 |
| `complex` | 2 | 0.6180 | 0.6255 | batch_size=256, hidden_features=16, learning_rate=0.02782, steps=200 |
| `complex` | 3 | 0.6449 | 0.6322 | batch_size=512, hidden_features=64, learning_rate=0.03977, steps=200 |
| `complex` | 4 | 0.5891 | 0.5916 | batch_size=128, hidden_features=32, learning_rate=0.001134, steps=800 |
| `complex` | 5 | 0.7095 | 0.7217 | batch_size=256, hidden_features=64, learning_rate=0.02364, steps=400 |
| `complex` | 6 | 0.6388 | 0.6432 | batch_size=256, hidden_features=32, learning_rate=0.01289, steps=200 |
| `complex` | 7 | 0.5752 | 0.5857 | batch_size=256, hidden_features=16, learning_rate=0.002315, steps=200 |
| `complex` | 8 | 0.5776 | 0.5852 | batch_size=256, hidden_features=16, learning_rate=0.002142, steps=200 |
| `complex` | 9 | 0.5786 | 0.5828 | batch_size=128, hidden_features=16, learning_rate=0.001647, steps=200 |
| `complex` | 10 | 0.6068 | 0.6079 | batch_size=128, hidden_features=32, learning_rate=0.004728, steps=400 |
| `complex` | 11 | 0.6422 | 0.6440 | batch_size=256, hidden_features=64, learning_rate=0.02619, steps=200 |
| `complex` | 12 | 0.6388 | 0.6378 | batch_size=128, hidden_features=64, learning_rate=0.008588, steps=400 |
| `complex` | 13 | 0.6303 | 0.6408 | batch_size=512, hidden_features=32, learning_rate=0.001191, steps=800 |
| `complex` | 14 | 0.6180 | 0.6287 | batch_size=256, hidden_features=32, learning_rate=0.003545, steps=400 |
| `complex` | 15 | 0.5932 | 0.5978 | batch_size=128, hidden_features=16, learning_rate=0.002283, steps=400 |
| `real_stacked` | 0 | 0.6670 | 0.6803 | batch_size=512, hidden_features=32, learning_rate=0.002444, steps=800 |
| `real_stacked` | 1 | 0.6633 | 0.6693 | batch_size=512, hidden_features=16, learning_rate=0.007904, steps=800 |
| `real_stacked` | 2 | 0.5313 | 0.5235 | batch_size=256, hidden_features=16, learning_rate=0.02782, steps=200 |
| `real_stacked` | 3 | 0.5238 | 0.5267 | batch_size=512, hidden_features=64, learning_rate=0.03977, steps=200 |
| `real_stacked` | 4 | 0.5969 | 0.5887 | batch_size=128, hidden_features=32, learning_rate=0.001134, steps=800 |
| `real_stacked` | 5 | 0.4976 | 0.5112 | batch_size=256, hidden_features=64, learning_rate=0.02364, steps=400 |
| `real_stacked` | 6 | 0.4929 | 0.4866 | batch_size=256, hidden_features=32, learning_rate=0.01289, steps=200 |
| `real_stacked` | 7 | 0.5252 | 0.5315 | batch_size=256, hidden_features=16, learning_rate=0.002315, steps=200 |
| `real_stacked` | 8 | 0.5194 | 0.5235 | batch_size=256, hidden_features=16, learning_rate=0.002142, steps=200 |
| `real_stacked` | 9 | 0.4463 | 0.4447 | batch_size=128, hidden_features=16, learning_rate=0.001647, steps=200 |
| `real_stacked` | 10 | 0.5731 | 0.5780 | batch_size=128, hidden_features=32, learning_rate=0.004728, steps=400 |
| `real_stacked` | 11 | 0.4796 | 0.4837 | batch_size=256, hidden_features=64, learning_rate=0.02619, steps=200 |
| `real_stacked` | 12 | 0.5245 | 0.5232 | batch_size=128, hidden_features=64, learning_rate=0.008588, steps=400 |
| `real_stacked` | 13 | 0.6109 | 0.6167 | batch_size=512, hidden_features=32, learning_rate=0.001191, steps=800 |
| `real_stacked` | 14 | 0.6214 | 0.6178 | batch_size=256, hidden_features=32, learning_rate=0.003545, steps=400 |
| `real_stacked` | 15 | 0.5514 | 0.5564 | batch_size=128, hidden_features=16, learning_rate=0.002283, steps=400 |
| `real_matched_params` | 0 | 0.6888 | 0.6968 | batch_size=512, hidden_features=32, learning_rate=0.002444, steps=800 |
| `real_matched_params` | 1 | 0.6697 | 0.6832 | batch_size=512, hidden_features=16, learning_rate=0.007904, steps=800 |
| `real_matched_params` | 2 | 0.4956 | 0.4955 | batch_size=256, hidden_features=16, learning_rate=0.02782, steps=200 |
| `real_matched_params` | 3 | 0.5126 | 0.5013 | batch_size=512, hidden_features=64, learning_rate=0.03977, steps=200 |
| `real_matched_params` | 4 | 0.6109 | 0.6068 | batch_size=128, hidden_features=32, learning_rate=0.001134, steps=800 |
| `real_matched_params` | 5 | 0.4255 | 0.4239 | batch_size=256, hidden_features=64, learning_rate=0.02364, steps=400 |
| `real_matched_params` | 6 | 0.5221 | 0.5256 | batch_size=256, hidden_features=32, learning_rate=0.01289, steps=200 |
| `real_matched_params` | 7 | 0.5391 | 0.5483 | batch_size=256, hidden_features=16, learning_rate=0.002315, steps=200 |
| `real_matched_params` | 8 | 0.5357 | 0.5473 | batch_size=256, hidden_features=16, learning_rate=0.002142, steps=200 |
| `real_matched_params` | 9 | 0.5187 | 0.5168 | batch_size=128, hidden_features=16, learning_rate=0.001647, steps=200 |
| `real_matched_params` | 10 | 0.5677 | 0.5684 | batch_size=128, hidden_features=32, learning_rate=0.004728, steps=400 |
| `real_matched_params` | 11 | 0.4102 | 0.4044 | batch_size=256, hidden_features=64, learning_rate=0.02619, steps=200 |
| `real_matched_params` | 12 | 0.4486 | 0.4607 | batch_size=128, hidden_features=64, learning_rate=0.008588, steps=400 |
| `real_matched_params` | 13 | 0.6476 | 0.6541 | batch_size=512, hidden_features=32, learning_rate=0.001191, steps=800 |
| `real_matched_params` | 14 | 0.6357 | 0.6432 | batch_size=256, hidden_features=32, learning_rate=0.003545, steps=400 |
| `real_matched_params` | 15 | 0.5571 | 0.5660 | batch_size=128, hidden_features=16, learning_rate=0.002283, steps=400 |
| `real_matched_flops` | 0 | 0.7000 | 0.6982 | batch_size=512, hidden_features=32, learning_rate=0.002444, steps=800 |
| `real_matched_flops` | 1 | 0.6463 | 0.6480 | batch_size=512, hidden_features=16, learning_rate=0.007904, steps=800 |
| `real_matched_flops` | 2 | 0.4908 | 0.4957 | batch_size=256, hidden_features=16, learning_rate=0.02782, steps=200 |
| `real_matched_flops` | 3 | 0.4952 | 0.5045 | batch_size=512, hidden_features=64, learning_rate=0.03977, steps=200 |
| `real_matched_flops` | 4 | 0.6204 | 0.6279 | batch_size=128, hidden_features=32, learning_rate=0.001134, steps=800 |
| `real_matched_flops` | 5 | 0.4490 | 0.4458 | batch_size=256, hidden_features=64, learning_rate=0.02364, steps=400 |
| `real_matched_flops` | 6 | 0.4347 | 0.4439 | batch_size=256, hidden_features=32, learning_rate=0.01289, steps=200 |
| `real_matched_flops` | 7 | 0.5551 | 0.5630 | batch_size=256, hidden_features=16, learning_rate=0.002315, steps=200 |
| `real_matched_flops` | 8 | 0.5507 | 0.5636 | batch_size=256, hidden_features=16, learning_rate=0.002142, steps=200 |
| `real_matched_flops` | 9 | 0.5367 | 0.5296 | batch_size=128, hidden_features=16, learning_rate=0.001647, steps=200 |
| `real_matched_flops` | 10 | 0.5677 | 0.5694 | batch_size=128, hidden_features=32, learning_rate=0.004728, steps=400 |
| `real_matched_flops` | 11 | 0.4449 | 0.4442 | batch_size=256, hidden_features=64, learning_rate=0.02619, steps=200 |
| `real_matched_flops` | 12 | 0.4925 | 0.4888 | batch_size=128, hidden_features=64, learning_rate=0.008588, steps=400 |
| `real_matched_flops` | 13 | 0.6707 | 0.6835 | batch_size=512, hidden_features=32, learning_rate=0.001191, steps=800 |
| `real_matched_flops` | 14 | 0.6105 | 0.6290 | batch_size=256, hidden_features=32, learning_rate=0.003545, steps=400 |
| `real_matched_flops` | 15 | 0.5670 | 0.5807 | batch_size=128, hidden_features=16, learning_rate=0.002283, steps=400 |
