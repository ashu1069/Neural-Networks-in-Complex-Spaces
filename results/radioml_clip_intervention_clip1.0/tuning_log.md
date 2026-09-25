# Tuning Log - RadioML 2018.01A Modulation Classification

Random-search sweep following `docs/tuning_budget.md`: shared trial samples across all families, seeded per trial, selection by mean validation accuracy.

- Trials per family: `16`
- Seeds per trial: `[0, 1, 2, 3, 4, 5]`
- Sweep seed: `20260503`
- Search space: see `trials.json`

## Selected configuration per family

| family | trial | val acc | test acc | test std | hyperparameters |
| --- | ---: | ---: | ---: | ---: | --- |
| `complex` | 5 | 0.7210 | 0.7284 | 0.0314 | batch_size=256, hidden_features=64, learning_rate=0.02364, steps=400 |
| `real_stacked` | 5 | 0.4976 | 0.5007 | 0.1304 | batch_size=256, hidden_features=64, learning_rate=0.02364, steps=400 |
| `real_matched_params` | 5 | 0.5070 | 0.5132 | 0.1397 | batch_size=256, hidden_features=64, learning_rate=0.02364, steps=400 |
| `real_matched_flops` | 5 | 0.5414 | 0.5398 | 0.1041 | batch_size=256, hidden_features=64, learning_rate=0.02364, steps=400 |
| `real_polar` | 5 | 0.6896 | 0.6880 | 0.0278 | batch_size=256, hidden_features=64, learning_rate=0.02364, steps=400 |
| `real_phase` | 5 | 0.5005 | 0.5045 | 0.1382 | batch_size=256, hidden_features=64, learning_rate=0.02364, steps=400 |
| `real_magnitude` | 5 | 0.5560 | 0.5525 | 0.0308 | batch_size=256, hidden_features=64, learning_rate=0.02364, steps=400 |

## All trials (mean across seeds)

| family | trial | val acc | test acc | hyperparameters |
| --- | ---: | ---: | ---: | --- |
| `complex` | 0 | 0.6486 | 0.6436 | batch_size=512, hidden_features=32, learning_rate=0.002444, steps=800 |
| `complex` | 1 | 0.6774 | 0.6854 | batch_size=512, hidden_features=16, learning_rate=0.007904, steps=800 |
| `complex` | 2 | 0.6425 | 0.6386 | batch_size=256, hidden_features=16, learning_rate=0.02782, steps=200 |
| `complex` | 3 | 0.6650 | 0.6688 | batch_size=512, hidden_features=64, learning_rate=0.03977, steps=200 |
| `complex` | 4 | 0.6299 | 0.6235 | batch_size=128, hidden_features=32, learning_rate=0.001134, steps=800 |
| `complex` | 5 | 0.7210 | 0.7284 | batch_size=256, hidden_features=64, learning_rate=0.02364, steps=400 |
| `complex` | 6 | 0.6139 | 0.6047 | batch_size=256, hidden_features=32, learning_rate=0.01289, steps=200 |
| `complex` | 7 | 0.5993 | 0.5940 | batch_size=256, hidden_features=16, learning_rate=0.002315, steps=200 |
| `complex` | 8 | 0.5970 | 0.5954 | batch_size=256, hidden_features=16, learning_rate=0.002142, steps=200 |
| `complex` | 9 | 0.5858 | 0.5856 | batch_size=128, hidden_features=16, learning_rate=0.001647, steps=200 |
| `complex` | 10 | 0.6256 | 0.6247 | batch_size=128, hidden_features=32, learning_rate=0.004728, steps=400 |
| `complex` | 11 | 0.6541 | 0.6492 | batch_size=256, hidden_features=64, learning_rate=0.02619, steps=200 |
| `complex` | 12 | 0.6738 | 0.6874 | batch_size=128, hidden_features=64, learning_rate=0.008588, steps=400 |
| `complex` | 13 | 0.6423 | 0.6437 | batch_size=512, hidden_features=32, learning_rate=0.001191, steps=800 |
| `complex` | 14 | 0.6359 | 0.6267 | batch_size=256, hidden_features=32, learning_rate=0.003545, steps=400 |
| `complex` | 15 | 0.6050 | 0.5917 | batch_size=128, hidden_features=16, learning_rate=0.002283, steps=400 |
| `real_stacked` | 0 | 0.6713 | 0.6696 | batch_size=512, hidden_features=32, learning_rate=0.002444, steps=800 |
| `real_stacked` | 1 | 0.6362 | 0.6356 | batch_size=512, hidden_features=16, learning_rate=0.007904, steps=800 |
| `real_stacked` | 2 | 0.5666 | 0.5718 | batch_size=256, hidden_features=16, learning_rate=0.02782, steps=200 |
| `real_stacked` | 3 | 0.5039 | 0.5139 | batch_size=512, hidden_features=64, learning_rate=0.03977, steps=200 |
| `real_stacked` | 4 | 0.5967 | 0.5873 | batch_size=128, hidden_features=32, learning_rate=0.001134, steps=800 |
| `real_stacked` | 5 | 0.4976 | 0.5007 | batch_size=256, hidden_features=64, learning_rate=0.02364, steps=400 |
| `real_stacked` | 6 | 0.4566 | 0.4692 | batch_size=256, hidden_features=32, learning_rate=0.01289, steps=200 |
| `real_stacked` | 7 | 0.5247 | 0.5315 | batch_size=256, hidden_features=16, learning_rate=0.002315, steps=200 |
| `real_stacked` | 8 | 0.5223 | 0.5230 | batch_size=256, hidden_features=16, learning_rate=0.002142, steps=200 |
| `real_stacked` | 9 | 0.4716 | 0.4667 | batch_size=128, hidden_features=16, learning_rate=0.001647, steps=200 |
| `real_stacked` | 10 | 0.5633 | 0.5633 | batch_size=128, hidden_features=32, learning_rate=0.004728, steps=400 |
| `real_stacked` | 11 | 0.4084 | 0.4149 | batch_size=256, hidden_features=64, learning_rate=0.02619, steps=200 |
| `real_stacked` | 12 | 0.5082 | 0.5027 | batch_size=128, hidden_features=64, learning_rate=0.008588, steps=400 |
| `real_stacked` | 13 | 0.6255 | 0.6185 | batch_size=512, hidden_features=32, learning_rate=0.001191, steps=800 |
| `real_stacked` | 14 | 0.6059 | 0.5944 | batch_size=256, hidden_features=32, learning_rate=0.003545, steps=400 |
| `real_stacked` | 15 | 0.5601 | 0.5617 | batch_size=128, hidden_features=16, learning_rate=0.002283, steps=400 |
| `real_matched_params` | 0 | 0.7074 | 0.6997 | batch_size=512, hidden_features=32, learning_rate=0.002444, steps=800 |
| `real_matched_params` | 1 | 0.6942 | 0.6942 | batch_size=512, hidden_features=16, learning_rate=0.007904, steps=800 |
| `real_matched_params` | 2 | 0.4821 | 0.4901 | batch_size=256, hidden_features=16, learning_rate=0.02782, steps=200 |
| `real_matched_params` | 3 | 0.5364 | 0.5462 | batch_size=512, hidden_features=64, learning_rate=0.03977, steps=200 |
| `real_matched_params` | 4 | 0.6141 | 0.6120 | batch_size=128, hidden_features=32, learning_rate=0.001134, steps=800 |
| `real_matched_params` | 5 | 0.5070 | 0.5132 | batch_size=256, hidden_features=64, learning_rate=0.02364, steps=400 |
| `real_matched_params` | 6 | 0.4384 | 0.4460 | batch_size=256, hidden_features=32, learning_rate=0.01289, steps=200 |
| `real_matched_params` | 7 | 0.5483 | 0.5479 | batch_size=256, hidden_features=16, learning_rate=0.002315, steps=200 |
| `real_matched_params` | 8 | 0.5553 | 0.5511 | batch_size=256, hidden_features=16, learning_rate=0.002142, steps=200 |
| `real_matched_params` | 9 | 0.5293 | 0.5267 | batch_size=128, hidden_features=16, learning_rate=0.001647, steps=200 |
| `real_matched_params` | 10 | 0.5686 | 0.5610 | batch_size=128, hidden_features=32, learning_rate=0.004728, steps=400 |
| `real_matched_params` | 11 | 0.4983 | 0.5037 | batch_size=256, hidden_features=64, learning_rate=0.02619, steps=200 |
| `real_matched_params` | 12 | 0.5402 | 0.5459 | batch_size=128, hidden_features=64, learning_rate=0.008588, steps=400 |
| `real_matched_params` | 13 | 0.6549 | 0.6433 | batch_size=512, hidden_features=32, learning_rate=0.001191, steps=800 |
| `real_matched_params` | 14 | 0.6222 | 0.6218 | batch_size=256, hidden_features=32, learning_rate=0.003545, steps=400 |
| `real_matched_params` | 15 | 0.5759 | 0.5637 | batch_size=128, hidden_features=16, learning_rate=0.002283, steps=400 |
| `real_matched_flops` | 0 | 0.7067 | 0.7025 | batch_size=512, hidden_features=32, learning_rate=0.002444, steps=800 |
| `real_matched_flops` | 1 | 0.7091 | 0.6993 | batch_size=512, hidden_features=16, learning_rate=0.007904, steps=800 |
| `real_matched_flops` | 2 | 0.4442 | 0.4538 | batch_size=256, hidden_features=16, learning_rate=0.02782, steps=200 |
| `real_matched_flops` | 3 | 0.6062 | 0.5998 | batch_size=512, hidden_features=64, learning_rate=0.03977, steps=200 |
| `real_matched_flops` | 4 | 0.6280 | 0.6231 | batch_size=128, hidden_features=32, learning_rate=0.001134, steps=800 |
| `real_matched_flops` | 5 | 0.5414 | 0.5398 | batch_size=256, hidden_features=64, learning_rate=0.02364, steps=400 |
| `real_matched_flops` | 6 | 0.4631 | 0.4643 | batch_size=256, hidden_features=32, learning_rate=0.01289, steps=200 |
| `real_matched_flops` | 7 | 0.5633 | 0.5517 | batch_size=256, hidden_features=16, learning_rate=0.002315, steps=200 |
| `real_matched_flops` | 8 | 0.5579 | 0.5497 | batch_size=256, hidden_features=16, learning_rate=0.002142, steps=200 |
| `real_matched_flops` | 9 | 0.5436 | 0.5282 | batch_size=128, hidden_features=16, learning_rate=0.001647, steps=200 |
| `real_matched_flops` | 10 | 0.5599 | 0.5605 | batch_size=128, hidden_features=32, learning_rate=0.004728, steps=400 |
| `real_matched_flops` | 11 | 0.4971 | 0.4963 | batch_size=256, hidden_features=64, learning_rate=0.02619, steps=200 |
| `real_matched_flops` | 12 | 0.4418 | 0.4427 | batch_size=128, hidden_features=64, learning_rate=0.008588, steps=400 |
| `real_matched_flops` | 13 | 0.6745 | 0.6735 | batch_size=512, hidden_features=32, learning_rate=0.001191, steps=800 |
| `real_matched_flops` | 14 | 0.6037 | 0.5998 | batch_size=256, hidden_features=32, learning_rate=0.003545, steps=400 |
| `real_matched_flops` | 15 | 0.5807 | 0.5732 | batch_size=128, hidden_features=16, learning_rate=0.002283, steps=400 |
| `real_polar` | 0 | 0.7106 | 0.7017 | batch_size=512, hidden_features=32, learning_rate=0.002444, steps=800 |
| `real_polar` | 1 | 0.7150 | 0.7097 | batch_size=512, hidden_features=16, learning_rate=0.007904, steps=800 |
| `real_polar` | 2 | 0.5409 | 0.5371 | batch_size=256, hidden_features=16, learning_rate=0.02782, steps=200 |
| `real_polar` | 3 | 0.6845 | 0.6791 | batch_size=512, hidden_features=64, learning_rate=0.03977, steps=200 |
| `real_polar` | 4 | 0.6326 | 0.6299 | batch_size=128, hidden_features=32, learning_rate=0.001134, steps=800 |
| `real_polar` | 5 | 0.6896 | 0.6880 | batch_size=256, hidden_features=64, learning_rate=0.02364, steps=400 |
| `real_polar` | 6 | 0.6122 | 0.6198 | batch_size=256, hidden_features=32, learning_rate=0.01289, steps=200 |
| `real_polar` | 7 | 0.5169 | 0.5283 | batch_size=256, hidden_features=16, learning_rate=0.002315, steps=200 |
| `real_polar` | 8 | 0.5169 | 0.5224 | batch_size=256, hidden_features=16, learning_rate=0.002142, steps=200 |
| `real_polar` | 9 | 0.5010 | 0.5075 | batch_size=128, hidden_features=16, learning_rate=0.001647, steps=200 |
| `real_polar` | 10 | 0.6217 | 0.6278 | batch_size=128, hidden_features=32, learning_rate=0.004728, steps=400 |
| `real_polar` | 11 | 0.6505 | 0.6468 | batch_size=256, hidden_features=64, learning_rate=0.02619, steps=200 |
| `real_polar` | 12 | 0.6561 | 0.6595 | batch_size=128, hidden_features=64, learning_rate=0.008588, steps=400 |
| `real_polar` | 13 | 0.6651 | 0.6636 | batch_size=512, hidden_features=32, learning_rate=0.001191, steps=800 |
| `real_polar` | 14 | 0.6549 | 0.6522 | batch_size=256, hidden_features=32, learning_rate=0.003545, steps=400 |
| `real_polar` | 15 | 0.5579 | 0.5678 | batch_size=128, hidden_features=16, learning_rate=0.002283, steps=400 |
| `real_phase` | 0 | 0.6679 | 0.6669 | batch_size=512, hidden_features=32, learning_rate=0.002444, steps=800 |
| `real_phase` | 1 | 0.6789 | 0.6715 | batch_size=512, hidden_features=16, learning_rate=0.007904, steps=800 |
| `real_phase` | 2 | 0.5855 | 0.5840 | batch_size=256, hidden_features=16, learning_rate=0.02782, steps=200 |
| `real_phase` | 3 | 0.5187 | 0.5138 | batch_size=512, hidden_features=64, learning_rate=0.03977, steps=200 |
| `real_phase` | 4 | 0.6239 | 0.6147 | batch_size=128, hidden_features=32, learning_rate=0.001134, steps=800 |
| `real_phase` | 5 | 0.5005 | 0.5045 | batch_size=256, hidden_features=64, learning_rate=0.02364, steps=400 |
| `real_phase` | 6 | 0.5087 | 0.5124 | batch_size=256, hidden_features=32, learning_rate=0.01289, steps=200 |
| `real_phase` | 7 | 0.5788 | 0.5697 | batch_size=256, hidden_features=16, learning_rate=0.002315, steps=200 |
| `real_phase` | 8 | 0.5735 | 0.5638 | batch_size=256, hidden_features=16, learning_rate=0.002142, steps=200 |
| `real_phase` | 9 | 0.5269 | 0.5215 | batch_size=128, hidden_features=16, learning_rate=0.001647, steps=200 |
| `real_phase` | 10 | 0.6035 | 0.5967 | batch_size=128, hidden_features=32, learning_rate=0.004728, steps=400 |
| `real_phase` | 11 | 0.4940 | 0.5020 | batch_size=256, hidden_features=64, learning_rate=0.02619, steps=200 |
| `real_phase` | 12 | 0.5945 | 0.5928 | batch_size=128, hidden_features=64, learning_rate=0.008588, steps=400 |
| `real_phase` | 13 | 0.6321 | 0.6274 | batch_size=512, hidden_features=32, learning_rate=0.001191, steps=800 |
| `real_phase` | 14 | 0.6248 | 0.6118 | batch_size=256, hidden_features=32, learning_rate=0.003545, steps=400 |
| `real_phase` | 15 | 0.5999 | 0.5962 | batch_size=128, hidden_features=16, learning_rate=0.002283, steps=400 |
| `real_magnitude` | 0 | 0.5603 | 0.5486 | batch_size=512, hidden_features=32, learning_rate=0.002444, steps=800 |
| `real_magnitude` | 1 | 0.5577 | 0.5540 | batch_size=512, hidden_features=16, learning_rate=0.007904, steps=800 |
| `real_magnitude` | 2 | 0.4964 | 0.4909 | batch_size=256, hidden_features=16, learning_rate=0.02782, steps=200 |
| `real_magnitude` | 3 | 0.5334 | 0.5413 | batch_size=512, hidden_features=64, learning_rate=0.03977, steps=200 |
| `real_magnitude` | 4 | 0.5247 | 0.5275 | batch_size=128, hidden_features=32, learning_rate=0.001134, steps=800 |
| `real_magnitude` | 5 | 0.5560 | 0.5525 | batch_size=256, hidden_features=64, learning_rate=0.02364, steps=400 |
| `real_magnitude` | 6 | 0.5262 | 0.5306 | batch_size=256, hidden_features=32, learning_rate=0.01289, steps=200 |
| `real_magnitude` | 7 | 0.4998 | 0.4945 | batch_size=256, hidden_features=16, learning_rate=0.002315, steps=200 |
| `real_magnitude` | 8 | 0.4954 | 0.4909 | batch_size=256, hidden_features=16, learning_rate=0.002142, steps=200 |
| `real_magnitude` | 9 | 0.4913 | 0.4955 | batch_size=128, hidden_features=16, learning_rate=0.001647, steps=200 |
| `real_magnitude` | 10 | 0.5262 | 0.5244 | batch_size=128, hidden_features=32, learning_rate=0.004728, steps=400 |
| `real_magnitude` | 11 | 0.5409 | 0.5326 | batch_size=256, hidden_features=64, learning_rate=0.02619, steps=200 |
| `real_magnitude` | 12 | 0.5340 | 0.5330 | batch_size=128, hidden_features=64, learning_rate=0.008588, steps=400 |
| `real_magnitude` | 13 | 0.5397 | 0.5395 | batch_size=512, hidden_features=32, learning_rate=0.001191, steps=800 |
| `real_magnitude` | 14 | 0.5284 | 0.5295 | batch_size=256, hidden_features=32, learning_rate=0.003545, steps=400 |
| `real_magnitude` | 15 | 0.5060 | 0.5044 | batch_size=128, hidden_features=16, learning_rate=0.002283, steps=400 |
