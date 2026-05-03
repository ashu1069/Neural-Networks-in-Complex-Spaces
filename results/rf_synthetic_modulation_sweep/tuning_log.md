# Tuning Log - Synthetic RF Modulation Classification

Random-search sweep following `docs/tuning_budget.md`: shared trial samples across all families, seeded per trial, selection by mean validation accuracy.

- Trials per family: `16`
- Seeds per trial: `[0, 1, 2, 3, 4, 5]`
- Sweep seed: `20260503`
- Search space: see `trials.json`

## Selected configuration per family

| family | trial | val acc | test acc | test std | hyperparameters |
| --- | ---: | ---: | ---: | ---: | --- |
| `complex` | 1 | 0.8164 | 0.8191 | 0.0163 | batch_size=512, hidden_features=16, learning_rate=0.007904, steps=800 |
| `real_stacked` | 1 | 0.7697 | 0.7740 | 0.0226 | batch_size=512, hidden_features=16, learning_rate=0.007904, steps=800 |
| `real_matched_params` | 0 | 0.7872 | 0.7914 | 0.0127 | batch_size=512, hidden_features=32, learning_rate=0.002444, steps=800 |
| `real_matched_flops` | 0 | 0.7799 | 0.7865 | 0.0159 | batch_size=512, hidden_features=32, learning_rate=0.002444, steps=800 |

## All trials (mean across seeds)

| family | trial | val acc | test acc | hyperparameters |
| --- | ---: | ---: | ---: | --- |
| `complex` | 0 | 0.8026 | 0.8063 | batch_size=512, hidden_features=32, learning_rate=0.002444, steps=800 |
| `complex` | 1 | 0.8164 | 0.8191 | batch_size=512, hidden_features=16, learning_rate=0.007904, steps=800 |
| `complex` | 2 | 0.7793 | 0.7772 | batch_size=256, hidden_features=16, learning_rate=0.02782, steps=200 |
| `complex` | 3 | 0.7944 | 0.7969 | batch_size=512, hidden_features=64, learning_rate=0.03977, steps=200 |
| `complex` | 4 | 0.7285 | 0.7282 | batch_size=128, hidden_features=32, learning_rate=0.001134, steps=800 |
| `complex` | 5 | 0.8005 | 0.8072 | batch_size=256, hidden_features=64, learning_rate=0.02364, steps=400 |
| `complex` | 6 | 0.7590 | 0.7669 | batch_size=256, hidden_features=32, learning_rate=0.01289, steps=200 |
| `complex` | 7 | 0.6280 | 0.6406 | batch_size=256, hidden_features=16, learning_rate=0.002315, steps=200 |
| `complex` | 8 | 0.6284 | 0.6393 | batch_size=256, hidden_features=16, learning_rate=0.002142, steps=200 |
| `complex` | 9 | 0.6239 | 0.6299 | batch_size=128, hidden_features=16, learning_rate=0.001647, steps=200 |
| `complex` | 10 | 0.7489 | 0.7526 | batch_size=128, hidden_features=32, learning_rate=0.004728, steps=400 |
| `complex` | 11 | 0.7874 | 0.7933 | batch_size=256, hidden_features=64, learning_rate=0.02619, steps=200 |
| `complex` | 12 | 0.7851 | 0.7875 | batch_size=128, hidden_features=64, learning_rate=0.008588, steps=400 |
| `complex` | 13 | 0.7771 | 0.7769 | batch_size=512, hidden_features=32, learning_rate=0.001191, steps=800 |
| `complex` | 14 | 0.7682 | 0.7682 | batch_size=256, hidden_features=32, learning_rate=0.003545, steps=400 |
| `complex` | 15 | 0.6741 | 0.6769 | batch_size=128, hidden_features=16, learning_rate=0.002283, steps=400 |
| `real_stacked` | 0 | 0.7692 | 0.7689 | batch_size=512, hidden_features=32, learning_rate=0.002444, steps=800 |
| `real_stacked` | 1 | 0.7697 | 0.7740 | batch_size=512, hidden_features=16, learning_rate=0.007904, steps=800 |
| `real_stacked` | 2 | 0.5548 | 0.5670 | batch_size=256, hidden_features=16, learning_rate=0.02782, steps=200 |
| `real_stacked` | 3 | 0.4874 | 0.4945 | batch_size=512, hidden_features=64, learning_rate=0.03977, steps=200 |
| `real_stacked` | 4 | 0.6974 | 0.6941 | batch_size=128, hidden_features=32, learning_rate=0.001134, steps=800 |
| `real_stacked` | 5 | 0.6459 | 0.6557 | batch_size=256, hidden_features=64, learning_rate=0.02364, steps=400 |
| `real_stacked` | 6 | 0.6352 | 0.6398 | batch_size=256, hidden_features=32, learning_rate=0.01289, steps=200 |
| `real_stacked` | 7 | 0.6420 | 0.6487 | batch_size=256, hidden_features=16, learning_rate=0.002315, steps=200 |
| `real_stacked` | 8 | 0.6377 | 0.6447 | batch_size=256, hidden_features=16, learning_rate=0.002142, steps=200 |
| `real_stacked` | 9 | 0.6402 | 0.6415 | batch_size=128, hidden_features=16, learning_rate=0.001647, steps=200 |
| `real_stacked` | 10 | 0.6869 | 0.6952 | batch_size=128, hidden_features=32, learning_rate=0.004728, steps=400 |
| `real_stacked` | 11 | 0.6457 | 0.6500 | batch_size=256, hidden_features=64, learning_rate=0.02619, steps=200 |
| `real_stacked` | 12 | 0.6811 | 0.6832 | batch_size=128, hidden_features=64, learning_rate=0.008588, steps=400 |
| `real_stacked` | 13 | 0.7608 | 0.7572 | batch_size=512, hidden_features=32, learning_rate=0.001191, steps=800 |
| `real_stacked` | 14 | 0.7157 | 0.7201 | batch_size=256, hidden_features=32, learning_rate=0.003545, steps=400 |
| `real_stacked` | 15 | 0.6433 | 0.6513 | batch_size=128, hidden_features=16, learning_rate=0.002283, steps=400 |
| `real_matched_params` | 0 | 0.7872 | 0.7914 | batch_size=512, hidden_features=32, learning_rate=0.002444, steps=800 |
| `real_matched_params` | 1 | 0.7754 | 0.7769 | batch_size=512, hidden_features=16, learning_rate=0.007904, steps=800 |
| `real_matched_params` | 2 | 0.6021 | 0.6058 | batch_size=256, hidden_features=16, learning_rate=0.02782, steps=200 |
| `real_matched_params` | 3 | 0.7124 | 0.7143 | batch_size=512, hidden_features=64, learning_rate=0.03977, steps=200 |
| `real_matched_params` | 4 | 0.7268 | 0.7273 | batch_size=128, hidden_features=32, learning_rate=0.001134, steps=800 |
| `real_matched_params` | 5 | 0.4940 | 0.4901 | batch_size=256, hidden_features=64, learning_rate=0.02364, steps=400 |
| `real_matched_params` | 6 | 0.6079 | 0.6090 | batch_size=256, hidden_features=32, learning_rate=0.01289, steps=200 |
| `real_matched_params` | 7 | 0.6482 | 0.6542 | batch_size=256, hidden_features=16, learning_rate=0.002315, steps=200 |
| `real_matched_params` | 8 | 0.6464 | 0.6548 | batch_size=256, hidden_features=16, learning_rate=0.002142, steps=200 |
| `real_matched_params` | 9 | 0.6305 | 0.6328 | batch_size=128, hidden_features=16, learning_rate=0.001647, steps=200 |
| `real_matched_params` | 10 | 0.7277 | 0.7315 | batch_size=128, hidden_features=32, learning_rate=0.004728, steps=400 |
| `real_matched_params` | 11 | 0.4835 | 0.4835 | batch_size=256, hidden_features=64, learning_rate=0.02619, steps=200 |
| `real_matched_params` | 12 | 0.6501 | 0.6563 | batch_size=128, hidden_features=64, learning_rate=0.008588, steps=400 |
| `real_matched_params` | 13 | 0.7596 | 0.7662 | batch_size=512, hidden_features=32, learning_rate=0.001191, steps=800 |
| `real_matched_params` | 14 | 0.7377 | 0.7372 | batch_size=256, hidden_features=32, learning_rate=0.003545, steps=400 |
| `real_matched_params` | 15 | 0.6453 | 0.6566 | batch_size=128, hidden_features=16, learning_rate=0.002283, steps=400 |
| `real_matched_flops` | 0 | 0.7799 | 0.7865 | batch_size=512, hidden_features=32, learning_rate=0.002444, steps=800 |
| `real_matched_flops` | 1 | 0.7777 | 0.7810 | batch_size=512, hidden_features=16, learning_rate=0.007904, steps=800 |
| `real_matched_flops` | 2 | 0.5478 | 0.5510 | batch_size=256, hidden_features=16, learning_rate=0.02782, steps=200 |
| `real_matched_flops` | 3 | 0.6404 | 0.6517 | batch_size=512, hidden_features=64, learning_rate=0.03977, steps=200 |
| `real_matched_flops` | 4 | 0.7511 | 0.7492 | batch_size=128, hidden_features=32, learning_rate=0.001134, steps=800 |
| `real_matched_flops` | 5 | 0.5259 | 0.5275 | batch_size=256, hidden_features=64, learning_rate=0.02364, steps=400 |
| `real_matched_flops` | 6 | 0.5984 | 0.6175 | batch_size=256, hidden_features=32, learning_rate=0.01289, steps=200 |
| `real_matched_flops` | 7 | 0.6354 | 0.6380 | batch_size=256, hidden_features=16, learning_rate=0.002315, steps=200 |
| `real_matched_flops` | 8 | 0.6324 | 0.6354 | batch_size=256, hidden_features=16, learning_rate=0.002142, steps=200 |
| `real_matched_flops` | 9 | 0.6352 | 0.6400 | batch_size=128, hidden_features=16, learning_rate=0.001647, steps=200 |
| `real_matched_flops` | 10 | 0.6974 | 0.7033 | batch_size=128, hidden_features=32, learning_rate=0.004728, steps=400 |
| `real_matched_flops` | 11 | 0.4222 | 0.4309 | batch_size=256, hidden_features=64, learning_rate=0.02619, steps=200 |
| `real_matched_flops` | 12 | 0.6908 | 0.6903 | batch_size=128, hidden_features=64, learning_rate=0.008588, steps=400 |
| `real_matched_flops` | 13 | 0.7765 | 0.7775 | batch_size=512, hidden_features=32, learning_rate=0.001191, steps=800 |
| `real_matched_flops` | 14 | 0.7266 | 0.7324 | batch_size=256, hidden_features=32, learning_rate=0.003545, steps=400 |
| `real_matched_flops` | 15 | 0.6708 | 0.6812 | batch_size=128, hidden_features=16, learning_rate=0.002283, steps=400 |
