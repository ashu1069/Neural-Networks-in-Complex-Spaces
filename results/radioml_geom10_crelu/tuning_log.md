# Tuning Log - RadioML 2018.01A Modulation Classification

Random-search sweep following `docs/tuning_budget.md`: shared trial samples across all families, seeded per trial, selection by mean validation accuracy.

- Trials per family: `16`
- Seeds per trial: `[0, 1, 2, 3, 4]`
- Sweep seed: `20260503`
- Search space: see `trials.json`

## Selected configuration per family

| family | trial | val acc | test acc | test std | hyperparameters |
| --- | ---: | ---: | ---: | ---: | --- |
| `complex` | 5 | 0.3065 | 0.3055 | 0.0103 | batch_size=256, hidden_features=64, learning_rate=0.02364, steps=400 |
| `real_stacked` | 5 | 0.2109 | 0.2090 | 0.0132 | batch_size=256, hidden_features=64, learning_rate=0.02364, steps=400 |
| `real_matched_params` | 5 | 0.1745 | 0.1746 | 0.0767 | batch_size=256, hidden_features=64, learning_rate=0.02364, steps=400 |
| `real_matched_flops` | 5 | 0.1971 | 0.2029 | 0.0149 | batch_size=256, hidden_features=64, learning_rate=0.02364, steps=400 |
| `real_polar` | 5 | 0.2989 | 0.2967 | 0.0101 | batch_size=256, hidden_features=64, learning_rate=0.02364, steps=400 |
| `real_phase` | 5 | 0.1184 | 0.1198 | 0.0442 | batch_size=256, hidden_features=64, learning_rate=0.02364, steps=400 |
| `real_magnitude` | 5 | 0.2705 | 0.2683 | 0.0299 | batch_size=256, hidden_features=64, learning_rate=0.02364, steps=400 |

## All trials (mean across seeds)

| family | trial | val acc | test acc | hyperparameters |
| --- | ---: | ---: | ---: | --- |
| `complex` | 0 | 0.2603 | 0.2630 | batch_size=512, hidden_features=32, learning_rate=0.002444, steps=800 |
| `complex` | 1 | 0.2795 | 0.2815 | batch_size=512, hidden_features=16, learning_rate=0.007904, steps=800 |
| `complex` | 2 | 0.2429 | 0.2453 | batch_size=256, hidden_features=16, learning_rate=0.02782, steps=200 |
| `complex` | 3 | 0.2814 | 0.2804 | batch_size=512, hidden_features=64, learning_rate=0.03977, steps=200 |
| `complex` | 4 | 0.2411 | 0.2409 | batch_size=128, hidden_features=32, learning_rate=0.001134, steps=800 |
| `complex` | 5 | 0.3065 | 0.3055 | batch_size=256, hidden_features=64, learning_rate=0.02364, steps=400 |
| `complex` | 6 | 0.2451 | 0.2404 | batch_size=256, hidden_features=32, learning_rate=0.01289, steps=200 |
| `complex` | 7 | 0.2309 | 0.2274 | batch_size=256, hidden_features=16, learning_rate=0.002315, steps=200 |
| `complex` | 8 | 0.2285 | 0.2253 | batch_size=256, hidden_features=16, learning_rate=0.002142, steps=200 |
| `complex` | 9 | 0.2210 | 0.2159 | batch_size=128, hidden_features=16, learning_rate=0.001647, steps=200 |
| `complex` | 10 | 0.2434 | 0.2434 | batch_size=128, hidden_features=32, learning_rate=0.004728, steps=400 |
| `complex` | 11 | 0.2565 | 0.2568 | batch_size=256, hidden_features=64, learning_rate=0.02619, steps=200 |
| `complex` | 12 | 0.2534 | 0.2535 | batch_size=128, hidden_features=64, learning_rate=0.008588, steps=400 |
| `complex` | 13 | 0.2475 | 0.2510 | batch_size=512, hidden_features=32, learning_rate=0.001191, steps=800 |
| `complex` | 14 | 0.2445 | 0.2454 | batch_size=256, hidden_features=32, learning_rate=0.003545, steps=400 |
| `complex` | 15 | 0.2328 | 0.2357 | batch_size=128, hidden_features=16, learning_rate=0.002283, steps=400 |
| `real_stacked` | 0 | 0.2839 | 0.2833 | batch_size=512, hidden_features=32, learning_rate=0.002444, steps=800 |
| `real_stacked` | 1 | 0.2900 | 0.2869 | batch_size=512, hidden_features=16, learning_rate=0.007904, steps=800 |
| `real_stacked` | 2 | 0.1926 | 0.1979 | batch_size=256, hidden_features=16, learning_rate=0.02782, steps=200 |
| `real_stacked` | 3 | 0.1877 | 0.1917 | batch_size=512, hidden_features=64, learning_rate=0.03977, steps=200 |
| `real_stacked` | 4 | 0.2229 | 0.2262 | batch_size=128, hidden_features=32, learning_rate=0.001134, steps=800 |
| `real_stacked` | 5 | 0.2109 | 0.2090 | batch_size=256, hidden_features=64, learning_rate=0.02364, steps=400 |
| `real_stacked` | 6 | 0.2060 | 0.2069 | batch_size=256, hidden_features=32, learning_rate=0.01289, steps=200 |
| `real_stacked` | 7 | 0.1730 | 0.1725 | batch_size=256, hidden_features=16, learning_rate=0.002315, steps=200 |
| `real_stacked` | 8 | 0.1714 | 0.1713 | batch_size=256, hidden_features=16, learning_rate=0.002142, steps=200 |
| `real_stacked` | 9 | 0.1425 | 0.1457 | batch_size=128, hidden_features=16, learning_rate=0.001647, steps=200 |
| `real_stacked` | 10 | 0.2245 | 0.2230 | batch_size=128, hidden_features=32, learning_rate=0.004728, steps=400 |
| `real_stacked` | 11 | 0.1901 | 0.1880 | batch_size=256, hidden_features=64, learning_rate=0.02619, steps=200 |
| `real_stacked` | 12 | 0.2394 | 0.2365 | batch_size=128, hidden_features=64, learning_rate=0.008588, steps=400 |
| `real_stacked` | 13 | 0.2582 | 0.2540 | batch_size=512, hidden_features=32, learning_rate=0.001191, steps=800 |
| `real_stacked` | 14 | 0.2413 | 0.2396 | batch_size=256, hidden_features=32, learning_rate=0.003545, steps=400 |
| `real_stacked` | 15 | 0.1750 | 0.1748 | batch_size=128, hidden_features=16, learning_rate=0.002283, steps=400 |
| `real_matched_params` | 0 | 0.2918 | 0.2856 | batch_size=512, hidden_features=32, learning_rate=0.002444, steps=800 |
| `real_matched_params` | 1 | 0.2930 | 0.2885 | batch_size=512, hidden_features=16, learning_rate=0.007904, steps=800 |
| `real_matched_params` | 2 | 0.1949 | 0.1940 | batch_size=256, hidden_features=16, learning_rate=0.02782, steps=200 |
| `real_matched_params` | 3 | 0.2071 | 0.2047 | batch_size=512, hidden_features=64, learning_rate=0.03977, steps=200 |
| `real_matched_params` | 4 | 0.2345 | 0.2309 | batch_size=128, hidden_features=32, learning_rate=0.001134, steps=800 |
| `real_matched_params` | 5 | 0.1745 | 0.1746 | batch_size=256, hidden_features=64, learning_rate=0.02364, steps=400 |
| `real_matched_params` | 6 | 0.2099 | 0.2123 | batch_size=256, hidden_features=32, learning_rate=0.01289, steps=200 |
| `real_matched_params` | 7 | 0.1824 | 0.1859 | batch_size=256, hidden_features=16, learning_rate=0.002315, steps=200 |
| `real_matched_params` | 8 | 0.1838 | 0.1856 | batch_size=256, hidden_features=16, learning_rate=0.002142, steps=200 |
| `real_matched_params` | 9 | 0.1565 | 0.1586 | batch_size=128, hidden_features=16, learning_rate=0.001647, steps=200 |
| `real_matched_params` | 10 | 0.2384 | 0.2388 | batch_size=128, hidden_features=32, learning_rate=0.004728, steps=400 |
| `real_matched_params` | 11 | 0.1674 | 0.1681 | batch_size=256, hidden_features=64, learning_rate=0.02619, steps=200 |
| `real_matched_params` | 12 | 0.2358 | 0.2325 | batch_size=128, hidden_features=64, learning_rate=0.008588, steps=400 |
| `real_matched_params` | 13 | 0.2628 | 0.2617 | batch_size=512, hidden_features=32, learning_rate=0.001191, steps=800 |
| `real_matched_params` | 14 | 0.2502 | 0.2507 | batch_size=256, hidden_features=32, learning_rate=0.003545, steps=400 |
| `real_matched_params` | 15 | 0.1901 | 0.1913 | batch_size=128, hidden_features=16, learning_rate=0.002283, steps=400 |
| `real_matched_flops` | 0 | 0.3036 | 0.2981 | batch_size=512, hidden_features=32, learning_rate=0.002444, steps=800 |
| `real_matched_flops` | 1 | 0.2913 | 0.2908 | batch_size=512, hidden_features=16, learning_rate=0.007904, steps=800 |
| `real_matched_flops` | 2 | 0.2038 | 0.2097 | batch_size=256, hidden_features=16, learning_rate=0.02782, steps=200 |
| `real_matched_flops` | 3 | 0.2319 | 0.2350 | batch_size=512, hidden_features=64, learning_rate=0.03977, steps=200 |
| `real_matched_flops` | 4 | 0.2422 | 0.2412 | batch_size=128, hidden_features=32, learning_rate=0.001134, steps=800 |
| `real_matched_flops` | 5 | 0.1971 | 0.2029 | batch_size=256, hidden_features=64, learning_rate=0.02364, steps=400 |
| `real_matched_flops` | 6 | 0.2062 | 0.2034 | batch_size=256, hidden_features=32, learning_rate=0.01289, steps=200 |
| `real_matched_flops` | 7 | 0.1816 | 0.1823 | batch_size=256, hidden_features=16, learning_rate=0.002315, steps=200 |
| `real_matched_flops` | 8 | 0.1791 | 0.1831 | batch_size=256, hidden_features=16, learning_rate=0.002142, steps=200 |
| `real_matched_flops` | 9 | 0.1612 | 0.1634 | batch_size=128, hidden_features=16, learning_rate=0.001647, steps=200 |
| `real_matched_flops` | 10 | 0.2436 | 0.2410 | batch_size=128, hidden_features=32, learning_rate=0.004728, steps=400 |
| `real_matched_flops` | 11 | 0.1868 | 0.1947 | batch_size=256, hidden_features=64, learning_rate=0.02619, steps=200 |
| `real_matched_flops` | 12 | 0.2315 | 0.2291 | batch_size=128, hidden_features=64, learning_rate=0.008588, steps=400 |
| `real_matched_flops` | 13 | 0.2794 | 0.2762 | batch_size=512, hidden_features=32, learning_rate=0.001191, steps=800 |
| `real_matched_flops` | 14 | 0.2560 | 0.2542 | batch_size=256, hidden_features=32, learning_rate=0.003545, steps=400 |
| `real_matched_flops` | 15 | 0.2072 | 0.2094 | batch_size=128, hidden_features=16, learning_rate=0.002283, steps=400 |
| `real_polar` | 0 | 0.3060 | 0.3054 | batch_size=512, hidden_features=32, learning_rate=0.002444, steps=800 |
| `real_polar` | 1 | 0.3110 | 0.3115 | batch_size=512, hidden_features=16, learning_rate=0.007904, steps=800 |
| `real_polar` | 2 | 0.2388 | 0.2417 | batch_size=256, hidden_features=16, learning_rate=0.02782, steps=200 |
| `real_polar` | 3 | 0.2776 | 0.2813 | batch_size=512, hidden_features=64, learning_rate=0.03977, steps=200 |
| `real_polar` | 4 | 0.2633 | 0.2604 | batch_size=128, hidden_features=32, learning_rate=0.001134, steps=800 |
| `real_polar` | 5 | 0.2989 | 0.2967 | batch_size=256, hidden_features=64, learning_rate=0.02364, steps=400 |
| `real_polar` | 6 | 0.2589 | 0.2585 | batch_size=256, hidden_features=32, learning_rate=0.01289, steps=200 |
| `real_polar` | 7 | 0.2007 | 0.1994 | batch_size=256, hidden_features=16, learning_rate=0.002315, steps=200 |
| `real_polar` | 8 | 0.1983 | 0.1993 | batch_size=256, hidden_features=16, learning_rate=0.002142, steps=200 |
| `real_polar` | 9 | 0.1708 | 0.1724 | batch_size=128, hidden_features=16, learning_rate=0.001647, steps=200 |
| `real_polar` | 10 | 0.2600 | 0.2624 | batch_size=128, hidden_features=32, learning_rate=0.004728, steps=400 |
| `real_polar` | 11 | 0.2773 | 0.2750 | batch_size=256, hidden_features=64, learning_rate=0.02619, steps=200 |
| `real_polar` | 12 | 0.2849 | 0.2872 | batch_size=128, hidden_features=64, learning_rate=0.008588, steps=400 |
| `real_polar` | 13 | 0.2872 | 0.2857 | batch_size=512, hidden_features=32, learning_rate=0.001191, steps=800 |
| `real_polar` | 14 | 0.2781 | 0.2774 | batch_size=256, hidden_features=32, learning_rate=0.003545, steps=400 |
| `real_polar` | 15 | 0.2075 | 0.2089 | batch_size=128, hidden_features=16, learning_rate=0.002283, steps=400 |
| `real_phase` | 0 | 0.2349 | 0.2365 | batch_size=512, hidden_features=32, learning_rate=0.002444, steps=800 |
| `real_phase` | 1 | 0.2465 | 0.2495 | batch_size=512, hidden_features=16, learning_rate=0.007904, steps=800 |
| `real_phase` | 2 | 0.1758 | 0.1734 | batch_size=256, hidden_features=16, learning_rate=0.02782, steps=200 |
| `real_phase` | 3 | 0.0980 | 0.1000 | batch_size=512, hidden_features=64, learning_rate=0.03977, steps=200 |
| `real_phase` | 4 | 0.2196 | 0.2207 | batch_size=128, hidden_features=32, learning_rate=0.001134, steps=800 |
| `real_phase` | 5 | 0.1184 | 0.1198 | batch_size=256, hidden_features=64, learning_rate=0.02364, steps=400 |
| `real_phase` | 6 | 0.1989 | 0.1963 | batch_size=256, hidden_features=32, learning_rate=0.01289, steps=200 |
| `real_phase` | 7 | 0.1744 | 0.1731 | batch_size=256, hidden_features=16, learning_rate=0.002315, steps=200 |
| `real_phase` | 8 | 0.1668 | 0.1693 | batch_size=256, hidden_features=16, learning_rate=0.002142, steps=200 |
| `real_phase` | 9 | 0.1489 | 0.1497 | batch_size=128, hidden_features=16, learning_rate=0.001647, steps=200 |
| `real_phase` | 10 | 0.2208 | 0.2200 | batch_size=128, hidden_features=32, learning_rate=0.004728, steps=400 |
| `real_phase` | 11 | 0.1160 | 0.1152 | batch_size=256, hidden_features=64, learning_rate=0.02619, steps=200 |
| `real_phase` | 12 | 0.1953 | 0.1998 | batch_size=128, hidden_features=64, learning_rate=0.008588, steps=400 |
| `real_phase` | 13 | 0.2319 | 0.2331 | batch_size=512, hidden_features=32, learning_rate=0.001191, steps=800 |
| `real_phase` | 14 | 0.2245 | 0.2232 | batch_size=256, hidden_features=32, learning_rate=0.003545, steps=400 |
| `real_phase` | 15 | 0.1900 | 0.1909 | batch_size=128, hidden_features=16, learning_rate=0.002283, steps=400 |
| `real_magnitude` | 0 | 0.2799 | 0.2724 | batch_size=512, hidden_features=32, learning_rate=0.002444, steps=800 |
| `real_magnitude` | 1 | 0.2774 | 0.2754 | batch_size=512, hidden_features=16, learning_rate=0.007904, steps=800 |
| `real_magnitude` | 2 | 0.2197 | 0.2187 | batch_size=256, hidden_features=16, learning_rate=0.02782, steps=200 |
| `real_magnitude` | 3 | 0.2306 | 0.2302 | batch_size=512, hidden_features=64, learning_rate=0.03977, steps=200 |
| `real_magnitude` | 4 | 0.2468 | 0.2458 | batch_size=128, hidden_features=32, learning_rate=0.001134, steps=800 |
| `real_magnitude` | 5 | 0.2705 | 0.2683 | batch_size=256, hidden_features=64, learning_rate=0.02364, steps=400 |
| `real_magnitude` | 6 | 0.2372 | 0.2299 | batch_size=256, hidden_features=32, learning_rate=0.01289, steps=200 |
| `real_magnitude` | 7 | 0.1761 | 0.1767 | batch_size=256, hidden_features=16, learning_rate=0.002315, steps=200 |
| `real_magnitude` | 8 | 0.1747 | 0.1760 | batch_size=256, hidden_features=16, learning_rate=0.002142, steps=200 |
| `real_magnitude` | 9 | 0.1645 | 0.1658 | batch_size=128, hidden_features=16, learning_rate=0.001647, steps=200 |
| `real_magnitude` | 10 | 0.2438 | 0.2412 | batch_size=128, hidden_features=32, learning_rate=0.004728, steps=400 |
| `real_magnitude` | 11 | 0.2304 | 0.2303 | batch_size=256, hidden_features=64, learning_rate=0.02619, steps=200 |
| `real_magnitude` | 12 | 0.2566 | 0.2591 | batch_size=128, hidden_features=64, learning_rate=0.008588, steps=400 |
| `real_magnitude` | 13 | 0.2602 | 0.2569 | batch_size=512, hidden_features=32, learning_rate=0.001191, steps=800 |
| `real_magnitude` | 14 | 0.2447 | 0.2473 | batch_size=256, hidden_features=32, learning_rate=0.003545, steps=400 |
| `real_magnitude` | 15 | 0.1981 | 0.1965 | batch_size=128, hidden_features=16, learning_rate=0.002283, steps=400 |
