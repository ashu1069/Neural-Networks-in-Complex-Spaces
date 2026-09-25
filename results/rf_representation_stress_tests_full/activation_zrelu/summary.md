# activation_zrelu

Question: How much does complex activation `zrelu` matter?

Contradiction signal: the complex result changes enough across activations that the broad 'complex NN' claim is underspecified.

Modulations: `['bpsk', 'qpsk', '8psk']`. SNR (dB): `[0, 10, 20]`. Architecture: `conv`. Activation: `zrelu`. Train transform: `none`. Test transform: `none`.

## Plots

![accuracy bar](accuracy_bar.png)

![accuracy by snr](accuracy_by_snr.png)


| model | hidden | params | MAdds | accuracy | std | 95% CI | loss | s/run |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| `complex` | 32 | 10886 | 2703744 | 0.9137 | 0.0058 | [0.9096, 0.9180] | 0.201 | 3.4 |
| `real_stacked` | 32 | 5603 | 696416 | 0.9113 | 0.0109 | [0.9033, 0.9206] | 0.198 | 1.4 |

## Accuracy by SNR (dB)

| model | 0 dB | 10 dB | 20 dB |
| --- | ---: | ---: | ---: |
| `complex` | 0.753 | 0.990 | 0.999 |
| `real_stacked` | 0.739 | 0.997 | 0.999 |
