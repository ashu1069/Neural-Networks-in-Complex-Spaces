# activation_modrelu

Question: How much does complex activation `modrelu` matter?

Contradiction signal: the complex result changes enough across activations that the broad 'complex NN' claim is underspecified.

Modulations: `['bpsk', 'qpsk', '8psk']`. SNR (dB): `[0, 10, 20]`. Architecture: `conv`. Activation: `modrelu`. Train transform: `none`. Test transform: `none`.

## Plots

![accuracy bar](accuracy_bar.png)

![accuracy by snr](accuracy_by_snr.png)


| model | hidden | params | MAdds | accuracy | std | 95% CI | loss | s/run |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| `complex` | 32 | 10888 | 2703744 | 0.8915 | 0.0081 | [0.8854, 0.8988] | 0.224 | 5.6 |
| `real_stacked` | 32 | 5603 | 696416 | 0.9102 | 0.0098 | [0.9031, 0.9187] | 0.198 | 0.93 |

## Accuracy by SNR (dB)

| model | 0 dB | 10 dB | 20 dB |
| --- | ---: | ---: | ---: |
| `complex` | 0.685 | 0.990 | 0.999 |
| `real_stacked` | 0.735 | 0.997 | 0.999 |
