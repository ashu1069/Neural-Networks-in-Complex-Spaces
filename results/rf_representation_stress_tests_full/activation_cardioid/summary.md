# activation_cardioid

Question: How much does complex activation `cardioid` matter?

Contradiction signal: the complex result changes enough across activations that the broad 'complex NN' claim is underspecified.

Modulations: `['bpsk', 'qpsk', '8psk']`. SNR (dB): `[0, 10, 20]`. Architecture: `conv`. Activation: `cardioid`. Train transform: `none`. Test transform: `none`.

## Plots

![accuracy bar](accuracy_bar.png)

![accuracy by snr](accuracy_by_snr.png)


| model | hidden | params | MAdds | accuracy | std | 95% CI | loss | s/run |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| `complex` | 32 | 10886 | 2703744 | 0.9294 | 0.0054 | [0.9251, 0.9331] | 0.165 | 4.3 |
| `real_stacked` | 32 | 5603 | 696416 | 0.9111 | 0.0106 | [0.9033, 0.9202] | 0.198 | 1 |

## Accuracy by SNR (dB)

| model | 0 dB | 10 dB | 20 dB |
| --- | ---: | ---: | ---: |
| `complex` | 0.788 | 1.000 | 1.000 |
| `real_stacked` | 0.739 | 0.996 | 0.999 |
