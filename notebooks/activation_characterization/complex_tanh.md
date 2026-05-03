# complex_tanh

- Grid: `121 x 121`
- Extent: `[-3.0, 3.0]`
- Finite output fraction: `1.000000`
- Blow-up fraction: `0.000000`
- Max `|f(z)|`: `48.0785`
- CR residual median: `0.000578168`
- CR residual p95: `0.0633163`
- CR residual max: `905.364`

## Edge Definition

Uses torch.tanh directly.

## Singularities / Blow-Ups

Meromorphic with poles at z = i*pi*(k + 1/2).

## Gradient Norms At Init (full reference MLP)

- Mean: `855.561`
- Std: `981.589`
- Min: `212.882`
- Max: `2494.65`

## Activation Jacobian Norms (intrinsic, no MLP)

- Mean: `107.558`
- Std: `194.461`
- Min: `9.48732`
- Max: `454.275`
