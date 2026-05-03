# modrelu

- Grid: `121 x 121`
- Extent: `[-3.0, 3.0]`
- Finite output fraction: `1.000000`
- Blow-up fraction: `0.000000`
- Max `|f(z)|`: `3.99264`
- CR residual median: `0.104853`
- CR residual p95: `0.31587`
- CR residual max: `0.876525`

## Edge Definition

Uses z / max(|z|, eps), so z = 0 maps to 0.

## Singularities / Blow-Ups

Nondifferentiable at |z| + b = 0 and near z = 0.

## Gradient Norms At Init (full reference MLP)

- Mean: `3.56529`
- Std: `0.820037`
- Min: `2.7496`
- Max: `4.50315`

## Activation Jacobian Norms (intrinsic, no MLP)

- Mean: `4.08012`
- Std: `0.0495732`
- Min: `4.0051`
- Max: `4.13195`
