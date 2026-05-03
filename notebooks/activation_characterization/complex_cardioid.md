# complex_cardioid

- Grid: `121 x 121`
- Extent: `[-3.0, 3.0]`
- Finite output fraction: `1.000000`
- Blow-up fraction: `0.000000`
- Max `|f(z)|`: `3.62132`
- CR residual median: `0.353406`
- CR residual p95: `0.497079`
- CR residual max: `0.499928`

## Edge Definition

Uses torch.angle(0) = 0, so z = 0 maps to 0.

## Singularities / Blow-Ups

Continuous at zero but phase derivative is singular there.

## Gradient Norms At Init (full reference MLP)

- Mean: `0.772534`
- Std: `0.139432`
- Min: `0.609342`
- Max: `0.989738`

## Activation Jacobian Norms (intrinsic, no MLP)

- Mean: `1.94862`
- Std: `0.0668123`
- Min: `1.86966`
- Max: `2.04759`
