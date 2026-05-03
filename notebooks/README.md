# Notebooks

Exploratory notebooks and activation characterization reports will live here.

Notebook outputs used in the paper should be backed by scripts or committed
result manifests so the analysis can be reproduced from a clean checkout.

Phase 2 activation characterization reports live in
`notebooks/activation_characterization/` and are regenerated with:

```bash
uv run python scripts/characterize_activations.py
```
