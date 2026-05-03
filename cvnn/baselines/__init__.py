"""Real-valued baselines for complex-network comparisons.

See `docs/baselines.md` for the four baseline families and the rules used to
match parameter count or FLOPs against a complex reference model.
"""

from cvnn.baselines.real_stacked import (
    RealStackedLinear,
    complex_to_real_reparam,
    count_real_parameters,
)

__all__ = [
    "RealStackedLinear",
    "complex_to_real_reparam",
    "count_real_parameters",
]
