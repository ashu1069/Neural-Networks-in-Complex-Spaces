"""Real-valued baselines for complex-network comparisons.

See `docs/baselines.md` for the four baseline families and the rules used to
match parameter count or FLOPs against a complex reference model.
"""

from cvnn.baselines.real_stacked import (
    RealStackedLinear,
    complex_to_real_reparam,
    count_real_parameters,
)
from cvnn.baselines.rotation_equivariant import (
    RotationEquivariantConv1d,
    complex_input_to_stacked,
    materialize_from_complex,
    stacked_to_complex_output,
)

__all__ = [
    "RealStackedLinear",
    "RotationEquivariantConv1d",
    "complex_input_to_stacked",
    "complex_to_real_reparam",
    "count_real_parameters",
    "materialize_from_complex",
    "stacked_to_complex_output",
]
