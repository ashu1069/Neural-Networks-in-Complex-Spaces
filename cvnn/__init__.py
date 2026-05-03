"""Complex-valued neural network research utilities."""

from cvnn._version import __version__
from cvnn.activations import (
    ComplexCardioid,
    ComplexTanh,
    CReLU,
    ModReLU,
    Siglog,
    ZReLU,
)
from cvnn.baselines import (
    RealStackedLinear,
    complex_to_real_reparam,
    count_real_parameters,
)
from cvnn.layers import ComplexDropout, ComplexLinear
from cvnn.losses import complex_mse_loss, magnitude_mse_loss, phase_aware_loss
from cvnn.nn import ComplexMLP
from cvnn.optim import validate_pytorch_complex_autograd
from cvnn.repro import Environment, ResultManifest, collect_environment, new_manifest

__all__ = [
    "CReLU",
    "ComplexCardioid",
    "ComplexTanh",
    "ComplexDropout",
    "ComplexLinear",
    "ComplexMLP",
    "Environment",
    "ModReLU",
    "RealStackedLinear",
    "ResultManifest",
    "Siglog",
    "ZReLU",
    "__version__",
    "collect_environment",
    "complex_mse_loss",
    "complex_to_real_reparam",
    "count_real_parameters",
    "magnitude_mse_loss",
    "new_manifest",
    "phase_aware_loss",
    "validate_pytorch_complex_autograd",
]
