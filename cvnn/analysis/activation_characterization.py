"""Activation characterization utilities."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn

from cvnn.activations import (
    ComplexCardioid,
    ComplexTanh,
    CReLU,
    ModReLU,
    Siglog,
    ZReLU,
    complex_cardioid,
    complex_tanh,
    crelu,
    modrelu,
    siglog,
    zrelu,
)
from cvnn.nn import ComplexMLP

ActivationFunction = Callable[[Tensor], Tensor]
ActivationFactory = Callable[[], nn.Module]


@dataclass(frozen=True)
class ActivationSpec:
    """Activation metadata used by characterization reports."""

    name: str
    function: ActivationFunction
    module_factory: ActivationFactory
    edge_definition: str
    singularity_notes: str


@dataclass(frozen=True)
class ActivationCharacterization:
    """Scalar summary for one activation characterization run."""

    name: str
    grid_size: int
    extent: float
    finite_fraction: float
    blowup_fraction: float
    max_abs: float
    cr_median: float
    cr_p95: float
    cr_max: float
    gradient_norm_mean: float
    gradient_norm_std: float
    gradient_norm_min: float
    gradient_norm_max: float
    jacobian_norm_mean: float
    jacobian_norm_std: float
    jacobian_norm_min: float
    jacobian_norm_max: float
    edge_definition: str
    singularity_notes: str

    def to_dict(self) -> dict[str, float | int | str]:
        """Return a JSON-compatible representation."""

        return asdict(self)


def activation_specs() -> tuple[ActivationSpec, ...]:
    """Return the Phase 2 activation set in report order."""

    return (
        ActivationSpec(
            name="crelu",
            function=crelu,
            module_factory=CReLU,
            edge_definition="Applies ReLU independently to real and imaginary parts.",
            singularity_notes="Piecewise linear; nondifferentiable on real/imag axes.",
        ),
        ActivationSpec(
            name="zrelu",
            function=zrelu,
            module_factory=ZReLU,
            edge_definition="Keeps values with Re(z) >= 0 and Im(z) >= 0.",
            singularity_notes="Phase gate is discontinuous across quadrant boundaries.",
        ),
        ActivationSpec(
            name="modrelu",
            function=lambda z: modrelu(z, bias=-0.25),
            module_factory=lambda: ModReLU(init_bias=-0.25),
            edge_definition="Uses z / max(|z|, eps), so z = 0 maps to 0.",
            singularity_notes="Nondifferentiable at |z| + b = 0 and near z = 0.",
        ),
        ActivationSpec(
            name="complex_cardioid",
            function=complex_cardioid,
            module_factory=ComplexCardioid,
            edge_definition="Uses torch.angle(0) = 0, so z = 0 maps to 0.",
            singularity_notes=(
                "Continuous at zero but phase derivative is singular there."
            ),
        ),
        ActivationSpec(
            name="siglog",
            function=siglog,
            module_factory=Siglog,
            edge_definition="No division by |z|; z = 0 maps to 0.",
            singularity_notes="Smooth away from zero; bounded and non-holomorphic.",
        ),
        ActivationSpec(
            name="complex_tanh",
            function=complex_tanh,
            module_factory=ComplexTanh,
            edge_definition="Uses torch.tanh directly.",
            singularity_notes="Meromorphic with poles at z = i*pi*(k + 1/2).",
        ),
    )


def complex_grid(
    *,
    grid_size: int = 121,
    extent: float = 3.0,
    dtype: torch.dtype = torch.float64,
) -> tuple[Tensor, Tensor, Tensor]:
    """Create a square complex grid and its real coordinate vectors."""

    if grid_size < 3:
        msg = "grid_size must be at least 3 for finite differences"
        raise ValueError(msg)
    if extent <= 0:
        msg = "extent must be positive"
        raise ValueError(msg)

    values = torch.linspace(-extent, extent, grid_size, dtype=dtype)
    imag, real = torch.meshgrid(values, values, indexing="ij")
    return torch.complex(real, imag), values, values


def cauchy_riemann_residual(values: Tensor, dx: float, dy: float) -> Tensor:
    """Estimate the Cauchy-Riemann residual on a complex-valued grid."""

    if not torch.is_complex(values):
        msg = "Cauchy-Riemann residual expects complex values"
        raise TypeError(msg)
    if values.dim() != 2:
        msg = "Cauchy-Riemann residual expects a 2D grid"
        raise ValueError(msg)

    u = values.real
    v = values.imag
    residual = torch.full_like(u, float("nan"))
    u_x = (u[1:-1, 2:] - u[1:-1, :-2]) / (2.0 * dx)
    u_y = (u[2:, 1:-1] - u[:-2, 1:-1]) / (2.0 * dy)
    v_x = (v[1:-1, 2:] - v[1:-1, :-2]) / (2.0 * dx)
    v_y = (v[2:, 1:-1] - v[:-2, 1:-1]) / (2.0 * dy)
    residual[1:-1, 1:-1] = torch.sqrt((u_x - v_y).square() + (u_y + v_x).square())
    return residual


def characterize_activation(
    spec: ActivationSpec,
    *,
    grid_size: int = 121,
    extent: float = 3.0,
    gradient_seeds: Sequence[int] = (0, 1, 2, 3, 4),
    blowup_threshold: float = 1e3,
) -> ActivationCharacterization:
    """Compute scalar characterization metrics for one activation."""

    z, x_values, y_values = complex_grid(grid_size=grid_size, extent=extent)
    outputs = spec.function(z)
    dx = float(x_values[1] - x_values[0])
    dy = float(y_values[1] - y_values[0])
    cr_residual = cauchy_riemann_residual(outputs, dx=dx, dy=dy)
    output_abs = outputs.abs()
    finite_output = torch.isfinite(output_abs)
    finite_cr = cr_residual[torch.isfinite(cr_residual)]
    gradient_norms = gradient_norms_at_init(spec.module_factory, seeds=gradient_seeds)
    jacobian_norms = activation_jacobian_norms(
        spec.function, seeds=gradient_seeds, extent=extent
    )

    return ActivationCharacterization(
        name=spec.name,
        grid_size=grid_size,
        extent=extent,
        finite_fraction=_fraction(finite_output),
        blowup_fraction=_fraction(output_abs > blowup_threshold),
        max_abs=_safe_max(output_abs[finite_output]),
        cr_median=_safe_quantile(finite_cr, 0.50),
        cr_p95=_safe_quantile(finite_cr, 0.95),
        cr_max=_safe_max(finite_cr),
        gradient_norm_mean=_mean(gradient_norms),
        gradient_norm_std=_std(gradient_norms),
        gradient_norm_min=min(gradient_norms),
        gradient_norm_max=max(gradient_norms),
        jacobian_norm_mean=_mean(jacobian_norms),
        jacobian_norm_std=_std(jacobian_norms),
        jacobian_norm_min=min(jacobian_norms),
        jacobian_norm_max=max(jacobian_norms),
        edge_definition=spec.edge_definition,
        singularity_notes=spec.singularity_notes,
    )


def gradient_norms_at_init(
    activation_factory: ActivationFactory,
    *,
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
    batch_size: int = 32,
    in_features: int = 8,
    hidden_features: Sequence[int] = (16, 16),
    out_features: int = 4,
) -> list[float]:
    """Measure total parameter gradient norm for a fixed MLP at initialization."""

    norms: list[float] = []
    for seed in seeds:
        torch.manual_seed(seed)
        model = ComplexMLP(
            in_features,
            hidden_features,
            out_features,
            activation_factory=activation_factory,
            dtype=torch.complex64,
        )
        inputs = _complex_randn((batch_size, in_features), seed=10_000 + seed)
        loss = model(inputs).abs().square().mean()
        loss.backward()
        squared_norm = torch.tensor(0.0)
        for parameter in model.parameters():
            if parameter.grad is not None:
                squared_norm = squared_norm + parameter.grad.abs().square().sum()
        norms.append(float(torch.sqrt(squared_norm).item()))
    return norms


def activation_jacobian_norms(
    function: ActivationFunction,
    *,
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
    n_samples: int = 1024,
    extent: float = 3.0,
) -> list[float]:
    """Mean autograd-gradient magnitude of the activation in isolation.

    For each seed, samples `n_samples` complex inputs uniformly on the box
    `[-extent, extent]^2` and measures `|d/dz_bar (|f(z)|^2).sum()|` averaged
    over the batch. This isolates the activation's contribution to gradient
    flow from any surrounding MLP - unlike `gradient_norms_at_init`, which
    reports total parameter gradient norm of a reference model.
    """

    norms: list[float] = []
    for seed in seeds:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        z_real = (torch.rand(n_samples, generator=generator) * 2.0 - 1.0) * extent
        z_imag = (torch.rand(n_samples, generator=generator) * 2.0 - 1.0) * extent
        z = torch.complex(z_real, z_imag).requires_grad_(True)
        loss = function(z).abs().square().sum()
        (grad,) = torch.autograd.grad(loss, z)
        norms.append(float(grad.abs().mean().item()))
    return norms


def _complex_randn(shape: tuple[int, ...], *, seed: int) -> Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    real = torch.randn(shape, generator=generator)
    imag = torch.randn(shape, generator=generator)
    return torch.complex(real, imag)


def _fraction(mask: Tensor) -> float:
    return float(mask.to(torch.float64).mean().item())


def _safe_max(values: Tensor) -> float:
    if values.numel() == 0:
        return math.nan
    return float(values.max().item())


def _safe_quantile(values: Tensor, q: float) -> float:
    if values.numel() == 0:
        return math.nan
    return float(torch.quantile(values, q).item())


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values))


def _std(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance)


__all__ = [
    "ActivationCharacterization",
    "ActivationSpec",
    "activation_jacobian_norms",
    "activation_specs",
    "cauchy_riemann_residual",
    "characterize_activation",
    "complex_grid",
    "gradient_norms_at_init",
]
