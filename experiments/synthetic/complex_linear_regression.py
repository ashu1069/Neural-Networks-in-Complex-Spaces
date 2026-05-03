"""Tiny complex linear regression convergence experiment."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, cast

import torch
from torch import Tensor

from cvnn.layers import ComplexLinear
from cvnn.losses import complex_mse_loss
from cvnn.repro import JsonObject, new_manifest

OptimizerName = Literal["adamw", "sgd"]


@dataclass(frozen=True)
class SyntheticLinearRegressionData:
    inputs: Tensor
    targets: Tensor
    true_weight: Tensor
    true_bias: Tensor


@dataclass(frozen=True)
class LinearRegressionResult:
    seed: int
    optimizer: str
    steps: int
    learning_rate: float
    final_loss: float
    closed_form_loss: float
    prediction_mse_to_closed_form: float
    weight_mse_to_closed_form: float
    bias_mse_to_closed_form: float

    def to_dict(self) -> JsonObject:
        return cast(JsonObject, asdict(self))


def make_complex_linear_regression(
    *,
    seed: int = 0,
    n_samples: int = 96,
    in_features: int = 3,
    out_features: int = 2,
    dtype: torch.dtype = torch.complex64,
    noise_std: float = 0.0,
) -> SyntheticLinearRegressionData:
    """Create a deterministic complex linear regression problem."""

    if noise_std < 0:
        msg = f"noise_std must be non-negative, got {noise_std}"
        raise ValueError(msg)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    inputs = _complex_randn((n_samples, in_features), generator=generator, dtype=dtype)
    true_weight = (
        _complex_randn(
            (out_features, in_features),
            generator=generator,
            dtype=dtype,
        )
        / in_features**0.5
    )
    true_bias = _complex_randn((out_features,), generator=generator, dtype=dtype) / 4.0
    targets = inputs.matmul(true_weight.transpose(-2, -1)) + true_bias
    if noise_std:
        targets = targets + noise_std * _complex_randn(
            targets.shape,
            generator=generator,
            dtype=dtype,
        )
    return SyntheticLinearRegressionData(
        inputs=inputs,
        targets=targets,
        true_weight=true_weight,
        true_bias=true_bias,
    )


def closed_form_complex_linear_regression(
    inputs: Tensor,
    targets: Tensor,
) -> tuple[Tensor, Tensor]:
    """Solve complex least squares for `targets = inputs @ weight.T + bias`."""

    if inputs.dim() != 2 or targets.dim() != 2:
        msg = "inputs and targets must be 2D tensors"
        raise ValueError(msg)
    if inputs.shape[0] != targets.shape[0]:
        msg = "inputs and targets must have the same number of samples"
        raise ValueError(msg)

    ones = torch.ones(
        inputs.shape[0],
        1,
        dtype=inputs.dtype,
        device=inputs.device,
    )
    design = torch.cat([inputs, ones], dim=1)
    solution = torch.linalg.lstsq(design, targets).solution
    weight = solution[:-1].transpose(0, 1).contiguous()
    bias = solution[-1].contiguous()
    return weight, bias


def train_complex_linear_regression(
    *,
    seed: int = 0,
    optimizer: OptimizerName = "adamw",
    steps: int = 800,
    learning_rate: float = 0.05,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.complex64,
) -> LinearRegressionResult:
    """Train a one-layer complex model and compare against closed form."""

    torch.manual_seed(seed)
    data = make_complex_linear_regression(seed=seed, dtype=dtype)
    inputs = data.inputs.to(device)
    targets = data.targets.to(device)
    closed_weight, closed_bias = closed_form_complex_linear_regression(inputs, targets)

    model = ComplexLinear(
        inputs.shape[1],
        targets.shape[1],
        dtype=dtype,
        device=device,
    )
    optimizer_instance = _make_optimizer(
        optimizer,
        model.parameters(),
        learning_rate=learning_rate,
    )

    for _ in range(steps):
        optimizer_instance.zero_grad()
        loss = complex_mse_loss(model(inputs), targets)
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer_instance.step()

    with torch.no_grad():
        predictions = model(inputs)
        closed_predictions = (
            inputs.matmul(closed_weight.transpose(-2, -1)) + closed_bias
        )
        final_loss = complex_mse_loss(predictions, targets)
        closed_form_loss = complex_mse_loss(closed_predictions, targets)
        prediction_mse = complex_mse_loss(predictions, closed_predictions)
        weight_mse = complex_mse_loss(model.weight, closed_weight)
        if model.bias is None:
            msg = "ComplexLinear unexpectedly has no bias"
            raise RuntimeError(msg)
        bias_mse = complex_mse_loss(model.bias, closed_bias)

    return LinearRegressionResult(
        seed=seed,
        optimizer=optimizer,
        steps=steps,
        learning_rate=learning_rate,
        final_loss=float(final_loss.item()),
        closed_form_loss=float(closed_form_loss.item()),
        prediction_mse_to_closed_form=float(prediction_mse.item()),
        weight_mse_to_closed_form=float(weight_mse.item()),
        bias_mse_to_closed_form=float(bias_mse.item()),
    )


def _make_optimizer(
    optimizer: OptimizerName,
    parameters: Iterable[Tensor],
    *,
    learning_rate: float,
) -> torch.optim.Optimizer:
    if optimizer == "adamw":
        return torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=0.0)
    if optimizer == "sgd":
        return torch.optim.SGD(parameters, lr=learning_rate)
    msg = f"unsupported optimizer: {optimizer}"
    raise ValueError(msg)


def _complex_randn(
    shape: tuple[int, ...],
    *,
    generator: torch.Generator,
    dtype: torch.dtype,
) -> Tensor:
    real_dtype = torch.float64 if dtype == torch.complex128 else torch.float32
    real = torch.randn(shape, generator=generator, dtype=real_dtype)
    imag = torch.randn(shape, generator=generator, dtype=real_dtype)
    return torch.complex(real, imag)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--optimizer", choices=["adamw", "sgd"], default="adamw")
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    result = train_complex_linear_regression(
        seed=args.seed,
        optimizer=args.optimizer,
        steps=args.steps,
        learning_rate=args.learning_rate,
        device=args.device,
    )
    payload = result.to_dict()
    print(json.dumps(payload, indent=2, sort_keys=True))

    if args.output is not None:
        manifest = new_manifest(
            run_id=f"synthetic-complex-linear-regression-seed-{args.seed}",
            config={
                "experiment": "synthetic_complex_linear_regression",
                "optimizer": args.optimizer,
                "steps": args.steps,
                "learning_rate": args.learning_rate,
            },
            seeds=[args.seed],
            metrics=payload,
            device=args.device,
            dtype="complex64",
            dataset={"name": "synthetic_complex_linear_regression", "version": "0.1.0"},
        )
        manifest.write_json(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
