"""Small model builders for complex-valued experiments."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
import torch.nn as torch_nn
from torch import Tensor

from cvnn.layers import ComplexDropout, ComplexLinear

ActivationFactory = Callable[[], torch_nn.Module]


class ComplexMLP(torch_nn.Module):
    """A minimal complex-valued multilayer perceptron."""

    def __init__(
        self,
        in_features: int,
        hidden_features: Sequence[int],
        out_features: int,
        *,
        activation_factory: ActivationFactory | None = None,
        dropout: float = 0.0,
        bias: bool = True,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if dropout < 0.0 or dropout > 1.0:
            msg = f"dropout probability has to be between 0 and 1, got {dropout}"
            raise ValueError(msg)

        dims = [in_features, *hidden_features, out_features]
        if any(dim <= 0 for dim in dims):
            msg = "all MLP feature dimensions must be positive"
            raise ValueError(msg)

        layers: list[torch_nn.Module] = []
        last_layer_index = len(dims) - 2
        for index, (current_features, next_features) in enumerate(
            zip(dims[:-1], dims[1:], strict=True)
        ):
            layers.append(
                ComplexLinear(
                    current_features,
                    next_features,
                    bias=bias,
                    device=device,
                    dtype=dtype,
                )
            )
            if index != last_layer_index:
                if activation_factory is not None:
                    layers.append(activation_factory())
                if dropout > 0.0:
                    layers.append(ComplexDropout(dropout))

        self.net = torch_nn.Sequential(*layers)

    def forward(self, input: Tensor) -> Tensor:
        output: Tensor = self.net(input)
        return output


__all__ = ["ActivationFactory", "ComplexMLP"]
