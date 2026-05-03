"""Audit PyTorch complex tensor support on CPU, Apple Silicon MPS, or CUDA.

Run with:

    uv run python scripts/check_torch_complex_support.py

For a stricter MPS audit, disable implicit CPU fallback before running:

    PYTORCH_ENABLE_MPS_FALLBACK=0 uv run python scripts/check_torch_complex_support.py
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

type CheckFn = Callable[[torch.device, torch.dtype], tuple[Tensor, Tensor]]


@dataclass(frozen=True)
class CheckResult:
    name: str
    device: str
    dtype: str
    status: str
    seconds: float
    max_abs_error: float | None = None
    message: str = ""


def _complex_randn(
    shape: tuple[int, ...],
    *,
    seed: int,
    dtype: torch.dtype,
) -> Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
    real = torch.randn(shape, generator=generator, dtype=real_dtype)
    imag = torch.randn(shape, generator=generator, dtype=real_dtype)
    return torch.complex(real, imag)


def _max_abs_error(reference: Tensor, candidate: Tensor) -> float:
    return float((reference.cpu() - candidate.cpu()).abs().max().item())


def _compare_tensors(
    reference: Tensor,
    candidate: Tensor,
    *,
    atol: float,
    rtol: float,
) -> tuple[bool, float]:
    error = _max_abs_error(reference, candidate)
    is_close = bool(
        torch.allclose(reference.cpu(), candidate.cpu(), atol=atol, rtol=rtol)
    )
    return is_close, error


def _run_check(
    name: str,
    check: CheckFn,
    *,
    device: torch.device,
    dtype: torch.dtype,
    atol: float,
    rtol: float,
) -> CheckResult:
    start = time.perf_counter()
    try:
        reference, candidate = check(device, dtype)
        passed, error = _compare_tensors(reference, candidate, atol=atol, rtol=rtol)
    except Exception as exc:  # noqa: BLE001 - this is an audit script.
        return CheckResult(
            name=name,
            device=str(device),
            dtype=str(dtype).removeprefix("torch."),
            status="error",
            seconds=time.perf_counter() - start,
            message=f"{type(exc).__name__}: {exc}",
        )

    return CheckResult(
        name=name,
        device=str(device),
        dtype=str(dtype).removeprefix("torch."),
        status="pass" if passed else "fail",
        seconds=time.perf_counter() - start,
        max_abs_error=error,
        message="" if passed else "candidate differed from CPU reference",
    )


def _np_complex_dtype(dtype: torch.dtype) -> type:
    return np.complex64 if dtype == torch.complex64 else np.complex128


def _matmul(device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
    left = _complex_randn((4, 6), seed=1, dtype=dtype)
    right = _complex_randn((6, 5), seed=2, dtype=dtype)
    np_ref = np.asarray(left.numpy()) @ np.asarray(right.numpy())
    reference = torch.from_numpy(np_ref.astype(_np_complex_dtype(dtype)))
    candidate = left.to(device) @ right.to(device)
    return reference, candidate


def _linear(device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
    inputs = _complex_randn((3, 5), seed=3, dtype=dtype)
    weight = _complex_randn((4, 5), seed=4, dtype=dtype)
    bias = _complex_randn((4,), seed=5, dtype=dtype)
    np_ref = inputs.numpy() @ weight.numpy().T + bias.numpy()
    reference = torch.from_numpy(np_ref.astype(_np_complex_dtype(dtype)))
    candidate = F.linear(inputs.to(device), weight.to(device), bias.to(device))
    return reference, candidate


def _conv1d(device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
    inputs = _complex_randn((2, 3, 16), seed=6, dtype=dtype)
    weight = _complex_randn((4, 3, 3), seed=7, dtype=dtype)
    bias = _complex_randn((4,), seed=8, dtype=dtype)
    reference = F.conv1d(inputs, weight, bias, padding=1)
    candidate = F.conv1d(
        inputs.to(device), weight.to(device), bias.to(device), padding=1
    )
    return reference, candidate


def _conv2d(device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
    inputs = _complex_randn((2, 2, 8, 8), seed=9, dtype=dtype)
    weight = _complex_randn((3, 2, 3, 3), seed=10, dtype=dtype)
    bias = _complex_randn((3,), seed=11, dtype=dtype)
    reference = F.conv2d(inputs, weight, bias, padding=1)
    candidate = F.conv2d(
        inputs.to(device), weight.to(device), bias.to(device), padding=1
    )
    return reference, candidate


def _fft(device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
    inputs = _complex_randn((4, 16), seed=12, dtype=dtype)
    np_ref = np.fft.fft(inputs.numpy(), axis=-1)
    reference = torch.from_numpy(np_ref.astype(_np_complex_dtype(dtype)))
    candidate = torch.fft.fft(inputs.to(device), dim=-1)
    return reference, candidate


def _autograd(device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
    cpu_inputs = _complex_randn((8,), seed=13, dtype=dtype).requires_grad_(True)
    cpu_loss = (cpu_inputs.abs() ** 2).sum()
    cpu_loss.backward()  # type: ignore[no-untyped-call]
    if cpu_inputs.grad is None:
        msg = "CPU autograd did not populate gradients"
        raise RuntimeError(msg)

    device_inputs = cpu_inputs.detach().to(device).requires_grad_(True)
    device_loss = (device_inputs.abs() ** 2).sum()
    device_loss.backward()  # type: ignore[no-untyped-call]
    if device_inputs.grad is None:
        msg = f"{device} autograd did not populate gradients"
        raise RuntimeError(msg)

    return cpu_inputs.grad, device_inputs.grad


def _complex_scalar_mul(
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor]:
    real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
    inputs = torch.arange(4, dtype=real_dtype).reshape(2, 2) / 3
    np_ref = inputs.numpy() * 1j
    reference = torch.from_numpy(np_ref.astype(_np_complex_dtype(dtype)))
    candidate = inputs.to(device) * 1j
    return reference, candidate


def _crelu(device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
    inputs = _complex_randn((5, 5), seed=14, dtype=dtype)
    np_in = inputs.numpy()
    np_ref = np.maximum(np_in.real, 0) + 1j * np.maximum(np_in.imag, 0)
    reference = torch.from_numpy(np_ref.astype(_np_complex_dtype(dtype)))
    return reference, _crelu_impl(inputs.to(device))


def _siglog(device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
    inputs = _complex_randn((5, 5), seed=15, dtype=dtype)
    np_in = inputs.numpy()
    np_ref = np_in / (1 + np.abs(np_in))
    reference = torch.from_numpy(np_ref.astype(_np_complex_dtype(dtype)))
    return reference, _siglog_impl(inputs.to(device))


def _cardioid(device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
    inputs = _complex_randn((5, 5), seed=16, dtype=dtype)
    np_in = inputs.numpy()
    np_ref = 0.5 * (1 + np.cos(np.angle(np_in))) * np_in
    reference = torch.from_numpy(np_ref.astype(_np_complex_dtype(dtype)))
    return reference, _cardioid_impl(inputs.to(device))


def _crelu_impl(inputs: Tensor) -> Tensor:
    return torch.complex(torch.relu(inputs.real), torch.relu(inputs.imag))


def _siglog_impl(inputs: Tensor) -> Tensor:
    return inputs / (1 + inputs.abs())


def _cardioid_impl(inputs: Tensor) -> Tensor:
    return 0.5 * (1 + torch.cos(torch.angle(inputs))) * inputs


def _available_target(device_name: str) -> tuple[torch.device | None, str | None]:
    if device_name == "cpu":
        return torch.device("cpu"), None
    if device_name == "mps":
        if not torch.backends.mps.is_available():
            return None, "MPS is not available in this PyTorch environment"
        return torch.device("mps"), None
    if device_name == "cuda":
        if not torch.cuda.is_available():
            return None, "CUDA is not available in this PyTorch environment"
        return torch.device("cuda"), None
    msg = f"unsupported device: {device_name}"
    raise ValueError(msg)


def _format_table(results: list[CheckResult]) -> str:
    rows = ["name | device | dtype | status | max_abs_error | seconds | message"]
    rows.append("--- | --- | --- | --- | ---: | ---: | ---")
    for result in results:
        error = "" if result.max_abs_error is None else f"{result.max_abs_error:.3e}"
        rows.append(
            " | ".join(
                [
                    result.name,
                    result.device,
                    result.dtype,
                    result.status,
                    error,
                    f"{result.seconds:.3f}",
                    result.message,
                ]
            )
        )
    return "\n".join(rows)


def _metadata() -> dict[str, str | bool | None]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "mps_available": torch.backends.mps.is_available(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cuda_device": _cuda_device_name(),
        "pytorch_enable_mps_fallback": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"),
    }


def _cuda_device_name() -> str | None:
    if not torch.cuda.is_available():
        return None
    return str(torch.cuda.get_device_name(0))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="mps")
    parser.add_argument(
        "--dtype", choices=["complex64", "complex128"], default="complex64"
    )
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument(
        "--json", action="store_true", help="Emit machine-readable JSON."
    )
    parser.add_argument(
        "--fail-on-gap",
        action="store_true",
        help="Exit nonzero when the target device fails or differs from CPU.",
    )
    args = parser.parse_args()

    dtype = torch.complex64 if args.dtype == "complex64" else torch.complex128
    checks: dict[str, CheckFn] = {
        "matmul": _matmul,
        "linear": _linear,
        "conv1d": _conv1d,
        "conv2d": _conv2d,
        "fft": _fft,
        "autograd": _autograd,
        "issue_119088_complex_scalar_mul": _complex_scalar_mul,
        "activation_crelu": _crelu,
        "activation_siglog": _siglog,
        "activation_cardioid": _cardioid,
    }

    results = [
        _run_check(
            name,
            check,
            device=torch.device("cpu"),
            dtype=dtype,
            atol=args.atol,
            rtol=args.rtol,
        )
        for name, check in checks.items()
    ]

    target_device, skip_message = _available_target(args.device)
    if target_device is None:
        results.extend(
            CheckResult(
                name=name,
                device=args.device,
                dtype=args.dtype,
                status="skipped",
                seconds=0.0,
                message=skip_message or "",
            )
            for name in checks
        )
    elif target_device.type != "cpu":
        results.extend(
            _run_check(
                name,
                check,
                device=target_device,
                dtype=dtype,
                atol=args.atol,
                rtol=args.rtol,
            )
            for name, check in checks.items()
        )

    payload = {
        "metadata": _metadata(),
        "results": [asdict(result) for result in results],
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(_format_table(results))

    cpu_failed = any(
        result.device == "cpu" and result.status != "pass" for result in results
    )
    target_failed = any(result.status not in {"pass", "skipped"} for result in results)
    if cpu_failed or (args.fail_on_gap and target_failed):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
