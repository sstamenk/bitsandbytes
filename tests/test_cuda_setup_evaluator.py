from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from bitsandbytes.cextension import get_cuda_bnb_library_path
from bitsandbytes.consts import DYNAMIC_LIBRARY_SUFFIX
from bitsandbytes.cuda_specs import CUDASpecs


def specs(version: tuple[int, int]) -> CUDASpecs:
    return CUDASpecs(
        cuda_version_string=f"{version[0]}{version[1]}",
        highest_compute_capability=(0, 0),
        cuda_version_tuple=version,
    )


@pytest.mark.parametrize(
    "backend,backend_version,runtime_version,available,expected,warning",
    [
        # Exact match.
        ("cuda", "12.4", (12, 4), [(12, 4)], (12, 4), False),
        # Same-major fallback to the oldest newer binary.
        ("cuda", "12.0", (12, 0), [(12, 1), (12, 4)], (12, 1), True),
        # Same-major fallback to the newest older binary.
        ("cuda", "12.9", (12, 9), [(12, 4), (12, 8)], (12, 8), True),
        # ROCm same-major fallback with a double-digit minor.
        ("hip", "7.13.0", (7, 13), [(7, 2)], (7, 2), True),
        # ROCm same-major fallback prefers the newest older binary.
        ("hip", "7.9.0", (7, 9), [(7, 2), (7, 14)], (7, 2), True),
        # ROCm cross-major fallback to the newest older binary.
        ("hip", "8.0.0", (8, 0), [(7, 14)], (7, 14), True),
        # ROCm cross-major fallback to the oldest newer binary.
        ("hip", "6.4.0", (6, 4), [(7, 0)], (7, 0), True),
        # CUDA does not fall back across major versions.
        ("cuda", "11.8", (11, 8), [(12, 1), (12, 4)], None, False),
        # No packaged libraries returns the requested path without a warning.
        ("cuda", "12.9", (12, 9), [], None, False),
        ("hip", "7.14.0", (7, 14), [], None, False),
    ],
)
def test_version_selection(
    monkeypatch,
    caplog,
    backend,
    backend_version,
    runtime_version,
    available,
    expected,
    warning,
):
    """Library selection: exact match, fallback, no-same-major, no-libs."""
    monkeypatch.delenv("BNB_CUDA_VERSION", raising=False)
    monkeypatch.delenv("BNB_ROCM_VERSION", raising=False)
    other_backend = "cuda" if backend == "hip" else "hip"
    prefix = "rocm" if backend == "hip" else "cuda"
    paths = {
        version: Path(f"libbitsandbytes_{prefix}{version[0]}{version[1]}{DYNAMIC_LIBRARY_SUFFIX}")
        for version in available
    }
    # ROCm release metadata is deliberately unrelated to HIP-based library selection.
    with (
        patch.object(torch.version, backend, backend_version),
        patch.object(torch.version, other_backend, None),
        patch.object(torch.version, "rocm", "10.0.0", create=True),
        patch("bitsandbytes.cextension._find_cuda_libs", return_value=paths),
        caplog.at_level("WARNING"),
    ):
        result = get_cuda_bnb_library_path(specs(runtime_version))

    if expected is None:
        tag = f"{runtime_version[0]}{runtime_version[1]}"
        assert result.name == f"libbitsandbytes_{prefix}{tag}{DYNAMIC_LIBRARY_SUFFIX}"
    else:
        assert result == paths[expected]
    assert bool(caplog.text) is warning


@pytest.mark.parametrize(
    "backend,backend_version,runtime_version,override,expected_stem",
    [
        ("hip", "7.0.0", (7, 0), "72", "libbitsandbytes_rocm72"),
        ("hip", "7.0.0", (7, 0), "7.2", "libbitsandbytes_rocm72"),
        ("hip", "7.0.0", (7, 0), "714", "libbitsandbytes_rocm714"),
        ("hip", "10.0.0", (10, 0), "1014", "libbitsandbytes_rocm1014"),
        ("cuda", "12.0", (12, 0), "128", "libbitsandbytes_cuda128"),
        ("cuda", "12.0", (12, 0), "12.8", "libbitsandbytes_cuda128"),
        ("cuda", "12.0", (12, 0), "12.8.1", "libbitsandbytes_cuda128"),
    ],
)
def test_override_formats(monkeypatch, caplog, backend, backend_version, runtime_version, override, expected_stem):
    other_backend = "cuda" if backend == "hip" else "hip"
    override_var = "BNB_ROCM_VERSION" if backend == "hip" else "BNB_CUDA_VERSION"
    other_override_var = "BNB_CUDA_VERSION" if backend == "hip" else "BNB_ROCM_VERSION"
    monkeypatch.setenv(override_var, override)
    monkeypatch.delenv(other_override_var, raising=False)
    with (
        patch.object(torch.version, backend, backend_version),
        patch.object(torch.version, other_backend, None),
        patch("bitsandbytes.cextension._find_cuda_libs", return_value={}),
        caplog.at_level("WARNING"),
    ):
        result = get_cuda_bnb_library_path(specs(runtime_version))
    assert result.stem == expected_stem
    assert override_var in caplog.text


@pytest.mark.parametrize(
    "backend,torch_version,runtime_version",
    [("cuda", "12.0", (12, 0)), ("hip", "7.2.0", (7, 2))],
)
def test_override_invalid_format(monkeypatch, backend, torch_version, runtime_version):
    """Reject malformed overrides for both backends."""
    other_backend = "cuda" if backend == "hip" else "hip"
    override_var = "BNB_ROCM_VERSION" if backend == "hip" else "BNB_CUDA_VERSION"
    other_override_var = "BNB_CUDA_VERSION" if backend == "hip" else "BNB_ROCM_VERSION"
    monkeypatch.setenv(override_var, "not-a-version")
    monkeypatch.delenv(other_override_var, raising=False)
    with (
        patch.object(torch.version, backend, torch_version),
        patch.object(torch.version, other_backend, None),
        pytest.raises(RuntimeError, match="dotted version"),
    ):
        get_cuda_bnb_library_path(specs(runtime_version))


@pytest.mark.parametrize(
    "backend,backend_version,runtime_version,wrong_var,correct_var",
    [
        ("cuda", "12.0", (12, 0), "BNB_ROCM_VERSION", "BNB_CUDA_VERSION"),
        ("hip", "7.2.0", (7, 2), "BNB_CUDA_VERSION", "BNB_ROCM_VERSION"),
    ],
)
def test_opposite_backend_override_warns(
    monkeypatch, caplog, backend, backend_version, runtime_version, wrong_var, correct_var
):
    other_backend = "cuda" if backend == "hip" else "hip"
    monkeypatch.setenv(wrong_var, "72")
    monkeypatch.delenv(correct_var, raising=False)
    with (
        patch.object(torch.version, backend, backend_version),
        patch.object(torch.version, other_backend, None),
        patch("bitsandbytes.cextension._find_cuda_libs", return_value={}),
        caplog.at_level("WARNING"),
    ):
        get_cuda_bnb_library_path(specs(runtime_version))
    assert f"{wrong_var} is ignored" in caplog.text
    assert f"use {correct_var} instead" in caplog.text
