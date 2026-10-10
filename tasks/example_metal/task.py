import pytest
import torch
from kernelfoundry import TestBase  # Tests must derive from TestBase

# The documentation and implementation of the fixtures can be found in kernelfoundry/conftest.py


# Markers for annotating the reference implementation for the LLM prompt
# [REFERENCE_START]
def reference_kernel(a, b):
    # Reference implementation of the kernel
    return torch.matmul(a, b)


# [REFERENCE_END]


@pytest.fixture(scope="session")
def device():
    return torch.device("mps")


@pytest.fixture(scope="session")
def kernel(use_reference):
    if use_reference:  # use_reference is True if --ref is passed as argument to pytest
        return reference_kernel
    else:
        import matmul_kernel

        return matmul_kernel.metal_matmul


def get_data(size, device):
    # data generation for correctness tests and benchmarking
    torch.manual_seed(42)
    a = torch.randn(size, size).to(device)
    b = torch.randn(size, size).to(device)
    c = reference_kernel(a, b)
    return a, b, c


class TestMatrixMultiplication(TestBase):
    """A custom task for Metal matrix multiplication kernels. No build step: the shader is
    compiled at runtime when matmul_kernel is imported."""

    def test_correctness_wrt_reference(self, kernel, device):
        """Tests the correctness of the kernel against the reference implementation."""
        a, b, c = get_data(1000, device)
        result = kernel(a, b)
        assert torch.allclose(result, c, rtol=1e-4, atol=1e-4), "Matrix multiplication result is incorrect."

    def test_correctness_identity(self, kernel, device):
        """Tests the correctness of the kernel when multiplying with an identity matrix."""
        a, _, _ = get_data(1000, device)
        b = torch.eye(a.shape[0], device=device)
        result = kernel(a, b)
        assert torch.allclose(result, a), "Multiplication with identity matrix failed."

    @pytest.mark.performance
    @pytest.mark.parametrize("size", [512, 1024, 2048])
    def test_benchmark(self, size, kernel, device, measure_runtime_torch):
        """Benchmark the runtime using the measure_runtime_torch fixture (milliseconds)."""
        a, b, _ = get_data(size, device)

        measure_runtime_torch(kernel, device, args=(a, b))
