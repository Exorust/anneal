"""Matrix multiply for Apple GPUs.

The Metal Shading Language source is compiled at runtime through ``torch.mps.compile_shader``
(PyTorch 2.7+), so there is no build step and no Xcode requirement. Only the block between
the EVOLVE markers is rewritten by KernelFoundry.
"""

import torch

# [EVOLVE_START]
METAL_SOURCE = """
#include <metal_stdlib>
using namespace metal;

kernel void matmul(device const float* A [[buffer(0)]],
                   device const float* B [[buffer(1)]],
                   device float* C [[buffer(2)]],
                   constant uint& M [[buffer(3)]],
                   constant uint& K [[buffer(4)]],
                   constant uint& N [[buffer(5)]],
                   uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= N || gid.y >= M) return;
    float acc = 0.0f;
    for (uint k = 0; k < K; ++k) {
        acc += A[gid.y * K + k] * B[k * N + gid.x];
    }
    C[gid.y * N + gid.x] = acc;
}
"""

_lib = torch.mps.compile_shader(METAL_SOURCE)


def launch(A, B, C, M, K, N):
    """Dispatch the kernel: one thread per output element, 16x16 threadgroups."""
    _lib.matmul(A, B, C, M, K, N, threads=(N, M), group_size=(16, 16))


# [EVOLVE_END]


def metal_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """C = A @ B for float32 tensors on the mps device."""
    assert A.dim() == 2 and B.dim() == 2, "Inputs must be 2D"
    assert A.dtype == torch.float32 and B.dtype == torch.float32, "Only float32 supported"
    assert A.device.type == "mps" and B.device.type == "mps", "Tensors must be on the mps device"
    A = A.contiguous()
    B = B.contiguous()
    M, K = A.shape
    assert B.shape[0] == K, "Inner dimensions mismatch"
    N = B.shape[1]
    C = torch.empty((M, N), dtype=torch.float32, device=A.device)
    launch(A, B, C, M, K, N)
    return C
