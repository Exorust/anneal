import torch

# Metal Shading Language source, compiled at runtime by PyTorch (no build step).
METAL_SOURCE = """
#include <metal_stdlib>
using namespace metal;

kernel void vector_add(device const float* x [[buffer(0)]],
                       device const float* y [[buffer(1)]],
                       device float* out [[buffer(2)]],
                       constant uint& n_elements [[buffer(3)]],
                       uint gid [[thread_position_in_grid]]) {
    if (gid >= n_elements) return;
    out[gid] = x[gid] + y[gid];
}
"""

_lib = torch.mps.compile_shader(METAL_SOURCE)


def forward(x: torch.Tensor, y: torch.Tensor):
    assert x.shape == y.shape
    x = x.contiguous()
    y = y.contiguous()
    n_elements = x.numel()
    out = torch.empty_like(x)
    # threads = total grid size, group_size = threads per threadgroup (multiple of 32)
    _lib.vector_add(x, y, out, n_elements, threads=n_elements, group_size=256)
    return out


class ModelNew(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.elementwise_add = forward

    def forward(self, a, b, template_args=[]):
        return self.elementwise_add(a, b, *template_args)
