"""Wiring checks for the Metal (Apple GPU) backend. No GPU needed except where marked."""

import pytest

from kernelfoundry.algorithm.evolve_database_optimization_aware import OptimizationFeatureClassifier
from kernelfoundry.algorithm.prompts.languages import KERNEL_OPTIMIZATION_TIPS, SUPPORTED_LANGUAGES
from kernelfoundry.algorithm.prompts.meta_prompting import get_default_evolvable_regions
from kernelfoundry.algorithm.prompts.optimization_aware import Backend, get_optimization_taxonomy_prompt
from kernelfoundry.eval_pipeline.profiler_command import get_profilers
from kernelfoundry.eval_pipeline.profiler_feedback import get_profiler_feedback_class
from kernelfoundry.eval_pipeline.tasks.task_runner import BUILD_SUPPORTED_LANGUAGES
from kernelfoundry.eval_pipeline.utils.gpu_specs import ARCH_TO_NAME, ARCH_TO_SPECS

NAIVE = """
kernel void k(device const float* A [[buffer(0)]], device float* C [[buffer(1)]],
              constant uint& N [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= N) return;
    C[gid] = A[gid] * 2.0f;
}
"""
TILED = """
kernel void k(device const float* A [[buffer(0)]], device float* C [[buffer(1)]],
              uint2 lid [[thread_position_in_threadgroup]]) {
    threadgroup float As[16][17];
    As[lid.y][lid.x] = A[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float acc = 0.0f;
    #pragma unroll
    for (uint k = 0; k < 16; ++k) acc = fma(As[lid.y][k], As[k][lid.x], acc);
    simdgroup_float8x8 a;
    C[0] = acc;
}
"""


def test_metal_is_registered_everywhere():
    assert "metal" in SUPPORTED_LANGUAGES
    assert "metal" in BUILD_SUPPORTED_LANGUAGES
    assert len(KERNEL_OPTIMIZATION_TIPS["metal"]) > 2  # template manager samples n_tips=2
    assert Backend.from_string("metal") is Backend.METAL
    assert ARCH_TO_SPECS["apple_m5"] and ARCH_TO_NAME["apple"]
    assert set(get_default_evolvable_regions("metal")) == set(get_default_evolvable_regions("triton"))


def test_metal_has_no_profiler_but_does_not_crash():
    assert get_profilers("metal", "apple_m5") == []
    fb = get_profiler_feedback_class("metal", "apple_m5")()
    assert fb.name == "none"
    assert fb.collate_and_create_feedback({}, worker_info={}) == ({}, "")


def test_metal_prompt_text_is_metal_not_sycl():
    text = get_optimization_taxonomy_prompt(backend="metal")
    assert "simdgroup" in text and "sycl::" not in text


def test_metal_classifier_separates_levels():
    naive = OptimizationFeatureClassifier.classify_from_code(NAIVE, "metal")
    tiled = OptimizationFeatureClassifier.classify_from_code(TILED, "metal")
    assert naive == (0, 0, 0, 0)
    memory, compute, parallelism, _ = tiled
    assert memory >= 2 and compute >= 1 and parallelism >= 1


@pytest.mark.skipif(not __import__("torch").backends.mps.is_available(), reason="needs an Apple GPU")
def test_measure_runtime_torch_on_mps():
    import torch

    from kernelfoundry.eval_pipeline.utils.performance import measure_runtime_torch

    x = torch.randn(1 << 16, device="mps")
    times = measure_runtime_torch(torch.sin, "mps", args=(x,), warmup_min_time=0.05, perf_trials_min_time=0.05)
    assert times and all(t > 0 for t in times)


def test_platform_defaults_fill_metal_on_apple(monkeypatch):
    from omegaconf import OmegaConf

    from kernelfoundry.algorithm import __main__ as cli

    monkeypatch.setattr(cli.sys, "platform", "darwin")
    monkeypatch.setattr("kernelfoundry.eval_pipeline.utils.sysinfo.get_apple_gpu_arch", lambda: "apple_m5")
    import torch

    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    cfg = cli._apply_platform_defaults(OmegaConf.create({"language": None, "gpu_arch": None}))
    assert (cfg.language, cfg.gpu_arch) == ("metal", "apple_m5")
    # explicit values are never overridden
    cfg = cli._apply_platform_defaults(OmegaConf.create({"language": "SYCL", "gpu_arch": "lnl"}))
    assert (cfg.language, cfg.gpu_arch) == ("SYCL", "lnl")
