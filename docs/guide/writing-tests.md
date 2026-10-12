# Writing tests

The tests are the specification. They decide whether a generated kernel is accepted as correct, and
they produce the runtime that speedup is measured from, so the quality of a KernelFoundry run depends
directly on them.

A coding agent normally writes them, reusing your existing tests where they exist; see
[Optimize a kernel with a coding agent](agentic-workflow.md). This page is for writing them yourself,
and for judging whether what an agent produced measures the right thing, which is worth a look either
way because everything downstream rests on it.

The framework is built on `pytest`. KernelFoundry supplies fixtures and helpers that remove
most of the boilerplate; see [Public API](../api/public.rst) for the full list.

**Naming:** tests must start with `test_`, per pytest. Every test is treated as a correctness
test unless it carries the `@pytest.mark.performance` decorator.

## Correctness tests

At least one is required. Tests live in a class deriving from `TestBase`.

The pattern that works for most tasks is to run the kernel and the reference on the same random
inputs and compare:

```python
from kernelfoundry.testing import assert_allclose

def test_correctness(self, data, kernel):
    x, y = data
    result = kernel(x, y)
    expected_result = reference(x, y)
    assert_allclose(result, expected_result)
```

How it works:

- A fixture creates random input data, for example two torch tensors `x`, `y`.
- The kernel and the reference both run on that data.
- `assert_allclose` compares the results with tolerances suited to GPU floating point.

For most tasks you can use this as-is and only change the `reference` function, the `kernel`
fixture (which imports the compiled kernel module), and the `data` fixture.

:::{tip}
Make the input data representative. A kernel tuned against a single small shape will often
regress on the shapes you actually run in production. Parametrize with `@pytest.mark.parametrize`
to cover several shapes.
:::

## Benchmark test

You must define **exactly one** benchmark, marked with `@pytest.mark.performance`. Additional
performance-marked functions will run but are ignored when computing runtime and speedup.

The interface between your benchmark and the framework is the `profile_store` fixture, which
records the measured runtimes. The simplest correct benchmark uses the provided helper:

```python
@pytest.mark.performance
def test_benchmark_my_kernel(self, kernel, device, data, measure_runtime_torch):
    measure_runtime_torch(kernel, device, args=data)
```

`measure_runtime_torch` handles moving inputs to the device, warmup trials, and storing results
through `profile_store`. For non-torch workloads use `measure_runtime`.

### Paired benchmark (recommended on laptops and shared machines)

`measure_runtime` times the candidate alone; the reference is timed once per job in a separate
`--ref` process and the speedup is a ratio of the two means. Anything that drifts between those
two runs (GPU thermal state, a browser tab, the fan curve) lands on one side of the ratio. On an
Apple laptop that is a 10-30% error, larger than most real kernel improvements.

`measure_runtime_paired` times the candidate **interleaved** with one or more baselines in the
same process (A/B/A/B, order alternated every trial) and reports the **median of per-trial
ratios**, so drift cancels. It also records the coefficient of variation of every series and,
when given the traffic per call and a probed copy bandwidth, the fraction of the memory roofline
the candidate reaches:

```python
@pytest.mark.performance
def test_benchmark(self, kernel, measure_runtime_paired):
    import mlx.core as mx
    import best_so_far  # a frozen copy of the current champion, shipped with the task

    def timed(fn):
        def run(x):
            mx.eval(fn(x))      # MLX is lazy: force the work inside the timed call
        return run

    measure_runtime_paired(
        timed(kernel),
        baselines={"champion": timed(best_so_far.kernel), "reference": timed(reference_kernel)},
        sync_fn=mx.synchronize,
        args=(x,),
        bytes_per_call=x.nbytes * 3,      # optional: minimal traffic of one call
        peak_bandwidth_gbs=125.0,         # optional: probed copy bandwidth, not the datasheet
    )
```

The first baseline is the primary one and defines `speedup` (and so the geomean fitness); the
others are reported next to it in the eval log and in `runtime_stats`. Pairing against the
current champion rather than the slow framework reference is the point: a 1% regression against
the champion must not read as a 3x win against the reference. With a paired benchmark the
separate reference timing pass is not needed, so set `test_reference: false` in `config.yaml`.
The eval log line looks like

```
test_benchmark[T16384]: 49.400 ms, 1.004x vs champion (49.600 ms), 3.816x vs scan (188.000 ms), 13% of copy roofline [noisy: cv 18%]
```

and `[noisy: cv N%]` appears when any series has a coefficient of variation above 8%: treat that
speedup as uncertain and rerun. Unlike `measure_runtime`, the paired helper does not replicate
inputs to defeat caches; pass a list of inputs yourself if cache residency matters for your kernel.

### Writing a custom benchmark

If you need to measure something the helpers don't cover, your function must:

- use the **`profile_store`** fixture to store results as a list of runtimes; and
- run the timed trials inside the **`profiler_session`** context. Otherwise profiling silently
  produces nothing, because the profiler has no entry point to attach to.

That second requirement is the usual cause of a run that reports timings but no profiler
feedback.

### Hardening options

- **Rotate inputs.** Pass `arg_sets=[args_a, args_b]` instead of `args`; calls cycle through the
  sets and every series sees the same sequence. A kernel that memoizes on input identity gains
  nothing. Pre-evaluate the inputs so allocation is not timed.
- **Recheck after timing.** Compare the candidate's output on one of the sets against the baseline
  after the timed loop; a kernel that validates once and then returns cached results fails here.
- **Noise-adaptive trials.** After the minimum rounds the fixture keeps adding rounds while the
  robust coefficient of variation of the per-trial ratio candidate/primary is above `cv_target`
  (default 3%), capped by `max_time` (10 s) and `perf_trials_max_iters` (200). The record carries
  `ratio_cv`.
- **Held-out shape gate.** Put one or two shapes the prompt does not describe in a *plain* test
  (not `@pytest.mark.performance`), call `measure_runtime_paired` directly with the baseline only,
  and assert the speedup is above a floor such as 0.97. A failing correctness-phase test marks the
  candidate incorrect; a failing performance test does not (only its recorded runtimes count).
- **Speed-of-light screen.** When the task passes `bytes_per_call` and `peak_bandwidth_gbs`, the
  evaluator scores any paired benchmark above `implausible_roofline_pct` (default 100) as a failure
  and flags speedups above `implausible_speedup` (default 2.0) for a hand check. Both are config keys.
- **Pin compile options outside the evolve block.** For MLX Metal kernels keep the
  `mx.fast.metal_kernel(...)` call, `compile_options={"math_mode": "safe"}` and input/output names
  in the wrapper; let the evolve block own the source string and launch constants only, and add a
  test that greps the block for `metal_kernel(` and `math_mode`.

## Debugging your tests

With the package installed, you can iterate locally before submitting anything.

Check that your tests pass against the reference. This validates the task definition itself:

```bash
pytest --ref -s task.py
```

Check that the kernel builds. This requires a working kernel already in the evolve block, and
runs the `build` function of your `TestBase` class to compile it to a shared object:

```bash
python task.py
```

Then run the tests against the compiled kernel:

```bash
pytest -s task.py
```

If the reference fails, fix the task before anything else: every speedup is measured against
it, so a broken reference makes every subsequent number meaningless.
