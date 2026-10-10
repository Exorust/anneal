# Anneal quickstart (Apple Silicon)

This walks through the shipped Metal example end to end on a Mac. It makes no LLM API calls, so it
needs no API key. The point is to prove your GPU and test setup work before you spend anything on
generation.

```{note}
This fork targets Apple GPUs first. The upstream SYCL example in `tasks/example_custom/` still
exists and needs an Intel GPU plus the oneAPI toolchain; see the README section "Other platforms".
```

## What you need

- macOS on an M-series chip
- Python 3.12 (hydra does not start on 3.14) and PyTorch 2.7 or newer. The default PyPI torch wheel
  includes Metal support; check with `python -c "import torch; print(torch.backends.mps.is_available())"`.
- A clone of the repository, for the example task and configs:

```bash
git clone https://github.com/Exorust/anneal.git
cd anneal
uv venv --python 3.12 .venv
uv pip install --python .venv -e . torch hydra-core pydra_config openai tqdm sqlalchemy \
    numpy python-dotenv pandas scipy celery clang boto3 requests pytest-dependency ninja
source .venv/bin/activate
```

No compiler or Xcode is needed: Metal source is compiled at runtime by `torch.mps.compile_shader`.

## 1. Look at the task

The example lives in `tasks/example_metal/`, a Metal matrix multiplication:

```
tasks/example_metal/
├── config.yaml          # task_name: matmul, gpu_arch: apple_m5, language: metal, has_build_step: false
├── task.py              # reference, fixtures, tests
├── matmul_kernel.py     # the Metal source string + launch helper, with the evolve block
└── conftest.py          # wires in the KernelFoundry fixtures (do not edit)
```

`task.py` defines a `TestMatrixMultiplication` class deriving from `TestBase` with:

- a **reference**, marked `[REFERENCE_START]` / `[REFERENCE_END]` (here `torch.matmul`)
- two **correctness tests**, against the reference and against an identity matrix
- one **benchmark**, marked `@pytest.mark.performance` and parametrized over sizes 512, 1024 and 2048

`matmul_kernel.py` holds the `[EVOLVE_START]` / `[EVOLVE_END]` block: the shader string, the
compiled library and a `launch(A, B, C, M, K, N)` helper that sets `threads` and `group_size`. The
torch wrapper below the block is the stable interface the tests call.

## 2. Validate the reference

```bash
cd tasks/example_metal
pytest --ref -s task.py
```

The correctness tests should pass and the benchmark should report runtimes for each size. On an M5
the reference takes about 0.7 ms at 1024x1024. If this fails, the problem is your environment or the
task, not the kernel.

## 3. Test and benchmark the kernel

There is no build step. Run the tests against the Metal kernel directly:

```bash
pytest -s task.py
```

The naive kernel takes about 6 ms at 1024x1024 on an M5, roughly 9x slower than the reference. That
ratio is the headroom the optimizer will work on.

## 4. Run it through the pipeline without generating

```bash
cd ../..
python -m kernelfoundry.algorithm run task=tasks/example_metal task_origin=custom \
    job_name=my_first_run gpu_arch=apple_m5 language=metal validate=true max_iters=0
```

`validate=true max_iters=0` means "evaluate what is there, generate nothing". Results land in
`runs/`, viewable in the UI:

```bash
kernelfoundry-gui       # http://localhost:8885
```

## Next steps

- **Optimize this kernel for real.** Same command without the validate flags, plus an API key. Pick
  the provider and model in `kernelfoundry/configs/inference/server.yaml` (`server_type: anthropic`
  or `openai`):

  ```bash
  export ANTHROPIC_API_KEY=...
  python -m kernelfoundry.algorithm run task=tasks/example_metal task_origin=custom \
      job_name=matmul_opt gpu_arch=apple_m5 language=metal
  ```

- **Package your own kernel**: [Anatomy of a task package](task-package.md). For Metal, copy
  `tasks/example_metal/` and keep the kernel as a Python file with the shader in a string.
- **Write tests for it**: [Writing tests](writing-tests.md)
- **Tune the run**: [Config parameters](config-parameters.md) and
  [Optimization strategies](optimization-strategies.md)
