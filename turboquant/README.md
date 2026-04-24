# TurboQuant extension

Optional build that adds [0xSero/turboquant](https://github.com/0xSero/turboquant)
KV-cache compression on top of `ghcr.io/aeon-7/vllm-dflash:latest`.

See the main [README](../README.md#turboquant-optional) for when to use this and
the measured overhead (~3% across prompt classes and concurrency levels).

## Build

```bash
cd turboquant
docker build -t vllm-dflash-tq:latest .
```

## Run

```bash
docker run -d --name vllm-dflash-tq \
    --gpus all --network host --ipc host --ulimit memlock=-1:-1 \
    -v /models/DFlash-Qwen3.5-27B-Uncensored-NVFP4:/models/target:ro \
    -e MODEL_PATH=/models/target \
    -e DFLASH_DRAFTER=z-lab/Qwen3.5-27B-DFlash \
    -e DFLASH_NUM_SPEC_TOKENS=15 \
    -e MAX_MODEL_LEN=65536 \
    -e MAX_NUM_SEQS=16 \
    -e MAX_NUM_BATCHED_TOKENS=32768 \
    -e GPU_MEMORY_UTILIZATION=0.85 \
    -e ATTENTION_BACKEND=flash_attn \
    -e ENABLE_TURBOQUANT=1 \
    -e TQ_MODE=hybrid \
    -e TQ_KEY_BITS=4 \
    -e TQ_VALUE_BITS=3 \
    vllm-dflash-tq:latest
```

## How it works

- `turbokv_hook.pth` is dropped into `site-packages` and picked up by Python's
  `site.py` during interpreter init — **before** any `sitecustomize` that Ubuntu
  ships (which would otherwise shadow a custom one).
- The `.pth` just runs `import turbokv_bootstrap`.
- `turbokv_bootstrap.py` gates on `ENABLE_TURBOQUANT=1`, then registers a
  meta-path finder that waits for `vllm.v1.worker.gpu_worker` to be imported.
  As soon as it is, a one-time monkey-patch wraps `Worker.load_model` so that
  after weights load, `install_turboquant_hooks(worker.model_runner, ...)` fires.
- All flags (`TQ_MODE`, `TQ_KEY_BITS`, etc.) are read from env at that point.

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `ENABLE_TURBOQUANT` | `0` | Master switch. `0` makes the bootstrap a no-op. |
| `TQ_MODE` | `hybrid` | `off` / `capture_only` / `hybrid` / `full_tq` |
| `TQ_KEY_BITS` | `4` | Key quantization bits |
| `TQ_VALUE_BITS` | `3` | Value quantization bits |
| `TQ_VALUE_GROUP_SIZE` | `32` | Value group size |
| `TQ_RING_CAPACITY` | `128` | Exact-precision recent-token buffer |
| `TQ_INITIAL_LAYERS` | `4` | First N layers get `key_bits+1` |

## Compatibility note

The Dockerfile installs [`AEON-7/turboquant`](https://github.com/AEON-7/turboquant)
branch `fix/cuda-graph-safe-qjl-powers`, which is the fork hosting
[PR #12](https://github.com/0xSero/turboquant/pull/12) — a 20-line patch making
`_pack_qjl_signs` / `_unpack_qjl_signs` CUDA-graph-safe. Without that patch,
enabling TurboQuant forces `--enforce-eager`, costing 20–30% decode speed.

Once the PR merges upstream, the `Dockerfile` pin will be updated to
`git+https://github.com/0xSero/turboquant.git@main`.
