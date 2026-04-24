# syntax=docker/dockerfile:1.6
# ============================================================================
#  vllm-dflash + TurboQuant KV-cache compression (optional extension)
#
#  Layers 0xSero/turboquant on top of ghcr.io/aeon-7/vllm-dflash:latest via a
#  clean Python plugin + site-packages .pth bootstrap.  No vLLM source patches.
#
#  Behaviour gate:
#    ENABLE_TURBOQUANT=0  (default) → container behaves exactly like stock dflash
#    ENABLE_TURBOQUANT=1            → TurboQuant wires in post-worker-load
#
#  Build:   docker build -t vllm-dflash-tq:latest .
#  Run:     see the README's "Enabling" section.
# ============================================================================
FROM ghcr.io/aeon-7/vllm-dflash:latest

LABEL org.opencontainers.image.title="vllm-dflash + TurboQuant"
LABEL org.opencontainers.image.description="aeon-7/vllm-dflash extended with 0xSero/turboquant KV-cache compression"
LABEL org.opencontainers.image.source="https://github.com/AEON-7/vllm-dflash"

# --- 1. Install 0xSero/turboquant ---
# Pin to the fork + branch that has the CUDA-graph-safe quantizer fix
# (https://github.com/0xSero/turboquant/pull/12).  Once that PR merges, switch
# to the upstream main branch.
RUN pip install scipy && \
    pip install --no-deps \
        git+https://github.com/AEON-7/turboquant.git@fix/cuda-graph-safe-qjl-powers && \
    python3 -c "import turboquant; from turboquant.integration.vllm import install_hooks; print('turboquant OK')"

# --- 2. Bootstrap: .pth file that Python autoloads during site init ---
COPY turbokv_bootstrap.py /usr/local/lib/python3.12/dist-packages/turbokv_bootstrap.py
COPY turbokv_hook.pth     /usr/local/lib/python3.12/dist-packages/turbokv_hook.pth

# --- 3. Clear stale bytecode so the hook runs on first interpreter start ---
RUN find /usr/local/lib/python3.12/dist-packages -name "__pycache__" \
         -path "*/vllm/*" -exec rm -rf {} + 2>/dev/null || true

# The aeon-7 base entrypoint is preserved.  ENABLE_TURBOQUANT=0 (default) → no-op.
