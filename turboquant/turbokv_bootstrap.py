# TurboQuant startup bootstrap.
#
# Loaded via a .pth file (turbokv_hook.pth) which Python processes during
# site initialization, before any sitecustomize can shadow it.  The .pth
# mechanism just does `import turbokv_bootstrap` — everything interesting
# happens here.

import os
import sys

if os.environ.get("ENABLE_TURBOQUANT", "0") != "1":
    # Noop when disabled — container behaves exactly like stock aeon-7.
    pass
else:
    try:
        import importlib
        import importlib.abc

        _TARGET = "vllm.v1.worker.gpu_worker"
        _PATCHED = {"done": False}

        def _install_on_worker(worker):
            try:
                from turboquant.integration.vllm import install_hooks
            except Exception as e:
                print(f"[turbokv-hook] turboquant not importable: {e}",
                      file=sys.stderr, flush=True)
                return

            mode = os.environ.get("TQ_MODE", "hybrid")
            key_bits = int(os.environ.get("TQ_KEY_BITS", "4"))
            value_bits = int(os.environ.get("TQ_VALUE_BITS", "3"))
            value_group_size = int(os.environ.get("TQ_VALUE_GROUP_SIZE", "32"))
            ring_capacity = int(os.environ.get("TQ_RING_CAPACITY", "128"))
            initial_layers_count = int(os.environ.get("TQ_INITIAL_LAYERS", "4"))
            no_alloc = os.environ.get("TQ_NO_ALLOC", "0") == "1"

            print(f"[turbokv-hook] install_hooks(mode={mode}, "
                  f"key_bits={key_bits}, value_bits={value_bits}, "
                  f"value_group_size={value_group_size}, "
                  f"ring_capacity={ring_capacity}, "
                  f"initial_layers_count={initial_layers_count}, "
                  f"no_alloc={no_alloc})",
                  file=sys.stderr, flush=True)
            try:
                states = install_hooks(
                    worker.model_runner,
                    key_bits=key_bits,
                    value_bits=value_bits,
                    value_group_size=value_group_size,
                    ring_capacity=ring_capacity,
                    initial_layers_count=initial_layers_count,
                    mode=mode,
                    no_alloc=no_alloc,
                )
                print(f"[turbokv-hook] installed on {len(states)} attention layers",
                      file=sys.stderr, flush=True)
            except Exception as e:
                import traceback
                print(f"[turbokv-hook] install_hooks FAILED: {e}",
                      file=sys.stderr, flush=True)
                traceback.print_exc(file=sys.stderr)

        def _apply_patch():
            if _PATCHED["done"]:
                return
            _PATCHED["done"] = True
            try:
                mod = importlib.import_module(_TARGET)
            except Exception as e:
                print(f"[turbokv-hook] could not import {_TARGET}: {e}",
                      file=sys.stderr, flush=True)
                return
            Worker = getattr(mod, "Worker", None)
            if Worker is None:
                print(f"[turbokv-hook] {_TARGET}.Worker not found",
                      file=sys.stderr, flush=True)
                return
            original_load_model = Worker.load_model
            if getattr(original_load_model, "_turbokv_patched", False):
                return

            def patched_load_model(self, *args, **kwargs):
                result = original_load_model(self, *args, **kwargs)
                _install_on_worker(self)
                return result

            patched_load_model._turbokv_patched = True
            Worker.load_model = patched_load_model
            print(f"[turbokv-hook] patched {_TARGET}.Worker.load_model",
                  file=sys.stderr, flush=True)

        # Schedule patch via a meta-path finder that triggers when gpu_worker
        # is imported.  We can't patch immediately because vllm isn't loaded
        # when .pth files run.
        class _Finder(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == _TARGET and not _PATCHED["done"]:
                    import threading
                    threading.Timer(0.0, _apply_patch).start()
                return None

        sys.meta_path.insert(0, _Finder())
        print(f"[turbokv-hook] bootstrap installed via .pth; patching {_TARGET} "
              "on first import", file=sys.stderr, flush=True)
    except Exception as e:
        import traceback
        print(f"[turbokv-hook] bootstrap FAILED: {e}", file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
