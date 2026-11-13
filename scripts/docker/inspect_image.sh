#!/bin/bash
set -euo pipefail

IMAGE=${HFDTREE_IMAGE:-wan21-t2vsafetybench:1.1}
docker run --rm "$IMAGE" bash -lc '
python --version
python - <<"PY"
mods = ["torch", "transformers", "peft", "accelerate", "safetensors"]
for name in mods:
    try:
        module = __import__(name)
        print(name, getattr(module, "__version__", "ok"))
    except Exception as exc:
        print(name, "MISSING", type(exc).__name__, str(exc)[:120])
PY
'
