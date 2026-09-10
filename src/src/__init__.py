"""scRNA-seq ICI heterogeneous graph baseline."""

import os
import sys

# Default accelerator is physical GPU 1. Use CPU with `--device cpu` or HTG_DEVICE=cpu.
# This must run before torch is imported.
def _requested_device() -> str:
    args = sys.argv[1:]
    if args and args[0] == "predict":
        args = args[1:]
    for i, arg in enumerate(args):
        if arg == "--device" and i + 1 < len(args):
            return args[i + 1].strip().lower()
        if arg.startswith("--device="):
            return arg.split("=", 1)[1].strip().lower()
    return os.environ.get("HTG_DEVICE", "auto").strip().lower()


DEVICE = _requested_device()
if DEVICE not in {"cpu", "cpu:0"}:
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
