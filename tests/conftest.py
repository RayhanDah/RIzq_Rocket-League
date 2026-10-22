import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

if os.environ.get("PJRT_DEVICE"):
    # tests XLA (Linux, torch_xla) : torch_xla doit être chargé avant le 1er backward du processus (cf. accel.py)
    try:
        import rlagent.accel  # noqa: F401
    except Exception:  # noqa: BLE001
        pass
