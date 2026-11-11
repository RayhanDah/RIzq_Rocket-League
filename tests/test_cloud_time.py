"""Temps de session Kaggle : arrêt calé sur la limite, autoréglage réutilisé d'une session à l'autre."""
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import autotune  # noqa: E402


def _tune(tmp_path, **rep):
    out = tmp_path / "autotune.yaml"
    out.write_text("ppo: {device: cpu}\n", encoding="utf-8")
    base = {"date": "2026-11-07T13:44:40", "ncpu": 224, "gpus": 0, "batch": 400_000, "learner": "cpu",
            "best": "local 222x8", "expected_sps": 174_917, "rule": autotune.PICK_RULE}
    base.update(rep)
    out.with_suffix(".json").write_text(json.dumps(base), encoding="utf-8")
    return out


def test_autotune_reused_on_same_machine_and_batch(tmp_path):
    out = _tune(tmp_path)
    assert autotune.reusable(out, 224, 0, 400_000, "xla,cpu")
    assert autotune.reusable(out, 224, 0, 400_000, None)


@pytest.mark.parametrize("ncpu,gpus,batch", [(96, 0, 400_000), (224, 2, 400_000), (224, 0, 200_000)])
def test_autotune_measured_again_when_machine_or_batch_differs(tmp_path, ncpu, gpus, batch):
    assert autotune.reusable(_tune(tmp_path), ncpu, gpus, batch, None) is None


def test_autotune_measured_again_without_report_or_learner(tmp_path, monkeypatch):
    out = _tune(tmp_path)
    assert autotune.reusable(out, 224, 0, 400_000, "cuda") is None
    xla = _tune(tmp_path, learner="xla")
    monkeypatch.setattr(autotune.accel, "xla_available", lambda: False)
    assert autotune.reusable(xla, 224, 0, 400_000, None) is None
    out.with_suffix(".json").unlink()
    assert autotune.reusable(out, 224, 0, 400_000, None) is None


def test_notebook_passes_session_start_before_anything_else():
    nb = json.loads((ROOT / "cloud" / "kaggle_train.ipynb").read_text(encoding="utf-8"))
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert "".join(code[0]["source"]).startswith("import time; KERNEL_T0 = time.time()")
    assert "KERNEL_T0=str(int(KERNEL_T0))" in "".join(code[2]["source"])


def _timer_block():
    """Bloc du script qui calcule l'heure d'arrêt (de « if HOURS = auto » au calcul de WAIT)."""
    src = (ROOT / "cloud" / "run_cloud.sh").read_text(encoding="utf-8")
    head = re.search(r'if \[ "\$HOURS" = "auto" \]; then.*?\nfi\n', src, re.S).group(0)
    wait = re.search(r'if \[ -n "\$STOP_AT" \]; then\n.*?\nelse .*?; fi\n', src, re.S).group(0)
    return head + wait + 'echo "WAIT=$WAIT"\n'


def _bash():
    """bash POSIX : Git Bash sous Windows (le bash.exe de WSL réécrit les guillemets), sinon celui du PATH."""
    if sys.platform == "win32":
        git = Path(r"C:\Program Files\Git\bin\bash.exe")
        return str(git) if git.exists() else None
    return shutil.which("bash")


@pytest.mark.skipif(_bash() is None, reason="bash absent")
@pytest.mark.parametrize("platform,hours,elapsed,expected", [
    ("kaggle_tpu", "auto", 600, 9 * 3600 - 20 * 60 - 600),      # 10 min d'installation : 8 h 30 d'entraînement
    ("kaggle", "auto", 600, 12 * 3600 - 20 * 60 - 600),
    ("kaggle_tpu", "auto", 9 * 3600, 60),                        # déjà trop tard : arrêt au bout d'une minute
    ("kaggle_tpu", "6", 600, 6 * 3600),                          # nombre d'heures : durée d'entraînement, comme avant
])
def test_clean_stop_counts_from_session_start(platform, hours, elapsed, expected):
    script = ("set -euo pipefail\nNOW=$(date +%s)\n"
              f'KERNEL_T0=$(( NOW - {elapsed} )); MARGIN_MIN=20; PLATFORM={platform}; HOURS={hours}\n'
              'PROFILES=p; RUN_NAME=r\n' + _timer_block())
    out = subprocess.run([_bash(), "-s"], input=script, capture_output=True, text=True, encoding="utf-8")
    assert out.returncode == 0, out.stderr
    wait = int(re.search(r"WAIT=(\d+)", out.stdout).group(1))
    assert abs(wait - expected) <= 2
