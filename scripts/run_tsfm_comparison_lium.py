#!/usr/bin/env python3
"""Run the TSFM-vs-TSFM significance comparison on a Lium GPU (CLI-driven).

The classical panel runs on CPU, but the foundation models need a GPU. This
driver rents a modest GPU on Lium, stages a **clean copy** of the repo + only
the eligible parquet, installs one model *group*'s extras, runs
``tsfm_comparison.run_comparison`` on the GPU, and pulls ``results.json`` back.

Why CLI, not SDK: the ``lium`` SDK lives in its own tool venv without this repo's
deps, so we shell out to the ``lium`` binary and run everything else under the
repo's own interpreter.

Secrets discipline (lium-expertise skill):
* ``lium rsync`` has no ``--exclude``, so we rsync from a **staging dir** that
  never contains ``.env`` / ``.git`` / the heavy ``data/`` — the operator can't
  read what was never uploaded.
* ``HF_TOKEN`` is read from the LOCAL ``.env`` and passed via ``lium exec -e`` at
  exec time only (for gated weights). It is never written to the staging dir.
* ``lium up --ttl`` sets a hard auto-teardown as a backstop, and the script also
  tears the pod down in ``finally``.

Model groups (dependency-incompatible sets can't share one env):
  A : chronos2 chronos-bolt timesfm25 toto2 tirex moirai2 flowstate tabpfn-ts
  B : sundial            (pins transformers==4.40.1 — its own env)

Usage
-----
    python scripts/run_tsfm_comparison_lium.py --gpu RTX4090 --group A --yes
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LIUM = os.path.expanduser("~/.local/bin/lium")

GROUPS = {
    "A": {
        "pip": (
            "pip install -q torch --index-url https://download.pytorch.org/whl/cu121 && "
            "pip install -q 'chronos-forecasting>=2.0' "
            "'git+https://github.com/google-research/timesfm.git' "
            "toto-models 'tirex-ts[cuda]' uni2ts "
            "'granite-tsfm @ git+https://github.com/ibm-granite/granite-tsfm.git' "
            "tabpfn-time-series scipy pandas pyarrow matplotlib pyyaml"
        ),
        "roster": [
            "chronos2", "chronos-bolt", "timesfm25", "toto2",
            "tirex", "moirai2", "flowstate", "tabpfn-ts",
        ],
    },
    "B": {
        "pip": (
            "pip install -q torch --index-url https://download.pytorch.org/whl/cu121 && "
            "pip install -q transformers==4.40.1 scipy pandas pyarrow matplotlib pyyaml"
        ),
        "roster": ["sundial"],
    },
}


def _load_local_env() -> None:
    env = REPO / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip("'").strip('"'))  # unquote


def _lium(*args, capture=False, check=True, **kw):
    cmd = [LIUM, *args]
    print("  $ lium", " ".join(args if len(" ".join(args)) < 200 else [args[0], "…"]))
    return subprocess.run(
        cmd, capture_output=capture, text=True, check=check,
        env={**os.environ, "PATH": os.path.expanduser("~/.local/bin") + ":" + os.environ["PATH"]},
        **kw,
    )


def _eligible_source_dirs(motif_len: int) -> list[str]:
    sys.path.insert(0, str(REPO / "src"))
    from scraped_source import ScrapedLiveSource

    src = ScrapedLiveSource(
        str(REPO / "src/sources/sources.yaml"),
        str(REPO / "src/sources/data"),
        min_series_length=motif_len,
    )
    return sorted({s["source_id"] for s in src._catalog()})


def _build_staging(eligible: list[str], stage: Path) -> None:
    """Clean tree the pod will see: src/ (minus data), pyproject, eligible parquet.

    Never copies .env / .git — they are simply not put in the staging dir.
    """
    if stage.exists():
        shutil.rmtree(stage)
    (stage / "src").mkdir(parents=True)
    # src/ without the heavy data dir and caches
    shutil.copytree(
        REPO / "src", stage / "src", dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("data", "__pycache__", "*.pyc"),
    )
    shutil.copy(REPO / "pyproject.toml", stage / "pyproject.toml")
    # only the eligible source parquet
    for sid in eligible:
        s = REPO / "src/sources/data" / sid
        if s.is_dir():
            shutil.copytree(s, stage / "src/sources/data" / sid)
    # tiny on-pod entrypoint (avoids nested-quote hell in `lium exec`)
    (stage / "run_on_pod.py").write_text(_POD_ENTRY)


_POD_ENTRY = """import os, sys
sys.path.insert(0, "src")
import tsfm_comparison as tc
roster = os.environ["ROSTER"].split(",")
out = tc.run_comparison(
    "src/sources/data", catalog="src/sources/sources.yaml", roster=roster,
    device="cuda", motif_len=int(os.environ["MOTIF"]),
    n_challenges=int(os.environ["NCH"]), seed=os.environ["SEED"], out_dir="results",
)
print("LOADED:", [r["name"] for r in out["load_report"] if r["loaded"]])
print("NOTE:", out["note"])
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="RTX4090")
    ap.add_argument("--group", default="A", choices=sorted(GROUPS))
    ap.add_argument("--max-price", type=float, default=2.0)
    ap.add_argument("--n-challenges", type=int, default=256)
    ap.add_argument("--motif-len", type=int, default=304)
    ap.add_argument("--seed", default="tsfm-significance-v1")
    ap.add_argument("--ttl", default="2h", help="hard pod auto-teardown backstop")
    ap.add_argument("--yes", action="store_true")
    args = ap.parse_args()

    _load_local_env()
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN", "")
    grp = GROUPS[args.group]
    roster = grp["roster"]

    eligible = _eligible_source_dirs(args.motif_len)
    print(f"group {args.group}: {roster}")
    print(f"staging {len(eligible)} eligible source series")

    # pick cheapest executor of the requested GPU under the price ceiling
    js = json.loads(_lium("ls", "--gpu", args.gpu, "--format", "json", capture=True).stdout)
    execs = [e for e in js if e.get("price_per_hour", 1e9) <= args.max_price]
    if not execs:
        print(f"no {args.gpu} under ${args.max_price}/hr", file=sys.stderr)
        return 1
    ex = sorted(execs, key=lambda e: e["price_per_hour"])[0]
    print(f"picked {ex['huid']} ({ex['config']}) @ ${ex['price_per_hour']:.2f}/hr in {ex.get('country')}")
    if not args.yes and input("proceed? [y/N] ").strip().lower() != "y":
        return 0

    stage = Path("/tmp/tsfm_stage")
    _build_staging(eligible, stage)

    name = f"tsfm-cmp-{args.group.lower()}"
    _lium("up", ex["id"], "--name", name, "--ttl", args.ttl, "-y")  # -y: skip confirm; ttl = safety backstop
    try:
        # wait until the pod reports ready
        for _ in range(60):
            ps = json.loads(_lium("ps", "--format", "json", capture=True, check=False).stdout or "[]")
            me = next((p for p in ps if p.get("name") == name or p.get("huid") == name), None)
            status = (me or {}).get("status", "")
            if me and str(status).lower() in ("running", "ready", "active"):
                break
            time.sleep(10)
        print(f"pod {name} status: {status}")
        _lium("rsync", name, str(stage) + "/", "/root/repo/")
        env_flags = ["-e", f"HF_TOKEN={tok}", "-e", f"HUGGING_FACE_HUB_TOKEN={tok}"] if tok else []
        print("installing deps on GPU (several minutes)…")
        _lium("exec", name, *env_flags, f"cd /root/repo && {grp['pip']}")
        print(_lium("exec", name, "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader",
                    capture=True).stdout)
        run_env = env_flags + [
            "-e", f"ROSTER={','.join(roster)}", "-e", f"MOTIF={args.motif_len}",
            "-e", f"NCH={args.n_challenges}", "-e", f"SEED={args.seed}",
        ]
        res = _lium("exec", name, *run_env, "cd /root/repo && PYTHONPATH=src python run_on_pod.py",
                    capture=True, check=False)
        print(res.stdout[-6000:])
        if res.returncode != 0:
            print("RUN STDERR:", (res.stderr or "")[-3000:], file=sys.stderr)
        out = REPO / "notebooks" / "results" / f"group_{args.group}"
        out.mkdir(parents=True, exist_ok=True)
        _lium("scp", name, "/root/repo/results/results.json", str(out), "-d", check=False)
        print(f"pulled results -> {out}/results.json")
    finally:
        _lium("rm", name, check=False)
        print("pod torn down.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
