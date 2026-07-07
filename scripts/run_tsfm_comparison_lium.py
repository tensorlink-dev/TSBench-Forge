#!/usr/bin/env python3
"""Run the TSFM-vs-TSFM significance comparison on a Lium GPU.

The classical panel runs on CPU, but the foundation models need a GPU. This
driver rents a modest GPU on Lium, stages the repo + a slice of the scraped
parquet, installs one model *group*'s extras, runs ``tsfm_comparison.run_comparison``
on the GPU, and pulls ``results/results.json`` + ``figures/`` back.

Secrets discipline (see the lium-expertise skill): ``LIUM_API_KEY`` is read from
the LOCAL ``.env`` to authenticate the Lium client — the ``.env`` is **never**
uploaded to the pod (the GPU operator is untrusted). The rsync excludes it
explicitly, belt-and-braces with ``.gitignore``.

Model groups (dependency-incompatible sets can't share one env):
  A : chronos2 chronos-bolt timesfm25 toto2 tirex moirai2 flowstate tabpfn-ts
  B : sundial            (pins transformers==4.40.1 — its own env)

The challenge set is identical across groups because
``(SEED, MOTIF_LEN, N_CHALLENGES, DATA_DIR)`` is fixed, so per-group
``results.json`` files merge by concatenating their ``leaderboard`` rows.

Usage
-----
    python scripts/run_tsfm_comparison_lium.py --gpu L40 --group A
    python scripts/run_tsfm_comparison_lium.py --gpu RTX4090 --group B --max-price 0.6

Always tears the pod down (try/finally), even on failure.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Group → (pip install spec, roster short-names). Installs are best-effort; a
# model whose wheel fails is skipped by the driver's load probe, so a partial
# group still produces a valid (smaller) leaderboard.
GROUPS = {
    "A": {
        "pip": (
            "torch --index-url https://download.pytorch.org/whl/cu121 "
            "&& pip install 'chronos-forecasting>=2.0' "
            "'git+https://github.com/google-research/timesfm.git' "
            "toto-models 'tirex-ts[cuda]' uni2ts "
            "'granite-tsfm @ git+https://github.com/ibm-granite/granite-tsfm.git' "
            "tabpfn-time-series scipy pandas pyarrow matplotlib"
        ),
        "roster": [
            "chronos2", "chronos-bolt", "timesfm25", "toto2",
            "tirex", "moirai2", "flowstate", "tabpfn-ts",
        ],
    },
    "B": {
        "pip": (
            "torch --index-url https://download.pytorch.org/whl/cu121 "
            "&& pip install transformers==4.40.1 scipy pandas pyarrow matplotlib"
        ),
        "roster": ["sundial"],
    },
}


def _load_local_env() -> None:
    """Read LIUM_API_KEY from the local .env into os.environ (never uploaded)."""
    env = REPO / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


def _eligible_source_dirs(motif_len: int) -> list[str]:
    """Source ids whose concatenated parquet already clears the length bar.

    Only these are worth uploading — staging the full ``data/`` would be huge and
    mostly sub-threshold. Uses the same loader the eval uses, so the pod sees an
    identical eligible universe.
    """
    sys.path.insert(0, str(REPO / "src"))
    from scraped_source import ScrapedLiveSource

    src = ScrapedLiveSource(
        str(REPO / "src/sources/sources.yaml"),
        str(REPO / "src/sources/data"),
        min_series_length=motif_len,
    )
    return sorted({s["source_id"] for s in src._catalog()})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="L40", help="GPU type (e.g. L40, RTX4090, A100)")
    ap.add_argument("--group", default="A", choices=sorted(GROUPS))
    ap.add_argument("--max-price", type=float, default=1.5, help="$/hr ceiling")
    ap.add_argument("--n-challenges", type=int, default=256)
    ap.add_argument("--motif-len", type=int, default=304)
    ap.add_argument("--seed", default="tsfm-significance-v1")
    ap.add_argument("--template", default="pytorch", help="Lium template search term")
    ap.add_argument("--yes", action="store_true", help="skip the cost-estimate prompt")
    args = ap.parse_args()

    _load_local_env()
    if not os.environ.get("LIUM_API_KEY"):
        print("ERROR: LIUM_API_KEY not found in environment or .env", file=sys.stderr)
        return 2

    try:
        from lium.sdk import Lium
    except ImportError:
        print("ERROR: `pip install lium` first (the lium extra).", file=sys.stderr)
        return 2

    grp = GROUPS[args.group]
    roster = grp["roster"]
    eligible = _eligible_source_dirs(args.motif_len)
    print(f"group {args.group}: {roster}")
    print(f"eligible source series to stage: {len(eligible)}")

    lium = Lium()
    execs = [e for e in lium.ls(gpu_type=args.gpu) if e.price_per_hour <= args.max_price]
    if not execs:
        print(f"no {args.gpu} executors under ${args.max_price}/hr", file=sys.stderr)
        return 1
    executor = sorted(execs, key=lambda e: e.price_per_hour)[0]
    est = executor.price_per_hour * 0.5  # ~30 min budget
    print(f"picked {executor.id} @ ${executor.price_per_hour:.2f}/hr  (~${est:.2f} for 30 min)")
    if not args.yes and est > 5:
        if input("proceed? [y/N] ").strip().lower() != "y":
            return 0

    pod = lium.up(executor=executor.id, name=f"tsfm-cmp-{args.group.lower()}", template=args.template)
    try:
        ready = lium.wait_ready(pod, timeout=600)
        # --- stage code (NO .env, NO .git, NO heavy data/) ---
        lium.rsync(
            ready, str(REPO) + "/", "/root/repo/",
            exclude=[".git", ".env", "__pycache__", "*.pyc", "src/sources/data", "notebooks/results"],
        )
        # --- stage ONLY the eligible source parquet ---
        for sid in eligible:
            lium.rsync(ready, str(REPO / "src/sources/data" / sid) + "/",
                       f"/root/repo/src/sources/data/{sid}/")
        # --- install this group's deps ---
        print("installing deps (several minutes)…")
        lium.exec(ready, command=f"cd /root/repo && pip install -q {grp['pip']}")
        print(lium.exec(ready, command="nvidia-smi --query-gpu=name,memory.total --format=csv,noheader")["stdout"])
        # --- run the comparison on GPU ---
        roster_arg = ",".join(roster)
        run = (
            "cd /root/repo && PYTHONPATH=src python -c \""
            "import tsfm_comparison as tc; "
            f"tc.run_comparison('src/sources/data', catalog='src/sources/sources.yaml', "
            f"roster='{roster_arg}'.split(','), device='cuda', "
            f"motif_len={args.motif_len}, n_challenges={args.n_challenges}, "
            f"seed='{args.seed}', out_dir='results')\""
        )
        res = lium.exec(ready, command=run)
        print(res["stdout"][-4000:])
        if res.get("stderr"):
            print("STDERR:", res["stderr"][-2000:], file=sys.stderr)
        # --- pull results back ---
        out = REPO / "notebooks" / "results" / f"group_{args.group}"
        out.mkdir(parents=True, exist_ok=True)
        lium.scp(ready, "/root/repo/results/results.json", str(out / "results.json"))
        print(f"pulled results -> {out/'results.json'}")
    finally:
        lium.down(pod)
        print("pod torn down.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
