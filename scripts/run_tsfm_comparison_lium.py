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

# Per-group install RECIPES. Each group runs in its OWN venv, so groups can (and
# must) pin DIFFERENT torch versions — co-installing them broke torch/transformers.
# Each group: `torch` (index-pinned wheel, must succeed) + `core` (the eval's own
# deps, must succeed) + `models` (best-effort; the load-probe skips a failed one).
# Specs with shell-special chars (>=, [ ) are pre-quoted. Recipes verified 2026-07-09.
_CU121 = "--index-url https://download.pytorch.org/whl/cu121"
_CU126 = "--index-url https://download.pytorch.org/whl/cu126"
_EVAL = "numpy pandas pyarrow scipy pyyaml matplotlib accelerate"

GROUPS = {
    # proven-working cu121 groups
    "self":    {"roster": ["timesfm25", "tirex"], "torch": f"torch torchvision {_CU121}",
                "core": _EVAL,
                "models": ["git+https://github.com/google-research/timesfm.git", "'tirex-ts[cuda]'"]},
    "chronos": {"roster": ["chronos2", "chronos-bolt"], "torch": f"torch torchvision {_CU121}",
                "core": _EVAL, "models": ["'chronos-forecasting>=2.0'"]},
    "sundial": {"roster": ["sundial"], "torch": f"torch torchvision {_CU121}",
                "core": _EVAL, "models": ["transformers==4.40.1"]},
    # FlowState: granite-tsfm 0.3.6 needs torch>=2.10 (cu126) + transformers>=4.57.6 —
    # cu121's old torch dragged in an old transformers → 'PreTrainedModel' import fail.
    "flow":    {"roster": ["flowstate"], "torch": f"'torch>=2.10,<2.11' torchvision {_CU126}",
                "core": _EVAL,
                "models": ["granite-tsfm==0.3.6", "-U 'transformers>=4.57.6,<5' safetensors"]},
    # Moirai-2: uni2ts 2.0.0 has the moirai2 module; its unpinned jax[cpu] mismatches
    # jaxlib → circular import. Pin the jax pair; torch<2.5; numpy 1.26 (gluonts).
    "moirai":  {"roster": ["moirai2"], "torch": f"torch==2.4.1 torchvision==0.19.1 {_CU121}",
                "core": "'numpy~=1.26.0' pandas pyarrow scipy pyyaml matplotlib",
                "models": ["uni2ts==2.0.0", "'jax[cpu]==0.4.34' jaxlib==0.4.34"]},
    # Toto-2: no flash-attn/xformers; just torch>=2.4 (2.5.1). Earlier crash was a
    # plain-torch wheel mismatch; pin the cu121 wheel.
    "toto":    {"roster": ["toto2"], "torch": f"torch==2.5.1 torchvision==0.20.1 {_CU121}",
                "core": _EVAL, "models": ["toto-models"]},
    "tabpfn":  {"roster": ["tabpfn-ts"], "torch": f"torch torchvision {_CU121}",
                "core": _EVAL, "models": ["tabpfn-time-series"]},
}
ALL_GROUPS = ["self", "chronos", "moirai", "toto", "flow", "tabpfn", "sundial"]


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


def _lium(*args, capture=False, check=True, timeout=1800, **kw):
    # HARD timeout on every lium call: `lium up` opens an SSH session that can hang
    # indefinitely (observed: an 8-hour hang). Without this a stuck call blocks the
    # whole run forever. A timed-out call raises TimeoutExpired, which callers treat
    # as a failed step (e.g. _provision rotates to the next executor).
    cmd = [LIUM, *args]
    print("  $ lium", " ".join(args if len(" ".join(args)) < 200 else [args[0], "…"]))
    return subprocess.run(
        cmd, capture_output=capture, text=True, check=check, timeout=timeout,
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
    n_challenges=int(os.environ["NCH"]), seed=os.environ["SEED"],
    out_dir=os.environ.get("OUT_DIR", "results"),
)
print("LOADED:", [r["name"] for r in out["load_report"] if r["loaded"]])
print("NOTE:", out["note"])
"""


def _ps():
    return json.loads(_lium("ps", "--format", "json", capture=True, check=False).stdout or "[]")


def _provision(gname: str, execs: list[dict], args) -> str | None:
    """Provision a pod that actually registers in `ps`, retrying across executors.

    Some Lium nodes hand out ephemeral pods that vanish when the non-interactive
    SSH from `up` drops — they never appear in `ps`. So: `up`, then wait for a NEW
    huid to show up; if it doesn't within ~90s, that executor is no good — remove
    and try the next. Returns the resolved huid (reliable id for exec/rsync/scp).
    """
    name = f"tsfm-cmp-{gname}"
    for ex in execs[:10]:  # some nodes hand out ephemeral pods; try more before giving up
        before = {p.get("huid") for p in _ps()}
        try:
            # up opens an SSH session; cap it so a hang can't block the run.
            _lium("up", ex["id"], "--name", name, "--ttl", args.ttl, "-y", timeout=150)
        except subprocess.TimeoutExpired:
            print(f"up hung on {ex['huid']}; rotating executor", file=sys.stderr)
            _lium("rm", name, check=False)
            continue
        for _ in range(9):  # ~90s for the pod to register in ps
            new = [p for p in _ps() if p.get("huid") not in before]
            if new:
                huid = new[0]["huid"]
                print(f"pod {huid} registered on {ex['huid']} (${ex['price_per_hour']:.2f}/hr)")
                return huid
            time.sleep(10)
        print(f"executor {ex['huid']} gave an ephemeral/missing pod; trying next", file=sys.stderr)
        _lium("rm", name, check=False)
    return None


def _run_all_one_pod(groups: list[str], execs: list[dict], stage: Path, args,
                     tok: str, tabtok: str) -> None:
    """All groups on ONE pod, a fresh venv per group (sequential).

    Provisioning once dodges the flaky repeated `up` (ephemeral/hanging pods);
    a per-group venv isolates the conflicting model libs (torch/torchvision/
    transformers) that broke the single shared env. pip's wheel cache means
    torch downloads once and installs fast into each later venv. Every group
    scores the identical seeded challenge set → results merge.
    """
    target = _provision("all", execs, args)
    if not target:
        print("could not provision a pod; aborting", file=sys.stderr)
        return
    env_flags = ["-e", f"HF_TOKEN={tok}", "-e", f"HUGGING_FACE_HUB_TOKEN={tok}"] if tok else []
    if tabtok:
        env_flags += ["-e", f"TABPFN_TOKEN={tabtok}"]
    try:
        _lium("rsync", target, str(stage) + "/", "/root/repo/")
        # venv support on the minimized-Ubuntu base
        _lium("exec", target, "apt-get update -qq && apt-get install -y -qq python3-venv python3-pip",
              check=False, timeout=600)
        for gname in groups:
            if gname == "tabpfn" and not tabtok:
                print(f"skip '{gname}': no TABPFN_TOKEN"); continue
            grp = GROUPS[gname]
            v = f"/root/v_{gname}"
            pip = f"{v}/bin/pip install -q"
            print(f"\n--- group '{gname}': {grp['roster']} (venv {v}) ---")
            # torch (index-pinned) + core eval deps must succeed; per-group recipe.
            core = (f"python3 -m venv {v} && {v}/bin/pip install -q --upgrade pip && "
                    f"{pip} {grp['torch']} && {pip} {grp['core']}")
            try:
                _lium("exec", target, *env_flags, core, timeout=2400)  # must succeed
            except Exception as e:  # noqa: BLE001
                print(f"  core install failed for {gname}: {e}", file=sys.stderr); continue
            # model libs best-effort, in order (later specs override earlier pins).
            models = " ; ".join(f"{pip} {s} || true" for s in grp["models"])
            _lium("exec", target, *env_flags, models, check=False, timeout=2400)
            run_env = env_flags + [
                "-e", f"ROSTER={','.join(grp['roster'])}", "-e", f"MOTIF={args.motif_len}",
                "-e", f"NCH={args.n_challenges}", "-e", f"SEED={args.seed}",
                "-e", f"OUT_DIR=results_{gname}",
            ]
            res = _lium("exec", target, *run_env,
                        f"cd /root/repo && PYTHONPATH=src {v}/bin/python run_on_pod.py",
                        capture=True, check=False, timeout=1800)
            print((res.stdout or "")[-2500:])
            if res.returncode != 0:
                print(f"  run stderr: {(res.stderr or '')[-1500:]}", file=sys.stderr)
            out = REPO / "notebooks" / "results" / f"group_{gname}"
            out.mkdir(parents=True, exist_ok=True)
            _lium("scp", target, f"/root/repo/results_{gname}/results.json", str(out), "-d", check=False)
            print(f"  pulled -> {out}/results.json")
    finally:
        _lium("rm", target, check=False)
        print(f"pod {target} torn down.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="RTX4090")
    ap.add_argument("--group", default="self",
                    help="one group, 'all', or a comma-list (e.g. flow,moirai,toto)")
    ap.add_argument("--max-price", type=float, default=2.0)
    ap.add_argument("--n-challenges", type=int, default=256)
    ap.add_argument("--motif-len", type=int, default=304)
    ap.add_argument("--seed", default="tsfm-significance-v1")
    ap.add_argument("--ttl", default="2h", help="hard pod auto-teardown backstop")
    ap.add_argument("--yes", action="store_true")
    args = ap.parse_args()

    _load_local_env()
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN", "")
    tabtok = os.environ.get("TABPFN_TOKEN", "")
    if args.group == "all":
        groups = ALL_GROUPS
    else:
        groups = [g.strip() for g in args.group.split(",") if g.strip()]
    bad = [g for g in groups if g not in GROUPS]
    if bad:
        print(f"unknown group(s) {bad}; valid: {ALL_GROUPS}", file=sys.stderr)
        return 2
    # >1 group → single-pod venv-per-group path; exactly 1 → multi-executor path.

    eligible = _eligible_source_dirs(args.motif_len)
    print(f"groups: {groups}")
    print(f"staging {len(eligible)} eligible source series")

    js = json.loads(_lium("ls", "--gpu", args.gpu, "--format", "json", capture=True).stdout)
    execs = [e for e in js if e.get("price_per_hour", 1e9) <= args.max_price]
    if not execs:
        print(f"no {args.gpu} under ${args.max_price}/hr", file=sys.stderr)
        return 1
    ex = sorted(execs, key=lambda e: e["price_per_hour"])[0]
    print(f"cheapest {args.gpu}: {ex['huid']} ({ex['config']}) @ ${ex['price_per_hour']:.2f}/hr")
    if not args.yes and input(f"run {len(groups)} group(s)? [y/N] ").strip().lower() != "y":
        return 0

    stage = Path("/tmp/tsfm_stage")
    _build_staging(eligible, stage)

    avail = sorted(execs, key=lambda e: e["price_per_hour"])
    # Always ONE pod, a fresh venv per group (provision once → dodges the flaky
    # repeated `up`; venvs isolate each group's pinned torch/deps).
    _run_all_one_pod(groups, avail, stage, args, tok, tabtok)
    print("\nMerge with:  python scripts/merge_tsfm_results.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
