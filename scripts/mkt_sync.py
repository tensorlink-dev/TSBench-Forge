"""Mirror the MarketCast bar store (scripts/mkt_record.py) to Hippius, then prune.

Keys: ``marketcast/bars/<class>/<asset>/<YYYY-MM-DD>/<field>.npy`` and
``marketcast/bars/<class>/<asset>/funding.jsonl`` in the forge bucket
(``HIPPIUS_S3_BUCKET``), beside -- never inside -- the forge mirror's
``sources.yaml`` / ``data/`` tree.

A day file is a fixed-size float64[1440], so a size diff cannot see a change.
The recorder rewrites up to 48 h back, so every day in the last
``HOT_DAYS`` is re-sent each pass; older days are final and go up once.
``funding.jsonl`` is append-only and re-sent when its size changes.

Local days older than ``--keep-days`` are deleted only once every file of the
day is in the bucket at the same size. Restore with
``aws s3 sync s3://<bucket>/marketcast/bars <store>``.

usage: mkt_sync.py --store DIR [--keep-days 30] [--dry-run]
"""
from __future__ import annotations

import argparse
import datetime as dt
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "sources"))
import sync_storage as S  # noqa: E402

PREFIX = "marketcast/bars/"
HOT_DAYS = 3


def remote_sizes(s3, bucket: str) -> dict[str, int]:
    out, token = {}, None
    while True:
        kw = {"Bucket": bucket, "Prefix": PREFIX}
        if token:
            kw["ContinuationToken"] = token
        r = s3.list_objects_v2(**kw)
        out.update({o["Key"]: o["Size"] for o in r.get("Contents", [])})
        if not r.get("IsTruncated"):
            return out
        token = r["NextContinuationToken"]


def main() -> int:
    import os
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", type=Path, required=True)
    ap.add_argument("--bucket", default=os.environ.get("HIPPIUS_S3_BUCKET") or S.DEFAULT_BUCKET)
    ap.add_argument("--keep-days", type=int, default=30)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    s3 = S._client()
    remote = remote_sizes(s3, a.bucket)
    today = dt.datetime.now(dt.UTC).date()
    hot = {str(today - dt.timedelta(days=k)) for k in range(HOT_DAYS)}

    sent = failed = 0
    for p in sorted(a.store.rglob("*")):
        if not p.is_file() or p.name.startswith("."):
            continue
        rel = p.relative_to(a.store).as_posix()
        key = PREFIX + rel
        day = p.parent.name
        if key in remote and remote[key] == p.stat().st_size and day not in hot:
            continue
        if a.dry_run:
            sent += 1
            continue
        try:
            s3.upload_file(str(p), a.bucket, key)
            remote[key] = p.stat().st_size
            sent += 1
        except Exception as e:  # noqa: BLE001 — retried next pass
            failed += 1
            print(f"  upload failed {rel}: {type(e).__name__}", file=sys.stderr)

    cutoff = str(today - dt.timedelta(days=a.keep_days))
    pruned = 0
    for d in sorted(a.store.glob("*/*/????-??-??")):
        if d.name >= cutoff:
            continue
        files = [f for f in d.iterdir() if f.is_file()]
        if files and all(remote.get(PREFIX + f.relative_to(a.store).as_posix()) == f.stat().st_size
                         for f in files):
            if not a.dry_run:
                shutil.rmtree(d)
            pruned += 1
    print(f"{dt.datetime.now(dt.UTC):%FT%TZ} {'would send' if a.dry_run else 'sent'} {sent} file(s), "
          f"{failed} failed, pruned {pruned} day dir(s) older than {cutoff} -> {a.bucket}/{PREFIX}",
          flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
