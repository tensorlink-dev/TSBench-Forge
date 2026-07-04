"""Mirror the scraped parquet archive to an S3-compatible bucket (Hippius).

The dated parquet under ``src/sources/data`` is the benchmark's accumulating
asset — freshness history, unseen-weight provenance, and future contiguity for
the fast bins — and without this step it exists as a single copy on the scrape
host. Run after a scrape (or from the same cron):

    set -a; source .env; set +a
    python src/sources/sync_storage.py            # sync everything new/changed
    python src/sources/sync_storage.py --dry-run  # show what would upload

Environment (``.env``):
    HIPPIUS_S3_ACCESS_KEY   required
    HIPPIUS_S3_SECRET_KEY   required
    HIPPIUS_S3_ENDPOINT     default https://s3.hippius.com
    HIPPIUS_S3_BUCKET       default tsbench-forge-sources

Sync rule: upload when the object is missing remotely or the size differs —
today's parquet grows as re-scrapes append, so a size change means new rows.
Objects are never deleted; the bucket is an append-only mirror.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
DEFAULT_ENDPOINT = "https://s3.hippius.com"
DEFAULT_BUCKET = "tsbench-forge-sources"


def _client():
    import boto3

    access = os.environ.get("HIPPIUS_S3_ACCESS_KEY")
    secret = os.environ.get("HIPPIUS_S3_SECRET_KEY")
    if not access or not secret:
        sys.exit("HIPPIUS_S3_ACCESS_KEY / HIPPIUS_S3_SECRET_KEY not set — "
                 "load .env first (set -a; source .env; set +a)")
    return boto3.client(
        "s3",
        endpoint_url=os.environ.get("HIPPIUS_S3_ENDPOINT", DEFAULT_ENDPOINT),
        region_name=os.environ.get("HIPPIUS_S3_REGION", "decentralized"),
        aws_access_key_id=access,
        aws_secret_access_key=secret,
    )


def _remote_sizes(s3, bucket: str) -> dict[str, int]:
    sizes: dict[str, int] = {}
    token = None
    while True:
        kw = {"Bucket": bucket, "ContinuationToken": token} if token else {"Bucket": bucket}
        resp = s3.list_objects_v2(**kw)
        for obj in resp.get("Contents", []):
            sizes[obj["Key"]] = int(obj["Size"])
        if not resp.get("IsTruncated"):
            return sizes
        token = resp.get("NextContinuationToken")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    ap.add_argument("--bucket", default=os.environ.get("HIPPIUS_S3_BUCKET", DEFAULT_BUCKET))
    ap.add_argument("--dry-run", action="store_true", help="list what would upload, upload nothing")
    args = ap.parse_args()

    data = Path(args.data_dir)
    local = sorted(p for p in data.rglob("*.parquet"))
    if not local:
        sys.exit(f"no parquet under {data}")

    s3 = _client()
    try:
        s3.head_bucket(Bucket=args.bucket)
    except Exception:
        if args.dry_run:
            print(f"bucket {args.bucket!r} missing (would create)")
        else:
            try:
                s3.create_bucket(Bucket=args.bucket)
                print(f"created bucket {args.bucket!r}")
            except Exception as e:
                sys.exit(
                    f"bucket {args.bucket!r} does not exist and this token cannot "
                    f"create buckets ({type(e).__name__}). Create it in the Hippius "
                    f"console (S3 Storage -> buckets) or use a Master Token, then re-run."
                )

    remote = {} if args.dry_run else _remote_sizes(s3, args.bucket)
    if args.dry_run:
        try:
            remote = _remote_sizes(s3, args.bucket)
        except Exception:
            remote = {}

    uploaded = skipped = failed = 0
    up_bytes = 0
    for p in local:
        key = str(p.relative_to(data))
        size = p.stat().st_size
        if remote.get(key) == size:
            skipped += 1
            continue
        if args.dry_run:
            print(f"would upload {key} ({size:,} B)")
            uploaded += 1
            up_bytes += size
            continue
        try:
            s3.upload_file(str(p), args.bucket, key)
            uploaded += 1
            up_bytes += size
        except Exception as e:  # noqa: BLE001 — keep syncing the rest
            print(f"FAILED {key}: {e}", file=sys.stderr)
            failed += 1

    verb = "would upload" if args.dry_run else "uploaded"
    print(f"{verb} {uploaded} files ({up_bytes/1e6:.1f} MB), "
          f"{skipped} unchanged, {failed} failed -> "
          f"{os.environ.get('HIPPIUS_S3_ENDPOINT', DEFAULT_ENDPOINT)}/{args.bucket}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
