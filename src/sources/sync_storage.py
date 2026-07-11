"""Mirror the scraped parquet archive to/from an S3-compatible bucket (Hippius).

The dated parquet under ``src/sources/data`` is the benchmark's accumulating
asset — freshness history, unseen-weight provenance, and future contiguity for
the fast bins — and without this step it exists as a single copy on the scrape
host. Run after a scrape (or from the same cron):

    set -a; source .env; set +a
    python src/sources/sync_storage.py                    # upload new/changed
    python src/sources/sync_storage.py --dry-run          # show what would upload
    python src/sources/sync_storage.py --download --today # pull today's parquet back

Environment (``.env``):
    HIPPIUS_S3_ACCESS_KEY   required
    HIPPIUS_S3_SECRET_KEY   required
    HIPPIUS_S3_ENDPOINT     default https://s3.hippius.com
    HIPPIUS_S3_BUCKET       default tsbench-forge-sources

Sync rule: transfer when the object is missing on the destination or the size
differs — today's parquet grows as re-scrapes append, so a size change means
new rows. Objects are never deleted; the bucket is an append-only mirror.

``--download`` runs the mirror in reverse (remote -> local), pulling objects
whose local copy is missing or a different size. It exists for ephemeral
runners (GitHub Actions): pull the current day's parquet down before scraping
so the scraper appends to it, rather than starting empty and later overwriting
the day's remote object. ``--today`` restricts the pull to
``*/<UTC-today>.parquet`` — the only files a fresh scrape will touch — so the
transfer stays small.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
DEFAULT_ENDPOINT = "https://s3.hippius.com"
DEFAULT_BUCKET = "tsbench-forge-sources"


def _endpoint() -> str:
    return os.environ.get("HIPPIUS_S3_ENDPOINT", DEFAULT_ENDPOINT)


def _client():
    import boto3

    access = os.environ.get("HIPPIUS_S3_ACCESS_KEY")
    secret = os.environ.get("HIPPIUS_S3_SECRET_KEY")
    if not access or not secret:
        sys.exit("HIPPIUS_S3_ACCESS_KEY / HIPPIUS_S3_SECRET_KEY not set — "
                 "load .env first (set -a; source .env; set +a)")
    return boto3.client(
        "s3",
        endpoint_url=_endpoint(),
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


def _upload(s3, bucket: str, data: Path, dry_run: bool) -> int:
    local = sorted(p for p in data.rglob("*.parquet"))
    if not local:
        sys.exit(f"no parquet under {data}")

    try:
        s3.head_bucket(Bucket=bucket)
    except Exception:
        if dry_run:
            print(f"bucket {bucket!r} missing (would create)")
        else:
            try:
                s3.create_bucket(Bucket=bucket)
                print(f"created bucket {bucket!r}")
            except Exception as e:
                sys.exit(
                    f"bucket {bucket!r} does not exist and this token cannot "
                    f"create buckets ({type(e).__name__}). Create it in the Hippius "
                    f"console (S3 Storage -> buckets) or use a Master Token, then re-run."
                )

    try:
        remote = _remote_sizes(s3, bucket)
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
        if dry_run:
            print(f"would upload {key} ({size:,} B)")
            uploaded += 1
            up_bytes += size
            continue
        try:
            s3.upload_file(str(p), bucket, key)
            uploaded += 1
            up_bytes += size
        except Exception as e:  # noqa: BLE001 — keep syncing the rest
            print(f"FAILED {key}: {e}", file=sys.stderr)
            failed += 1

    verb = "would upload" if dry_run else "uploaded"
    print(f"{verb} {uploaded} files ({up_bytes/1e6:.1f} MB), "
          f"{skipped} unchanged, {failed} failed -> {_endpoint()}/{bucket}")
    return 1 if failed else 0


def _download(s3, bucket: str, data: Path, dry_run: bool, only_date: str | None) -> int:
    try:
        s3.head_bucket(Bucket=bucket)
    except Exception:
        print(f"bucket {bucket!r} not present remotely — nothing to download")
        return 0
    try:
        remote = _remote_sizes(s3, bucket)
    except Exception as e:  # noqa: BLE001
        print(f"could not list {bucket!r}: {e} — nothing to download", file=sys.stderr)
        return 0

    suffix = f"/{only_date}.parquet" if only_date else None
    downloaded = skipped = failed = 0
    dl_bytes = 0
    for key, size in sorted(remote.items()):
        if not key.endswith(".parquet"):
            continue
        if suffix and not (key.endswith(suffix) or key == f"{only_date}.parquet"):
            continue
        dest = data / key
        if dest.exists() and dest.stat().st_size == size:
            skipped += 1
            continue
        if dry_run:
            print(f"would download {key} ({size:,} B)")
            downloaded += 1
            dl_bytes += size
            continue
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(bucket, key, str(dest))
            downloaded += 1
            dl_bytes += size
        except Exception as e:  # noqa: BLE001 — keep pulling the rest
            print(f"FAILED {key}: {e}", file=sys.stderr)
            failed += 1

    verb = "would download" if dry_run else "downloaded"
    scope = f" ({only_date} only)" if only_date else ""
    print(f"{verb} {downloaded} files ({dl_bytes/1e6:.1f} MB), "
          f"{skipped} unchanged, {failed} failed{scope} <- {_endpoint()}/{bucket}")
    return 1 if failed else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    ap.add_argument("--bucket", default=os.environ.get("HIPPIUS_S3_BUCKET", DEFAULT_BUCKET))
    ap.add_argument("--download", action="store_true",
                    help="pull remote -> local (missing or size-changed) instead of uploading")
    ap.add_argument("--today", action="store_true",
                    help="with --download, only pull */<UTC-today>.parquet (the scrape's append target)")
    ap.add_argument("--dry-run", action="store_true", help="list what would transfer, transfer nothing")
    args = ap.parse_args()

    data = Path(args.data_dir)
    s3 = _client()

    if args.download:
        data.mkdir(parents=True, exist_ok=True)
        only_date = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d") if args.today else None
        return _download(s3, args.bucket, data, args.dry_run, only_date)

    return _upload(s3, args.bucket, data, args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
