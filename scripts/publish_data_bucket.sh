#!/usr/bin/env sh
# Publish the scraper's output to a private S3-compatible bucket, so a
# downstream consumer (e.g. cascade's eval-pool publisher) can mirror the
# catalog + dated parquet without touching this host. Append this to the
# scrape cron:
#
#   0 * * * *  python src/sources/scraper.py --all && TSFORGE_BUCKET=... scripts/publish_data_bucket.sh
#
# Configuration (env):
#   TSFORGE_BUCKET        target bucket, e.g. "tsforge-raw" (required)
#   TSFORGE_BUCKET_PREFIX optional key prefix inside the bucket (default "")
#   S3_ENDPOINT_URL       optional custom endpoint (Cloudflare R2, MinIO, ...)
#   AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY  credentials (or an aws profile)
#
# The layout is dated and append-only (data/<source_id>/<YYYY-MM-DD>.parquet),
# so sync is cheap and idempotent; a partial upload just completes next run.
# Keep the bucket PRIVATE: the catalog is public, but the exact curated
# snapshot a consumer evaluates on should not be handed out.
set -eu

SRC_DIR="$(cd "$(dirname "$0")/.." && pwd)/src/sources"
: "${TSFORGE_BUCKET:?set TSFORGE_BUCKET to the target bucket name}"
PREFIX="${TSFORGE_BUCKET_PREFIX:-}"
DEST="s3://${TSFORGE_BUCKET}${PREFIX:+/${PREFIX}}"
ENDPOINT_ARGS=""
[ -n "${S3_ENDPOINT_URL:-}" ] && ENDPOINT_ARGS="--endpoint-url ${S3_ENDPOINT_URL}"

# The parquet mirror; --size-only keeps re-uploads to genuinely new files
# (dated snapshots are never rewritten in place).
aws s3 sync "${SRC_DIR}/data" "${DEST}/data" --size-only ${ENDPOINT_ARGS}

# The catalog the consumer needs to interpret the data (domains, frequencies,
# panels, disabled flags). Copied last so a consumer that sees the new catalog
# also sees the data it describes.
aws s3 cp "${SRC_DIR}/sources.yaml" "${DEST}/sources.yaml" ${ENDPOINT_ARGS}

echo "published ${SRC_DIR} -> ${DEST}"
