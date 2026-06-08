#!/usr/bin/env bash
# Fetch the Chinook SQLite sample database used by the RAG / tool-use
# scenarios.
#
# Chinook is a tiny (~1 MB) sample database modelling a digital media
# store: artists, albums, tracks, customers, invoices. Lerocha's mirror
# is the most-cited canonical source.
#
# Output: <repo>/data/chinook.db (gitignored; fetch your own copy so
# byte-stability is on your side).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="${REPO_ROOT}/data"
DB_PATH="${DATA_DIR}/chinook.db"
SOURCE_URL="https://github.com/lerocha/chinook-database/raw/master/ChinookDatabase/DataSources/Chinook_Sqlite.sqlite"

mkdir -p "${DATA_DIR}"

if [[ -f "${DB_PATH}" ]]; then
  echo "[chinook] already present at ${DB_PATH} — skipping download."
  exit 0
fi

echo "[chinook] downloading from ${SOURCE_URL}"
curl -fL -o "${DB_PATH}" "${SOURCE_URL}"

# Quick sanity check: SQLite file magic is "SQLite format 3\000".
if ! head -c 16 "${DB_PATH}" | grep -q "SQLite format 3"; then
  echo "[chinook] downloaded file is not a valid SQLite database — aborting." >&2
  rm -f "${DB_PATH}"
  exit 1
fi

echo "[chinook] saved to ${DB_PATH} ($(du -h "${DB_PATH}" | cut -f1))"
