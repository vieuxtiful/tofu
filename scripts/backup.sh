#!/usr/bin/env bash
## 🍢 ToFU — one deployment, one archive
## vieuxtiful
##
##   scripts/backup.sh [destination-dir]
##
## Run from the deployment directory (the one holding docker-compose.prod.yml
## and ./data). Suitable for cron:
##
##   0 3 * * *  cd /srv/tofu && scripts/backup.sh /srv/backups >> /var/log/tofu-backup.log 2>&1
##
## The database is copied with sqlite3's own .backup, not with cp. A live
## SQLite database is a file being written to under WAL: copying it with cp
## captures whatever pages happened to be on disk at the moment, which may be
## a torn transaction, and the copy restores as a corrupt database WITHOUT
## erroring at backup time. .backup takes a read lock and walks the pages
## consistently. This is the single reason this script exists rather than a
## line of tar in the crontab.

set -euo pipefail

DEST="${1:-./backups}"
STAMP="$(date +%Y%m%d-%H%M%S)"
DATA_DIR="./data"
KEEP="${TOFU_BACKUP_KEEP:-14}"

if [ ! -d "$DATA_DIR" ]; then
  echo "no $DATA_DIR here -- run this from the deployment directory" >&2
  exit 1
fi

mkdir -p "$DEST"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

DB="$DATA_DIR/app/tofu.db"
if [ -f "$DB" ]; then
  echo "snapshotting database"
  # Prefer the sqlite3 in the app container so the client version always
  # matches the one that wrote the file; fall back to a host sqlite3.
  if docker compose -f docker-compose.prod.yml ps -q app >/dev/null 2>&1 \
     && [ -n "$(docker compose -f docker-compose.prod.yml ps -q app)" ]; then
    docker compose -f docker-compose.prod.yml exec -T app \
      python -c "
import sqlite3, sys
src = sqlite3.connect('/app/data/tofu.db')
dst = sqlite3.connect('/app/data/.backup.tofu.db')
src.backup(dst); dst.close(); src.close()
"
    mv "$DATA_DIR/app/.backup.tofu.db" "$STAGE/tofu.db"
  elif command -v sqlite3 >/dev/null 2>&1; then
    sqlite3 "$DB" ".backup '$STAGE/tofu.db'"
  else
    echo "REFUSING to cp a live SQLite database: no container and no sqlite3." >&2
    echo "Start the stack, or install sqlite3, then rerun." >&2
    exit 2
  fi
else
  echo "no database at $DB (fresh deployment?)"
fi

echo "staging uploads, outputs, thumbnails and certificates"
## Everything is assembled under one root before tar runs, rather than tar
## being handed several -C hops and a conditional argument. The conditional
## form works until the day the database is absent, at which point the shell
## expands it to nothing and tar silently archives the wrong tree.
ROOT="$STAGE/tofu-$STAMP"
mkdir -p "$ROOT"
[ -f "$STAGE/tofu.db" ] && mv "$STAGE/tofu.db" "$ROOT/tofu.db"

## The live database and its sidecars are excluded: the consistent snapshot
## above is the copy that restores.
tar -cf - \
  --exclude='app/tofu.db' \
  --exclude='app/tofu.db-wal' \
  --exclude='app/tofu.db-shm' \
  -C "$DATA_DIR" . \
  | tar -xf - -C "$ROOT"
[ -f .env ] && cp .env "$ROOT/.env"

ARCHIVE="$DEST/tofu-$STAMP.tar.gz"
tar -czf "$ARCHIVE" -C "$STAGE" "tofu-$STAMP"

SIZE="$(du -h "$ARCHIVE" | cut -f1)"
echo "wrote $ARCHIVE ($SIZE)"

## Pruning is by count, not age: a deployment that stops being backed up for a
## month should still keep its last good archives rather than expire them all
## on the day someone notices.
COUNT="$(find "$DEST" -maxdepth 1 -name 'tofu-*.tar.gz' | wc -l)"
if [ "$COUNT" -gt "$KEEP" ]; then
  find "$DEST" -maxdepth 1 -name 'tofu-*.tar.gz' -print0 \
    | sort -z \
    | head -z -n "$((COUNT - KEEP))" \
    | xargs -0 rm -f
  echo "pruned $((COUNT - KEEP)) archive(s), keeping $KEEP"
fi

## Note what is NOT in here, deliberately: the built n-gram and vector models
## under ./models. They are reproducible from corpora.lock.json and the build
## scripts, they are large, and a backup that includes them is one an operator
## stops running because it takes too long.
echo "done. (./models is excluded -- rebuild it from corpora.lock.json)"
