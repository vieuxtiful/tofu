# Deploying ToFU

A single-host deployment: Docker Compose, Caddy for TLS, SQLite and the
uploads on one bind-mounted directory. This is the shape the demo ships in —
one shared access key, one project space, everything on one box.

It is deliberately not a cluster. The rate limiter counts in process memory,
SQLite takes one writer, and the OCR pipeline holds the request while it
works: every one of those is a single-instance assumption, and pretending
otherwise by adding replicas would break them quietly rather than loudly.

---

## What the box needs

| | minimum | comfortable |
|---|---|---|
| vCPU | 2 | 4 |
| RAM | 4 GB | 8 GB |
| Disk | 25 GB | 50 GB+ |

The app image is around 3 GB — torch (CPU build), EasyOCR and its baked-in
CRAFT/CRNN weights account for most of it. Disk beyond that is uploads and
renders, which grow with use.

Detection is CPU-bound and bursty: one 4 MP photograph is several seconds of
real work. Two vCPU runs, but two people working at once will feel each
other. `TOFU_APP_CPUS` in `.env` leaves headroom for Caddy on purpose — a
container allowed to take every core starves the healthcheck that decides
whether it is alive.

---

## First deployment

**1. Point the domain at the box.** The A/AAAA record must resolve before the
stack first starts; Caddy answers the ACME challenge on port 80 and cannot
obtain a certificate for a name that does not reach it. Open 80 and 443.

**2. Clone and configure.**

```bash
git clone <repo> /srv/tofu && cd /srv/tofu && cp .env.example .env
```

Generate the access key:

```bash
python3 -c "import secrets; print(secrets.token_urlsafe(32))"
```

Put it in `TOFU_API_KEYS` and set `TOFU_DOMAIN`. `server/config.py` refuses to
start in production without both — that refusal is the feature. Read the rest
of `.env.example`; every variable says what breaks if it is wrong.

**3. Build and start.**

```bash
docker compose -f docker-compose.prod.yml up -d --build
```

The first build takes a while: it compiles the frontend, installs torch, and
downloads the EasyOCR weights. That download is in the build on purpose —
otherwise it lands on the first user's first request, from a third party, in
a process already holding their upload.

**4. Check it.**

```bash
curl https://<domain>/api/health
```

Expect `{"status":"ok","environment":"production","auth_enabled":true,...}`.
Then open the domain in a browser: the key gate asks once, and the app is
behind it.

---

## The language models (optional, and worth it)

The OCR pipeline gives a language-model signal 10% of its arbitration weight
and uses it to settle readings the pixels support equally well. With no
artifact installed that signal returns `None`, the weight renormalizes away,
and the pipeline behaves as though it did not exist. **This is the default
state.** Installing the models is what turns it on.

One fetch, two builds, once:

```bash
# 1. corpora (Leipzig Corpora Collection — CC-licensed, checksummed into
#    corpora.lock.json so a rebuild is reproducible)
python scripts/fetch_corpora.py --package fra_news_2024_100K
python scripts/fetch_corpora.py --package jpn_news_2024_100K

# 2. n-gram models — ranks whole readings
python scripts/build_ngram_models.py --family latin \
    --corpus corpora/fra_news_2024_100K.txt --out models/ngram
python scripts/build_ngram_models.py --family cjk \
    --corpus corpora/jpn_news_2024_100K.txt --out models/ngram
```

`./models` is bind-mounted read-only into the container at `/app/models`;
`TOFU_NGRAM_DIR` already points there. Restart the app and confirm:

```bash
curl -H "X-API-Key: $KEY" https://<domain>/api/capabilities | grep -A3 kneser
```

Pick corpora that match what you will actually photograph. The models rank
strings, so a French model on French signage is worth far more than a bigger
model of the wrong language — and `scripts/fetch_corpora.py --list-known`
names the packages that suit each family.

---

## Backups

```bash
scripts/backup.sh /srv/backups
```

Cron it:

```
0 3 * * *  cd /srv/tofu && scripts/backup.sh /srv/backups >> /var/log/tofu-backup.log 2>&1
```

The script snapshots SQLite with `.backup` rather than copying the file. A
live SQLite database under WAL copied with `cp` can capture a torn
transaction, and the copy restores as a corrupt database **without erroring at
backup time** — which is the worst possible failure mode for a backup.

`./models` is excluded: large, and reproducible from `corpora.lock.json`.

**To restore:** stop the stack, replace `./data` with the archive's contents,
start it again.

---

## Upgrading

```bash
cd /srv/tofu && scripts/backup.sh /srv/backups
git pull
docker compose -f docker-compose.prod.yml up -d --build
```

State lives in `./data`, outside the image, so a rebuild keeps it. That is
true only because `server/db.py` resolves the database path from
`TOFU_DATA_DIR` — it used to sit beside the source, inside the image layer,
where a rebuild discarded every project while leaving the files those rows
pointed at on disk.

---

## Rotating the key

Change `TOFU_API_KEYS` in `.env` and restart the app. Every browser session
drops immediately: the session cookie carries the key itself and every request
re-validates it, so revocation does not wait for an expiry.

`TOFU_API_KEYS` takes a comma-separated list, so a rotation can overlap — add
the new key, hand it out, then remove the old one.

---

## What this deployment does not do

**One key, one project space.** The key identifies the deployment, not the
person holding it. Everyone who has it sees every project. That is the demo's
design; per-user accounts (a `users` table, `owner_id` on projects and assets,
an ownership filter on every query) are the next milestone, and the session
cookie is already the transport they will use.

**Single instance.** Adding replicas breaks the rate limiter (per-process
counters), SQLite (one writer), and nothing will tell you.

**No PaddleOCR or LaMa.** The CJK detector and the neural inpainter need their
own interpreters — see the sidecar profiles in `docker-compose.yml`. Without
them CJK detection falls back to EasyOCR and inpainting to the classical
providers, which is a real degradation and a deliberate one: the app reports
what it has on `/api/capabilities` rather than pretending.
