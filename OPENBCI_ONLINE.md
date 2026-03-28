# OpenBCI Online Stream

This repo now includes a Cloudflare Worker that publishes the live OpenBCI
Cyton stream online and archives each live snapshot into Cloudflare D1.

Public endpoints currently in use:

- `https://openbci-status-worker.simfish-openbci-live.workers.dev/`
- `https://openbci-status-worker.simfish-openbci-live.workers.dev/spectrum`
- `https://openbci-status-worker.simfish-openbci-live.workers.dev/live.json`
- `https://openbci-status-worker.simfish-openbci-live.workers.dev/status.txt`
- `https://openbci-status-worker.simfish-openbci-live.workers.dev/history.json?limit=20`
- `https://openbci-status-worker.simfish-openbci-live.workers.dev/spectra.json?limit=12`

Files:

- `openbci_status_worker/`
- `push_openbci_status_online.py`

## Worker behavior

- `GET /`: returns a live dashboard with rolling traces and metrics
- `GET /status.txt`: returns the latest plain-text status
- `GET /status.json`: returns JSON with `status` and `updated_at`
- `GET /live.json`: returns the latest rolling EEG payload with metrics and recent channel samples
- `GET /history.json?limit=60`: returns recent archived snapshot summaries from D1
- `GET /snapshot/<id>.json`: returns a specific archived snapshot payload from D1
- `GET /spectra.json?limit=12`: returns recent archived spectrum snapshots from D1
- `GET /spectrum/<id>.json`: returns a specific archived spectrum payload from D1
- `POST /update`: updates the published status, requires `Authorization: Bearer <STATUS_TOKEN>`
- `POST /live`: updates the published live payload, requires `Authorization: Bearer <STATUS_TOKEN>`

Spectrum archiving:

- each live payload includes a compact PSD snapshot
- the Worker archives one spectrum snapshot about every `120` seconds by default
- the interval is controlled by `SPECTRUM_ARCHIVE_INTERVAL_SEC` in `wrangler.toml`

## Deploy steps

1. Create a KV namespace named `STATUS_KV` and bind it in `wrangler.toml`.
2. Set a Worker secret named `STATUS_TOKEN`.
3. Deploy the Worker.
4. Create the D1 database and apply migrations.

Suggested commands:

```bash
cd openbci_status_worker
npx wrangler kv namespace create STATUS_KV
npx wrangler d1 create openbci-archive
npx wrangler d1 migrations apply openbci-archive --remote
npx wrangler secret put STATUS_TOKEN
npx wrangler deploy
```

Then run the uploader locally:

```bash
python3.11 push_openbci_status_online.py \
  --status-path openbci_live_status.txt \
  --payload-path openbci_live_payload.json \
  --status-url https://<your-worker-domain>/update \
  --live-url https://<your-worker-domain>/live \
  --token <same-token>
```
