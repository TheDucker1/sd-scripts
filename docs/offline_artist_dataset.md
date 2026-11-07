# Download an offline artist subset

`tools/download_artist_dataset.py` runs locally without importing torch, loading
models, scanning tars or downloading metadata. It accepts your completed
`dataset-index` ZIP directly, or an extracted index directory. From a ZIP it
extracts only the final `manifest.json` and `artist-index.npz`; intermediate builder
files are ignored. The full final image table is loaded into RAM for selection.

Install the small downloader dependencies in the Python environment you will use:

```powershell
& C:\Users\d\.conda\envs\ml\python.exe -m pip install numpy Pillow tqdm "httpx[http2]"
```

First select and report the subset, without making any network requests:

```powershell
& C:\Users\d\.conda\envs\ml\python.exe tools/download_artist_dataset.py `
  --index dataset-index-20261107T072410Z-1-001.zip `
  --output D:\danbooru-offline --dry-run
```

Start downloading with the same command without `--dry-run`:

```powershell
& C:\Users\d\.conda\envs\ml\python.exe tools/download_artist_dataset.py `
  --index dataset-index-20261107T072410Z-1-001.zip `
  --output D:\danbooru-offline
```

The default rolls artists without replacement, seed 42, until **1,000 artists**
qualify. For each artist, files are sorted by timestamp. Each image contributes
its later partners within the index's saved date constraint (365 days for this
index). Suffix pair counts identify feasible starting images. The worker picks
one of those starts randomly, then advances the end only until **101 distinct
unordered pairs** are available. An artist with too few pairs is skipped and
another is rolled. A/B and B/A count as one pair; self-pairs are excluded.

The endpoint is the minimum for the chosen random start. The selected range can
span multiple years: the one-year constraint applies to each pair, not the entire
range. For dense dates, `--min-pairs 50` selects 11 images (55 pairs); the default
101 needs 15 (105 pairs). Sparse dates may require more images. The worker downloads
only that selected range, once per image, retaining original bytes/resolution.

`selection.json` lists each chosen artist, image/pair counts, bytes and subfolder.
The summary reports total images, payload GiB, and anticipated multi-range request
count, alongside the single-range fallback count. It does not reserve disk space.
To try a smaller selection, use `--artists 2` and a different output directory.
`--min-pairs`, `--seed` and `--artists` are selection settings; keep them unchanged
when resuming. Connection tuning options may be changed on resume.

## Connections and exact ranges

Images are grouped by their pinned tar and sorted by byte offset. Requests contain
only each selected image's 512-byte tar header and its exact payload. **Ranges are
never merged across gaps** and no whole tar is downloaded.

The downloader tries 16 separate ranges in one HTTP request, capped at 64 MiB of
requested bytes. It validates multipart `Content-Range` values and payload lengths.
If the server rejects multi-range requests, combines ranges or only returns one,
it switches to individual exact-range requests for the rest of that invocation.
It rejects an HTTP 200/full-archive response without reading its body.

A persistent HTTPX client reuses its connection, with one connection allowed at
a time. HTTP/2 is negotiated when available; up to four fallback workers can then
multiplex requests on it. With HTTP/1.1 those requests queue on the same pooled
connection. Redirects/expiry/server disconnects can require a new connection;
there is no guarantee a single TCP socket survives an entire tar or run. Resolved
CDN URLs are cached for each tar and refreshed after authorization/expiry errors.
Public data normally needs no token; an optional `HF_TOKEN` environment variable
is sent only to Hugging Face, not directly to CDN URLs.

Defaults allow one request start every 0.2 seconds, eight retries and a 60-second
socket timeout. Retries use exponential backoff, jitter and shared `Retry-After`
cooldowns, including HTTP 429. Connection reuse reduces reconnects; it does not
remove the server's request limits. Useful tuning flags:

| Flag | Default | Meaning |
|---|---|---|
| `--ranges-per-request` | 16 | Exact separate ranges per request; 1 skips multi-range probing. |
| `--response-mib` | 64 | Cap requested bytes per batch; increase if one image exceeds it. |
| `--workers` | 4 | Fallback single-range workers, maximum 8. |
| `--request-interval` | 0.2 | Minimum seconds between request starts, across workers. |
| `--retries` | 8 | Retries after the initial request. |
| `--timeout` | 60 | Network operation timeout in seconds. |
| `--http1` | off | Disable HTTP/2 negotiation if a proxy/server has issues. |
| `--retry-skipped` | off | Retry invalid images previously recorded in `skipped.jsonl`. |
| `--index-subdir` | automatic | Select a completed index when a ZIP contains several. |

## Resume, validation and progress

Rerun the same command to continue. The compact selection is saved. Completed
images are rechecked against their recorded sizes and **actual payload SHA256**
with a verification progress bar; missing/corrupt files are downloaded again.
Single-range transfers retain partial bytes and request only the remainder after
interruptions. An interrupted multipart response retries its bounded batch; images
already committed to the journal remain completed. An OS lock prevents two workers
from writing the same output directory and is released even after a process crash.

Each image must match the tar header's post ID/size and decode fully. If its actual
dimensions differ from metadata, the downloader records the correction and uses
the actual dimensions in the final index, provided they still pass the source
index's `min_side`/`min_pixels` filters. Metadata MD5 differences are recorded,
rather than rejecting valid archive images: this dataset can contain valid bytes
with a different MD5 from `posts.json`. Local SHA256 is computed from the downloaded
payload itself.

Invalid images (bad member identity, corrupt payload or below-minimum actual size)
are recorded in `skipped.jsonl` and do not abort later downloads. The worker excludes
failed posts and picks replacement ranges from the source index, reusing verified
downloads. If an artist no longer supplies enough pairs, another is rolled. It
finishes only with the requested artist count, or reports that too few artists
remain. Disk failures and exhausted network/protocol failures still stop with
recoverable state; they do not silently exclude images.

Older oversized plans automatically shrink when you rerun the same command in
the same output directory. Existing artist folders are retained where possible;
download journals are matched by post ID, so changing row positions does not lose
completed work. A new versioned NPZ is written before switching `selection.json`
atomically. Old plans and unselected image files stay on disk. Stop an older
downloader process before rerunning. Use the packer below to exclude those leftover
files from your Colab upload.

Before finalizing, the index drops any images with no valid partner. The date
constraint is unchanged; `artist_pair_counts` records the usable pair counts.

The image progress bar shows processed/total (saved plus skipped), speed and ETA. Its suffix shows the
current tar and remaining payload GiB; a separate network bar shows bytes and
throughput, including retry/multipart overhead. `progress.json` records counts,
payload bytes, request/retry totals for the current invocation, MD5 discrepancies
and completion state. The journal is flushed per image and synced after each batch.

Output structure:

```text
danbooru-offline/
  selection.json
  download-plan-range-<hash>.npz
  downloaded.jsonl
  progress.json
  manifest.json
  artist-index.npz
  images/a000123-artist_name/123456.jpg
  images/a000456-another_artist/234567.png
  .source-index/             # Only when input was a ZIP
  .partial/                 # Interrupted single-range payloads
```

The final index becomes `complete` only after every selected image has been saved
or explicitly skipped and the remaining pair eligibility has been checked.
It contains relative paths, so move/copy the **whole output directory** as needed.
Set the existing training configuration to:

```python
config['data']['artist_index'] = r'D:\danbooru-offline'
```

Artist/date pairing, bucketing, resizing, prompts and prefetching still use the
existing pipeline. The local backend verifies payload SHA256 and performs no HTTP
requests.

## Colab: load a tar into RAM once

After the downloader finishes, pack its training files as an **uncompressed tar**:

```powershell
& C:\Users\d\.conda\envs\ml\python.exe tools/pack_artist_dataset.py `
  --dataset D:\danbooru-offline --output D:\artists-pair-dataset.tar
```

The packer includes only the completed index, its manifest and referenced images.
Old unselected downloads, caches and journals are omitted. Selected files are
verified against their saved SHA256, and the final tar appears only after packing
finishes. An existing output tar is never overwritten. Do not pack a still-running
or incomplete download. Upload the tar to
`MyDrive/moss-training/artists-pair-dataset.tar` and run the updated Anima notebook's
dataset cell. It sets `INDEX_DIR` to `/dev/shm/anima-artist-dataset` by default.

`tools/colab/stage_artist_dataset.py` copies the tar from Drive once with progress
and partial-copy resume. It verifies the final index and each image's saved SHA256,
then saves a compact `local_tar` index pointing at the new archive offsets. Images
stay in their original JPEG/PNG/etc encoding inside the uncompressed tar. Training
reads exact image bytes from RAM; there is no image-file extraction or second
in-process copy of the whole dataset. Repeating the staging cell reuses its ready
cache without rereading the Drive archive body. Source size/mtime changes require
a new staging directory. Keep the same tar for checkpoint resume: repacking can
change offsets and the resolved dataset index checksum.

The default `DATA_STORAGE='ram'` checks for an actual tmpfs mount and sufficient
space/available memory, with `RESERVE_RAM_GIB=12` left for model loading, CPU buffers
and training. The tar must fit both the tmpfs capacity and available RAM. `/content`
is disk, not RAM. For a dataset that does not fit, choose fewer artists or explicitly
set `DATA_STORAGE='local'`: the notebook then stages to Colab's local SSD. Neither
mode reads image data from Drive during training. Nothing silently falls back to
Drive, and a Colab runtime reset requires staging again.

The saved `pair_max_days` remains authoritative. The downloader counts only
distinct unordered same-artist pairs within that window and drops images with no
partner. The selected range can retain separate time clusters from the same artist: the training
loader's timestamp bounds always choose B within 365 days of A, including the
boundary, and exclude A itself. Packing/staging does not relax that constraint.
