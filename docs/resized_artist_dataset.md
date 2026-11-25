# Download a bucketed JPEG artist dataset, one tar at a time

This worker is set up for the current Windows machine and existing `ml` Python:

```text
C:\Users\d\.conda\envs\ml\python.exe
```

It uses the existing full `dataset-index-20261107T072410Z-1-001.zip`, not the small
previously downloaded artist subset. There is no metadata rebuild or 20 GB posts
download. All images in the final index are considered, retaining its single-artist,
date and source-size filtering. The index identifies the correct pinned original,
recent or update archive for each image; obsolete copies are not selected again.

The default output is `C:\Users\d\playground\sd-scripts\danbooru-artists-512`.
Original input archives are staged on that same drive, one at a time.

## Run later from CMD

Use your saved Hugging Face login. If you have not logged in locally:

```bat
"C:\Users\d\.conda\envs\ml\Scripts\hf.exe" auth login
```

An existing `HF_TOKEN` environment variable also works and takes precedence over
the saved token. The worker never writes the token to its plan, receipts or logs.

Preview the archive/image counts and remaining source bytes without downloading:

```bat
tools\download_resized_artist_dataset.cmd --dry-run
```

Process just one additional archive, then stop cleanly:

```bat
tools\download_resized_artist_dataset.cmd --max-archives 1
```

Process the rest, or resume an interrupted run:

```bat
tools\download_resized_artist_dataset.cmd
```

The launcher changes to the repository root and uses the Python above. To choose
another output disk on this same machine:

```bat
tools\download_resized_artist_dataset.cmd --output "D:\danbooru-artists-512"
```

Keep the same output/source/resize settings when resuming. Worker count, retry
count, disk reserve and archive limit may change. An output lock prevents two
workers from processing the same destination simultaneously.

## Download, resize, commit, remove

For each source tar the worker:

1. Uses `huggingface_hub.hf_hub_download` with the saved token and index's pinned
   commit. `local_dir` points inside this output's temporary download directory.
   It does not use a second global Hub blob copy. The installed `hf_xet` transport
   handles authenticated CDN transfers and download progress; high-performance
   mode is enabled unless already configured otherwise. SDK metadata and a partial
   original download are retained on interruption so the SDK can recover it.
2. Reads only the indexed image members, validating tar member ID/size and exact
   metadata dimensions. Originals are never extracted as individual files.
3. Reuses `make_stream_bucket_manager`, `stream_rgb_image`, and
   `trim_and_resize_if_required`: aspect-ratio bucket selection, area resizing,
   deterministic center crop, EXIF orientation and white transparency compositing.
4. Writes JPEG members into a compact output tar. Default quality **95**, chroma
   subsampling **0 / 4:4:4**, and JPEG Huffman optimization retain colored line
   detail. Pixel budget is **512 x 512 total pixels**, not a fixed square crop.
   Defaults match the current training buckets: step 64, minimum side 256, maximum
   side 1536 and aspect-ratio limit 3. No image is upscaled.
5. Finishes the compact tar, offset/checksum index and completion receipt, flushes
   them, then renames the whole output shard directory into place.
6. Only after that commit removes the staged original tar and its SDK metadata.
   If removal fails, it stops; restarting cleans that committed tar's staging
   directory before advancing. It does not fetch another original while leaving
   the previous staged archive behind.

See the [Hub local-directory download documentation](https://huggingface.co/docs/huggingface_hub/guides/download#download-files-to-a-local-folder)
and [Xet high-performance setting](https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables#hf_xet_high_performance).
This machine already has `huggingface_hub` and `hf_xet` installed; the script loads
no models and performs no GPU work. Bucket/image helpers import the project's
existing Python dependencies.

## Resume is per tar

There are **no individual-image resume journals**. If resizing is interrupted,
the unfinished output tar is discarded and that source tar is processed again
from the start. A complete staged original can be reused by the SDK. Completed
compact tars are verified by receipt, file size and index checksum and skipped
without requesting their source archives again.

`--max-archives 1` means one additional unfinished archive in that invocation.
Running it repeatedly advances through archives. `--verify-shards` additionally
rehashes all completed compact tars; ordinary resume avoids rereading the whole
processed dataset. Equal-size JPEG corruption is also detected by per-image SHA256
when training/cache workers fetch the affected member.

Wrong source/index member IDs or lengths stop the archive without committing it.
Corrupt images, dimension mismatches, unsupported/animated images, extreme aspect
ratios and images that exceed the size limits are listed in the shard's
`skipped.json`. They are not silently resized using corrected metadata dimensions.
Known metadata-MD5 discrepancies are counted after member identity and dimensions
are checked; the newly encoded JPEG always receives its own MD5 and SHA256.

## Output, progress and training

```text
danbooru-artists-512/
  resize-plan.json
  progress.json
  manifest.json
  indexes/artist-index-<snapshot>.npz
  shards/00000/
    images.tar
    images.npz
    skipped.json
    complete.json
  shards/00001/...
```

Terminal progress shows completed tar checks, the HF download, and images resized
inside the current tar. `progress.json` records the current phase/tar, committed
counts/bytes, skips and a rough projected final compact size based on completed
images. That projection is an estimate, not a capacity guarantee. The script
checks space for the next original tar and defaults to a **4 GiB free disk reserve**.
The reserve is checked during output writes too. Compact outputs still accumulate;
only originals are removed.

On clean stop or interruption, a training snapshot is built from fully committed
tars. Artist IDs and Unix timestamps are preserved, output image sizes/checksums
are updated, and only images with another same-artist image within the index's
365-day constraint enter the training snapshot. Unpaired images remain stored in
their compact tar and can acquire partners when later source tars are processed.
No minimum pair count or random artist subset is imposed.

The existing artist loader accepts `storage=local_shards` and reads JPEG byte
ranges directly from these local compact tars, with at most eight open tar
handles. Artist cycling and the date constraint are preserved. The
Anima tag/latent cache workers can use this dataset directly; no image extraction
or unpacking of millions of JPEGs is necessary. Rebuild caches for this new image
dataset rather than reusing caches tied to the previous original bytes/index.

Snapshots use immutable index filenames, and rerunning without new committed
tars does not rewrite the index or change its identity. Existing training config
files are not automatically switched to this larger dataset.

To rebuild the index without downloading or resizing:

```bat
tools\download_resized_artist_dataset.cmd --finalize-only
```

Other controls: `--quality 92`, `--workers 4`, `--reserve-disk-gib 8`,
`--max-member-mib 128`, `--max-image-pixels 64000000`, `--retries 5`.
Changes to quality/bucket/image-limit settings require a new output directory.

The old `pack_artist_dataset.py` command packs loose-file datasets. It is not
needed for these already compact local shards; this worker and loader operate
directly on the shard directory.

## Test scope

`tests/test_artist_shard_resize.py` uses seven tiny generated members across two
local tars. Tests cover archive-level interruption/restart, keeping originals
until commit, cleanup failure recovery, strict mismatch skips, white alpha,
bucket sizes, training pair reads, artist/date constraints, stable resume IDs,
checksums and mocked authenticated SDK calls. No real dataset archive or model
is downloaded during tests.
