# Select completed artist shards and tag offline

The selection step reads only the completed NPZ index. It keeps artists with at
least the requested number of distinct unordered image pairs whose dates are
within the index's existing `pair_max_days` window. A/B and B/A count as one pair;
a post cannot pair with itself. The worker keeps all images with valid partners
from qualifying artists, rather than selecting a minimal window of images.

From Windows CMD:

```bat
set "PY=C:\Users\d\.conda\envs\ml\python.exe"
"%PY%" tools\select_artist_dataset.py --dataset danbooru-artists-512 --output work\danbooru-0000-0101-p20 --first-archive 0 --last-archive 101 --min-pairs 20
```

Archive bounds are inclusive positions in the completed snapshot. On the local
snapshot for shards 0–101, the threshold selects 12,745 artists, 289,767 images
and 2,593,027 pairs within 365 days. These counts precede online near-duplicate
classification and any later image/tag preprocessing rejects.

The output is a standard `local_shards` artist index. Its tar files are hard
links to the source archives, so the output must be on the same filesystem with
hard-link support. No JPEGs or tars are copied, changed or removed. The linked
tars still contain images from excluded artists; only selected image records
appear in the new index. Do not count linked file sizes twice when estimating
physical storage. Keep the selection stable while tagging; changed selection
settings need another output folder.

Outputs include `summary.json`, `artists.csv` for qualifying artists,
`all-artists.csv` for all evaluated artists, `manifest.json`, `artist-index.npz`,
and a `tagging-config.json` matching the resized shard geometry. Dates in the
artist reports are Unix timestamps. Rerunning the same selection command reuses
the hard links.

Run the PixAI tagger only, with BF16 weights and batch size four:

```bat
"%PY%" -u tools\cache_anima_artist_tags.py --dataset work\danbooru-0000-0101-p20 --config work\danbooru-0000-0101-p20\tagging-config.json --output work\danbooru-0000-0101-p20\tags --download-tagger --tagger-download-dir work\pixai-tagger --device cuda --batch-size 4
```

`--download-tagger` fetches/reuses only the tagger assets. If the model already
exists elsewhere, replace that flag and `--tagger-download-dir` with
`--tagger PATH`. No Anima, Qwen text encoder, VAE, LoRA or loss model is loaded.

Progress displays a tqdm count, speed and ETA. Completed batches are appended to
one `batches.jsonl` journal, rather than separate files for every batch. Rerun the
same command after interruption to skip completed IDs. An incomplete final
journal line is discarded; corruption in a completed line raises an error.
Legacy `part-*.json` progress files remain readable. Image reads are ordered by
archive and byte offset for local shard access.

The final `tags/tag-cache.json` stores the complete threshold-selected tags in
unshuffled order. Per-prompt retention and shuffling happen later in training;
tagging does not apply them. Use the selected index and the same image geometry
when loading this tag cache in training.
