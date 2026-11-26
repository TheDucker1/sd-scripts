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

## Faster general-only tagging

The prepared local run now uses OppaiOracle V1.1 through the
`Bedovyy/OppaiOracle-V1.1-timm` repack. It has a native 448 x 448 input and
roughly 248M parameters, compared with PixAI's 1008 x 1008 input and 491M.
The reusable settings are in
[`configs/tagger_oppai_oracle_v1_1.json`](../configs/tagger_oppai_oracle_v1_1.json).
This profile emits only general descriptive tags, uses BF16 and an initial
general-tag threshold of 0.75. It keeps the existing maximum of 256 selected
tags. Training still applies shuffling and the configured retention curve later.

The profile uses gray padding (114) and BGR channel order for this repack. The
[repack card](https://huggingface.co/Bedovyy/OppaiOracle-V1.1-timm) explicitly notes
the RGB-to-BGR conversion. The original Grio43 checkpoint instead specifies RGB;
do not mix its preprocessing with the repacked weights. The threshold is based
on the author's approximately 0.75 macro precision/recall operating point in
the [published calibration](https://huggingface.co/Grio43/OppaiOracle/blob/main/V1.1_safetensors/pr_thresholds.json).
This is a starting point, not a measured optimum on this local dataset. The
published file contains per-tag summaries, not a usable full per-tag threshold
table, so this profile uses the calibrated global threshold.

For existing local weights, run:

```bat
"%PY%" -u tools\cache_anima_artist_tags.py --dataset work\danbooru-0000-0101-p20 --config work\danbooru-0000-0101-p20\tagging-config.json --tagger-config configs\tagger_oppai_oracle_v1_1.json --output work\danbooru-0000-0101-p20\tags-oppai-v1_1 --tagger "C:\Users\d\Documents\comfy\ComfyUI\models\wd_taggers\OppaiOracle-V1.1-timm" --device cuda --batch-size 16
```

This reads only model files from the ComfyUI folder; it does not import or start
ComfyUI. The existing timm implementation executes native PyTorch. To download
the same repack elsewhere, replace `--tagger PATH` with
`--download-tagger --tagger-download-dir work\oppai-oracle-v1_1`; the worker reads
the repository and revision from the profile.

The output directory is separate from the earlier PixAI cache. Completed
PixAI captions are not reused for the new model. The worker saves the resolved
`tagging-config.json` beside its journal; use its `tagger` settings in training
with the final `tags-oppai-v1_1/tag-cache.json` so cache identity checks agree.
All artist/date filtering and shard references stay unchanged.
