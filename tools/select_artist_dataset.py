"""Select artists by valid pair count using metadata only; hard-link local tars."""

import argparse
import copy
import csv
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from library.danbooru_index import file_sha256, index_path, load_artist_index, write_index_arrays, write_index_json


def select_rows(rows, artists, window_days, min_pairs, first_archive, last_archive):
    if min_pairs < 1 or window_days < 1 or first_archive < 0 or last_archive < first_archive:
        raise ValueError("Invalid pair threshold, date window or archive range")
    indices = np.flatnonzero((rows["archive"] >= first_archive) & (rows["archive"] <= last_archive))
    block = rows[indices]
    if not len(block):
        raise ValueError("No images in the selected archive range")
    window = window_days * 86400
    neighbors = ((block["artist"][1:] == block["artist"][:-1])
                 & (block["timestamp"][1:] - block["timestamp"][:-1] <= window))
    paired = np.zeros(len(block), dtype=bool)
    paired[:-1] |= neighbors
    paired[1:] |= neighbors
    indices, block = indices[paired], block[paired]
    groups = np.r_[0, np.flatnonzero(np.diff(block["artist"])) + 1, len(block)]
    selected, report = [], []
    for start, end in zip(groups[:-1], groups[1:]):
        if start == end:
            continue
        times = block["timestamp"][start:end]
        if np.any(np.diff(times) < 0):
            raise ValueError("Artist dates must be sorted")
        count = int(np.sum(np.searchsorted(times, times + window, side="right") - np.arange(len(times)) - 1))
        qualified = count >= min_pairs
        report.append(dict(artist=str(artists[int(block[start]["artist"])]), images=int(end - start),
                           pairs=count, selected=qualified, first_timestamp=int(times[0]), last_timestamp=int(times[-1])))
        if qualified:
            selected.extend(indices[start:end])
    if not selected:
        raise ValueError("No artists meet the minimum pair count")
    return np.asarray(selected, dtype=np.int64), report


def build_selection(dataset, output, min_pairs=20, first_archive=0, last_archive=None, config=None):
    dataset, output = Path(dataset).resolve(), Path(output).resolve()
    if output == dataset:
        raise ValueError("Selection output must differ from the source dataset")
    manifest, rows, artists = load_artist_index(dataset)
    if manifest.get("storage") != "local_shards":
        raise ValueError("Selection worker expects a completed resized local_shards dataset")
    last_archive = len(manifest["archives"]) - 1 if last_archive is None else last_archive
    if last_archive >= len(manifest["archives"]):
        raise ValueError("Archive range exceeds the completed snapshot")
    identity = dict(format="artist-pair-selection/1", source_index_sha256=manifest["index_sha256"],
                    min_pairs=min_pairs, first_archive=first_archive, last_archive=last_archive,
                    pair_max_days=manifest["pair_max_days"])
    if (output / "selection.json").exists():
        if json.loads((output / "selection.json").read_text()) != identity:
            raise ValueError("Output belongs to a different selection; choose a new directory")
    elif output.exists() and any(output.iterdir()):
        raise ValueError("Output is not an empty selection directory")
    selected, report = select_rows(rows, artists, manifest["pair_max_days"], min_pairs, first_archive, last_archive)
    output.mkdir(parents=True, exist_ok=True)
    write_index_json(output / "selection.json", identity)
    with np.load(index_path(dataset, manifest["index_file"]), allow_pickle=False) as data:
        files, hashes = data["files"][selected], data["sha256"][selected]
    rows = rows[selected].copy()
    artist_ids = np.unique(rows["artist"])
    rows["artist"] = np.searchsorted(artist_ids, rows["artist"])
    archive_ids = np.unique(rows["archive"])
    rows["archive"] = np.searchsorted(archive_ids, rows["archive"])
    archives = []
    for number in archive_ids:
        archive = copy.deepcopy(manifest["archives"][int(number)])
        source = index_path(dataset, archive["file"])
        destination = index_path(output, archive["file"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if not os.path.samefile(source, destination):
                raise ValueError(f"Selection archive is not the original hard link: {destination}")
        else:
            try:
                os.link(source, destination)
            except OSError as error:
                raise OSError("Output must support hard links on the source filesystem; no tar is copied") from error
        if destination.stat().st_size != archive["size"]:
            raise ValueError("Completed source tar size changed")
        archives.append(archive)
    write_index_arrays(output / "artist-index.npz", images=rows, artists=artists[artist_ids], files=files, sha256=hashes)
    saved = dict(manifest, archives=archives, image_count=len(rows), artist_count=len(artist_ids),
                 index_file="artist-index.npz", index_sha256=file_sha256(output / "artist-index.npz"),
                 artist_selection=identity)
    write_index_json(output / "manifest.json", saved)
    qualified = [item for item in report if item["selected"]]
    for filename, records in (("all-artists.csv", report), ("artists.csv", qualified)):
        with (output / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(report[0]))
            writer.writeheader()
            writer.writerows(records)
    summary = dict(identity, source_artists=len(report), source_images=sum(item["images"] for item in report),
                   selected_artists=len(qualified), selected_images=len(rows),
                   selected_unordered_pairs=sum(item["pairs"] for item in qualified),
                   selected_image_bytes=int(rows["size"].sum()), shared_tar_bytes=sum(a["size"] for a in archives),
                   additional_image_bytes=0,
                   images_per_selected_artist=dict(min=min(item["images"] for item in qualified),
                       median=float(np.median([item["images"] for item in qualified])),
                       max=max(item["images"] for item in qualified)))
    write_index_json(output / "summary.json", summary)
    if config is not None:
        settings = json.loads(Path(config).read_text(encoding="utf-8"))
        settings["cache_latents"] = settings["cache_latents_to_disk"] = False
        settings.pop("tag_cache", None)
        settings.pop("latent_cache", None)
        settings["training"]["batch_size"] = 1
        geometry = manifest["resize"]
        settings["data"] = dict(artist_index=str(output), index_cache_bytes=0, index_fetch_workers=2,
            datasets=[dict(batch_size=1, resolution=[geometry["pixels"]] * 2, max_pixels=geometry["pixels"] ** 2,
                enable_bucket=True, min_bucket_reso=geometry["min_bucket_reso"], max_bucket_reso=geometry["max_bucket_reso"],
                bucket_reso_steps=geometry["bucket_reso_steps"], bucket_no_upscale=geometry["no_upscale"],
                max_aspect_ratio=geometry["max_aspect_ratio"], repeat=True, shuffle_buffer=0,
                shuffle_caption=True, resize_interpolation=geometry["interpolation"], subsets=[{}])])
        write_index_json(output / "tagging-config.json", settings)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-pairs", type=int, default=20)
    parser.add_argument("--first-archive", type=int, default=0)
    parser.add_argument("--last-archive", type=int)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/anima_style_colab.json")
    args = parser.parse_args()
    build_selection(args.dataset, args.output, args.min_pairs, args.first_archive, args.last_archive, args.config)


if __name__ == "__main__":
    main()
