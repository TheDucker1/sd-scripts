"""Run only a local PyTorch tagger over an artist index and save portable captions."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.download_artist_dataset import OutputLock
from library.danbooru_index import file_sha256, write_index_json


def read_tag_journal(path, captions):
    """Recover completed batches and discard only an interrupted trailing write."""
    if not path.exists():
        return
    with path.open("r+b") as handle:
        while True:
            start = handle.tell()
            line = handle.readline()
            if not line:
                break
            if not line.endswith(b"\n"):
                handle.seek(start)
                handle.truncate()
                break
            saved = json.loads(line)
            if not isinstance(saved, dict) or any(value is not None and not isinstance(value, str) for value in saved.values()):
                raise ValueError("Invalid caption batch in tag progress journal")
            if captions.keys() & saved.keys():
                raise ValueError("Duplicate IDs in tag progress journal")
            captions.update(saved)


def cache_tags(pairs, settings, output, source, load_tagger, batch_size, device):
    from tqdm import tqdm

    from library.anima_tag_cache import tag_cache_identity

    identity = tag_cache_identity(pairs, settings)
    progress_path = output / "progress.json"
    progress = dict(identity=identity, source=source)
    if progress_path.exists():
        if json.loads(progress_path.read_text(encoding="utf-8")) != progress:
            raise ValueError("Tag cache inputs changed. Choose another output directory")
    else:
        write_index_json(progress_path, progress)

    parts = sorted(output.glob("part-*.json"))
    captions = {}
    for part in parts:
        saved = json.loads(part.read_text(encoding="utf-8"))
        if captions.keys() & saved.keys():
            raise ValueError(f"Duplicate IDs in tag cache progress: {part}")
        captions.update(saved)
    journal = output / "batches.jsonl"
    read_tag_journal(journal, captions)
    indices = {str(int(row["id"])): index for index, row in enumerate(pairs.rows)}
    if captions.keys() - indices.keys():
        raise ValueError("Tag cache progress contains unknown image IDs")
    pending = [post_id for post_id in indices if post_id not in captions]
    if "archive" in pairs.rows.dtype.names and "offset" in pairs.rows.dtype.names:
        pending.sort(key=lambda post_id: (int(pairs.rows[indices[post_id]]["archive"]),
                                          int(pairs.rows[indices[post_id]]["offset"])))

    def read(post_id):
        return pairs.process(pairs.fetch(indices[post_id]), random.Random(int(post_id)))

    if pending:
        tagger = load_tagger().requires_grad_(False).eval()
        with journal.open("a", encoding="utf-8", newline="\n") as handle, \
                ThreadPoolExecutor(max_workers=pairs.fetch_workers) as pool, tqdm(
            total=len(indices), initial=len(captions), unit="image", desc="Cache Anima tags"
        ) as bar:
            for offset in range(0, len(pending), batch_size):
                ids = pending[offset:offset + batch_size]
                processed = list(pool.map(read, ids))
                result = dict.fromkeys(ids)
                valid = [(post_id, item["image"].to(device)) for post_id, item in zip(ids, processed) if item is not None]
                if valid:
                    scores = tagger([image for _, image in valid])
                    prompts = tagger.prompts(scores, random.Random(0), shuffle=False)
                    result.update(zip((post_id for post_id, _ in valid), prompts))
                handle.write(json.dumps(result, ensure_ascii=True) + "\n")
                handle.flush()
                if offset % 256 < batch_size:
                    os.fsync(handle.fileno())
                captions.update(result)
                bar.update(len(ids))
            os.fsync(handle.fileno())

    path = output / "tag-cache.json"
    write_index_json(path, dict(progress, captions=captions))
    print(f"Ready: {path} ({sum(value is not None for value in captions.values()):,} tagged images)", flush=True)
    return path


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="Completed local or staged artist index directory")
    parser.add_argument("--output", type=Path, required=True, help="Resumable cache directory")
    parser.add_argument("--config", type=Path, default=root / "configs/anima_style_colab.json")
    parser.add_argument("--tagger-config", type=Path, help="Tagger-only settings JSON, replacing the config's tagger section")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--tagger", type=Path, help="Existing timm-compatible model directory")
    source.add_argument("--download-tagger", action="store_true", help="Download only the configured tagger")
    parser.add_argument("--tagger-download-dir", type=Path, default=Path("cache/anima-tagger"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    import torch

    from library.moss_image_style_dataset import MossStylePairDataset, resolve_style_dataset_config
    from library.moss_image_style_pretrained import load_prompt_tagger

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; use a CUDA PyTorch environment or --device cpu")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.tagger_config is not None:
        config["tagger"] = json.loads(args.tagger_config.read_text(encoding="utf-8"))
    config["data"].update(artist_index=str(args.dataset.resolve()), index_cache_bytes=0)
    config = resolve_style_dataset_config(config)
    pairs = MossStylePairDataset(**config["data"]).artist_pairs
    if pairs.local_root is None:
        parser.error("Tag caching requires a completed local or staged artist dataset")
    directory = args.tagger
    if args.download_tagger:
        from huggingface_hub import snapshot_download

        manifest = json.loads((root / "configs/anima_colab_models.json").read_text(encoding="utf-8"))
        spec = next(item for item in manifest["models"] if item["name"] == "tagger")
        directory = Path(snapshot_download(config["tagger"].get("repo_id", spec["repo_id"]),
                                           revision=config["tagger"].get("revision", spec["revision"]),
                                           allow_patterns=spec["allow_patterns"], local_dir=args.tagger_download_dir))
    source = {path.name: file_sha256(path) for path in sorted(directory.iterdir())
              if path.is_file() and path.suffix in (".safetensors", ".json", ".csv", ".py")}
    args.output.mkdir(parents=True, exist_ok=True)
    with OutputLock(args.output):
        snapshot = args.output / "tagging-config.json"
        if snapshot.exists() and json.loads(snapshot.read_text(encoding="utf-8")) != config:
            raise ValueError("Saved tagging config differs; use the same settings or another output directory")
        write_index_json(snapshot, config)
        cache_tags(pairs, config["tagger"], args.output, source,
                   lambda: load_prompt_tagger(directory, config["tagger"], device), args.batch_size, device)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped. Rerun the command to reuse completed tag batches.", file=sys.stderr)
        sys.exit(130)
