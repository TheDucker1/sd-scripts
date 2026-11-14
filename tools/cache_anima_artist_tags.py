"""Run only the PixAI tagger over a local artist index and save portable captions."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from download_artist_dataset import OutputLock
from library.danbooru_index import file_sha256, write_index_json


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
    indices = {str(int(row["id"])): index for index, row in enumerate(pairs.rows)}
    if captions.keys() - indices.keys():
        raise ValueError("Tag cache progress contains unknown image IDs")
    pending = [post_id for post_id in indices if post_id not in captions]

    def read(post_id):
        return pairs.process(pairs.fetch(indices[post_id]), random.Random(int(post_id)))

    if pending:
        tagger = load_tagger().requires_grad_(False).eval()
        with ThreadPoolExecutor(max_workers=pairs.fetch_workers) as pool, tqdm(
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
                write_index_json(output / f"part-{len(parts) + offset // batch_size:08d}.json", result)
                captions.update(result)
                bar.update(len(ids))

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
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--tagger", type=Path, help="Existing PixAI model directory")
    source.add_argument("--download-tagger", action="store_true", help="Download only PixAI from the Anima model manifest")
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
        directory = Path(snapshot_download(spec["repo_id"], revision=spec["revision"],
                                           allow_patterns=spec["allow_patterns"], local_dir=args.tagger_download_dir))
    source = {path.name: file_sha256(path) for path in sorted(directory.iterdir())
              if path.is_file() and path.suffix in (".safetensors", ".json", ".csv", ".py")}
    args.output.mkdir(parents=True, exist_ok=True)
    with OutputLock(args.output):
        cache_tags(pairs, config["tagger"], args.output, source,
                   lambda: load_prompt_tagger(directory, config["tagger"], device), args.batch_size, device)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped. Rerun the command to reuse completed tag batches.", file=sys.stderr)
        sys.exit(130)
