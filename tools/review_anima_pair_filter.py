"""Render a bounded, diverse active-review set from offline pair classifier scores."""

from collections import Counter
import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from library.danbooru_index import index_path, load_artist_index, write_index_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--scan", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--tags", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=72)
    parser.add_argument("--seed", type=int, default=24)
    parser.add_argument("--verification", action="store_true", help="Sample score bands on wholly unreviewed artists")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest, rows, artists = load_artist_index(args.index)
    if manifest.get("storage") != "local":
        parser.error("This review renderer reads the downloaded local image directory")
    with np.load(index_path(args.index, manifest["index_file"]), allow_pickle=False) as data:
        files = data["files"]
    row_lookup = {int(row["id"]): i for i, row in enumerate(rows)}
    with np.load(args.scan / "all_pairs.npz", allow_pickle=False) as data:
        ids, groups, features = data["ids"], data["artists"], data["features"]
        feature_names = list(data["feature_names"])
    with np.load(args.scan / "scores.npz", allow_pickle=False) as data:
        scores, novelty = data["scores"], data["outlier_scores"]
        if not np.array_equal(data["ids"], ids):
            raise ValueError("Pair scores do not match this feature table")
    reviewed = set()
    with args.labels.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            reviewed.add(tuple(sorted((int(row["id_a"]), int(row["id_b"])))))
    captions = json.loads(args.tags.read_text())["captions"]
    reviewed_artists = {int(rows[row_lookup[pair[0]]]["artist"]) for pair in reviewed}
    exclude = {"nude", "nipples", "pussy", "penis", "sex", "intercourse", "fellatio", "cunnilingus",
               "penetration", "cum", "anus", "spread legs", "loli", "shota"}
    safe = {}
    def reviewable(post_id):
        if post_id not in safe:
            tags = set((captions.get(str(post_id)) or "").lower().replace("_", " ").split(", "))
            safe[post_id] = not bool(tags & exclude)
        return safe[post_id]
    rng = np.random.default_rng(args.seed)
    selected, used, artist_counts = [], set(), Counter()
    def take(order, count, source):
        added = 0
        for i in order:
            i = int(i)
            key = tuple(sorted(map(int, ids[i])))
            if i in used or key in reviewed or artist_counts[int(groups[i])] >= (1 if args.verification else 2):
                continue
            if args.verification and int(groups[i]) in reviewed_artists:
                continue
            if not all(reviewable(int(post_id)) for post_id in ids[i]):
                continue
            selected.append((i, source))
            used.add(i)
            artist_counts[int(groups[i])] += 1
            added += 1
            if added == count:
                break
    if args.verification:
        take(rng.permutation(np.flatnonzero(scores >= .95)), args.count // 2, "verify_high")
        take(rng.permutation(np.flatnonzero((scores >= .5) & (scores < .95))), args.count // 4, "verify_borderline")
        take(rng.permutation(np.flatnonzero(scores < .2)), args.count - len(selected), "verify_low")
        rng.shuffle(selected)
    else:
        take(np.argsort(-scores), args.count // 3, "high_score")
        take(np.argsort(np.abs(scores - .5)), args.count // 3, "uncertain")
        structure = features[:, feature_names.index("edge_cosine")]
        outliers = np.flatnonzero(structure > .88)
        take(outliers[np.argsort(-novelty[outliers])], args.count // 6, "outlier")
        take(rng.permutation(len(ids)), args.count - len(selected), "random_control")
    if len(selected) != args.count:
        raise ValueError(f"Only {len(selected)} eligible review pairs; reduce --count")
    records = []
    for number, (i, source) in enumerate(selected, 1):
        records.append(dict(number=number, id_a=int(ids[i, 0]), id_b=int(ids[i, 1]), artist=str(artists[groups[i]]),
                            score=float(scores[i]), outlier_score=float(novelty[i]), mse=float(features[i, 0]),
                            label="", source=source, note=""))
    with (args.output / "review.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    write_index_json(args.output / "selection.json", records)
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 15)
    for start in range(0, len(records), 6):
        page = records[start:start + 6]
        board = Image.new("RGB", (1080, 960), "#eeeeee")
        draw = ImageDraw.Draw(board)
        for position, pair in enumerate(page):
            x, y = (position % 2) * 540, (position // 2) * 320
            # Keep model scores off the sheet so the visual labels are blind.
            title = f"{pair['number']:02d} {pair['artist'][:40]}"
            draw.text((x + 7, y + 6), title, fill="black", font=font)
            for column, key in enumerate(("id_a", "id_b")):
                source = index_path(args.index, str(files[row_lookup[pair[key]]]))
                with Image.open(source) as original:
                    original.draft("RGB", (540, 540))
                    image = ImageOps.exif_transpose(original).convert("RGBA")
                    image.thumbnail((260, 260), Image.Resampling.LANCZOS)
                    background = Image.new("RGBA", image.size, "white")
                    image = Image.alpha_composite(background, image).convert("RGB")
                left = x + column * 270 + (270 - image.width) // 2
                board.paste(image, (left, y + 30 + (260 - image.height) // 2))
                draw.text((x + column * 270 + 7, y + 296), str(pair[key]), fill="black", font=font)
        board.save(args.output / f"page-{start // 6 + 1:02d}.jpg", quality=92)
    print(f"Rendered {len(records)} pairs across {len(artist_counts)} artists in {args.output}", flush=True)


if __name__ == "__main__":
    main()
