"""Pack only the final offline artist index and its selected images for Colab."""

import argparse
from pathlib import Path
import sys
import tarfile

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from library.danbooru_index import file_sha256, index_path, load_artist_index
from download_artist_dataset import OutputLock


def pack_dataset(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Archive already exists: {output}; choose a new output filename")
    if output.suffix != ".tar":
        raise ValueError("Use an uncompressed .tar output for RAM staging")
    with OutputLock(source):
        manifest, rows, _ = load_artist_index(source)
        if manifest.get("storage") != "local":
            raise ValueError("Packing requires a completed local image dataset")
        index = index_path(source, manifest["index_file"])
        with np.load(index, allow_pickle=False) as data:
            files, hashes = data["files"], data["sha256"]
        if len(files) != len(rows) or len(hashes) != len(rows) or len(set(files)) != len(files):
            raise ValueError("Local index must contain one unique file and hash per image")
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(".tar.part")
        with tarfile.open(temporary, "w", dereference=True) as archive:
            archive.add(source / "manifest.json", arcname="manifest.json", recursive=False)
            archive.add(index, arcname=index.relative_to(source).as_posix(), recursive=False)
            for row, relative, expected in tqdm(zip(rows, files, hashes), total=len(rows), desc="Pack selected images", unit="image"):
                path = index_path(source, str(relative))
                if (not path.is_file() or path.stat().st_size != int(row["size"])
                        or file_sha256(path).encode("ascii") != expected):
                    raise ValueError(f"Missing/corrupt selected image: {relative}; rerun the downloader")
                archive.add(path, arcname=path.relative_to(source).as_posix(), recursive=False)
        temporary.replace(output)
    print(f"Packed {len(rows):,} selected images: {output} ({output.stat().st_size / 1024**3:.2f} GiB)", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pack_dataset(args.dataset, args.output)


if __name__ == "__main__":
    main()
