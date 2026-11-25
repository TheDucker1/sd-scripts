"""Download one indexed HF tar, bucket/resize to JPEG, commit, delete original.

Archive-level resume only. No models/GPU and no individual-image resume journals.
Defaults target this machine's existing training environment and full artist index.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import httpx
from tqdm import tqdm

from library.danbooru_index import load_artist_index, write_index_json
from tools.download_artist_dataset import OutputLock, prepare_source


def hf_download(manifest, archive, directory, retries=5):
    # Imported only after main configures the SDK/Xet environment.
    from huggingface_hub import get_token, hf_hub_download
    from huggingface_hub.errors import HfHubHTTPError

    token = get_token()
    if not token:
        raise RuntimeError("No Hugging Face login found. Run `hf auth login` or set HF_TOKEN before downloading")
    for attempt in range(retries + 1):
        try:
            result = Path(hf_hub_download(repo_id=manifest["repo_id"], filename=archive["path"], repo_type="dataset",
                          revision=manifest["revision"], local_dir=directory, token=token))
            if result.is_symlink() or not result.resolve().is_relative_to(Path(directory).resolve()):
                raise ValueError("SDK returned a shared-cache path instead of an owned local download")
            if result.stat().st_size != archive["size"]:
                raise ValueError("Downloaded archive size differs from the pinned index")
            return result
        except HfHubHTTPError as error:
            status = error.response.status_code
            if status not in (408, 429, 500, 502, 503, 504) or attempt == retries:
                raise RuntimeError(f"Hugging Face download failed (HTTP {status}); staged files retained") from None
            retry_after = error.response.headers.get("Retry-After", "")
            pause = max(min(60, 2**attempt), float(retry_after) if retry_after.isdigit() else 0)
        except (OSError, RuntimeError, httpx.TransportError) as error:
            if attempt == retries:
                raise RuntimeError(f"Archive download failed ({type(error).__name__}); staged files retained") from None
            pause = min(60, 2**attempt)
        tqdm.write(f"Retrying archive download in {pause:g}s ({attempt + 1}/{retries})")
        time.sleep(pause)


def run(args, downloader=hf_download):
    from library.artist_shard_resize import FORMAT, committed_shard, finalize_dataset, make_geometry, process_archive, remove_work_directory

    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with OutputLock(root):
        if (root / "manifest.json").exists() and not (root / "resize-plan.json").exists():
            raise FileExistsError("Output already contains another dataset; choose a new output directory")
        source = prepare_source(args.index, root, args.index_subdir)
        if source.resolve() == root:
            raise ValueError("Output must differ from the source index directory")
        manifest, rows, artists = load_artist_index(source)
        if manifest.get("storage") in ("local", "local_tar", "local_shards"):
            raise ValueError("Use the full remote tar index, not a previously downloaded local subset")
        if len(np.unique(rows["id"])) != len(rows):
            raise ValueError("Source index contains duplicate post IDs; use the completed deduplicated index")
        settings = dict(pixels=args.pixels, min_bucket_reso=args.min_bucket_reso, max_bucket_reso=args.max_bucket_reso,
                        bucket_reso_steps=args.bucket_reso_steps, max_aspect_ratio=args.max_aspect_ratio,
                        quality=args.quality, max_member_bytes=args.max_member_mib * 1024**2,
                        max_image_pixels=args.max_image_pixels, interpolation="area", alpha_background="white",
                        subsampling=0, no_upscale=True)
        make_geometry(settings)  # Reject invalid bucket settings before any download.
        identity = dict(format=FORMAT, source_index_sha256=manifest["index_sha256"],
                        repo_id=manifest["repo_id"], revision=manifest["revision"], resize=settings)
        plan = root / "resize-plan.json"
        if plan.exists() and json.loads(plan.read_text(encoding="utf-8")) != identity:
            raise ValueError("Output belongs to different source/resize settings; choose another output directory")
        if not plan.exists():
            write_index_json(plan, identity)
        order = np.argsort(rows["archive"], kind="stable")
        sorted_ids = rows["archive"][order]
        groups = np.r_[0, np.flatnonzero(np.diff(sorted_ids)) + 1, len(rows)]
        jobs = [(int(sorted_ids[start]), order[start:end]) for start, end in zip(groups[:-1], groups[1:])]
        receipts = {}
        for archive_id, _ in tqdm(jobs, desc="Check completed tars", unit="tar"):
            archive = manifest["archives"][archive_id]
            relative = PurePosixPath(archive["path"])
            if relative.is_absolute() or ".." in relative.parts or "\\" in archive["path"] or relative.suffix != ".tar":
                raise ValueError("Expected safe pinned uncompressed .tar archive paths")
            receipt = committed_shard(root, archive_id, identity, archive, args.verify_shards)
            if receipt:
                receipts[archive_id] = receipt
        pending_bytes = sum(manifest["archives"][index]["size"] for index, _ in jobs if index not in receipts)
        print(f"Indexed images: {len(rows):,}; tars: {len(jobs):,}; completed: {len(receipts):,}; "
              f"remaining source download: {pending_bytes / 1024**4:.3f} TiB", flush=True)
        print("Only indexed single-artist images are kept. One full original tar is staged at a time.", flush=True)
        if args.dry_run:
            return
        reserve = int(args.reserve_disk_gib * 1024**3)
        done_here = 0
        source_bytes = sum(manifest["archives"][key]["size"] for key, _ in jobs)
        def progress(phase, current=None, trainable_images=None):
            output_bytes = sum(item["bytes"] for item in receipts.values())
            count = sum(item["count"] for item in receipts.values())
            skipped = sum(item["skipped"] for item in receipts.values())
            estimate = output_bytes / count * len(rows) if count else None
            write_index_json(root / "progress.json", dict(phase=phase, current_archive=current,
                archives_done=len(receipts), archives_total=len(jobs), archives_remaining=len(jobs) - len(receipts),
                images_saved=count, images_skipped=skipped, images_processed=count + skipped,
                images_remaining=len(rows) - count - skipped, trainable_images=trainable_images, indexed_images_total=len(rows),
                source_archive_bytes_total=source_bytes,
                source_archive_bytes_pending=sum(manifest["archives"][key]["size"] for key, _ in jobs if key not in receipts),
                compact_bytes=output_bytes, estimated_final_compact_bytes=estimate,
                updated_at=datetime.now(timezone.utc).isoformat()))
        try:
            for archive_id, indices in tqdm(jobs, desc="Archives", unit="tar"):
                archive = manifest["archives"][archive_id]
                download_dir = root / ".download" / f"{archive_id:05d}"
                if archive_id in receipts:
                    remove_work_directory(download_dir, root)
                    continue
                if args.finalize_only or (args.max_archives is not None and done_here >= args.max_archives):
                    break
                # Leave room for the whole source tar plus the user's reserve.
                local = download_dir.joinpath(*PurePosixPath(archive["path"]).parts)
                have = local.stat().st_size if local.is_file() else sum(
                    path.stat().st_size for path in download_dir.rglob("*.incomplete"))
                needed = max(0, archive["size"] - have)
                if shutil.disk_usage(root).free < needed + reserve:
                    raise OSError(f"Not enough free disk for {archive['path']} plus --reserve-disk-gib; no new download started")
                download_dir.mkdir(parents=True, exist_ok=True)
                progress("download", archive["path"])
                tqdm.write(f"Downloading {archive['path']} ({archive['size'] / 1024**3:.2f} GiB) using Hugging Face")
                path = downloader(manifest, archive, download_dir, args.retries)
                progress("resize", archive["path"])
                receipt = process_archive(root, archive_id, archive, path, rows[indices], identity, args.workers, reserve)
                receipts[archive_id] = receipt
                progress("committed", archive["path"])
                remove_work_directory(download_dir, root)
                done_here += 1
                tqdm.write(f"Committed {receipt['count']:,} images; skipped {receipt['skipped']:,}; "
                           f"compact tar {receipt['bytes'] / 1024**2:.1f} MiB; staged original removed")
        finally:
            count = finalize_dataset(root, manifest, artists, receipts, identity, len(receipts) == len(jobs))
            progress("complete" if len(receipts) == len(jobs) else "paused", trainable_images=count)
            print(f"Trainable snapshot: {count:,} images with same-artist/date partners; completed tars retained.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=ROOT / "dataset-index-20261107T072410Z-1-001.zip")
    parser.add_argument("--index-subdir")
    parser.add_argument("--output", type=Path, default=ROOT / "danbooru-artists-512")
    parser.add_argument("--pixels", type=int, default=512, help="Square root of total pixel budget, not a square crop")
    parser.add_argument("--min-bucket-reso", type=int, default=256)
    parser.add_argument("--max-bucket-reso", type=int, default=1536)
    parser.add_argument("--bucket-reso-steps", type=int, default=64)
    parser.add_argument("--max-aspect-ratio", type=float, default=3.0)
    parser.add_argument("--quality", type=int, default=95)
    parser.add_argument("--workers", type=int, choices=range(1, 9), default=2)
    parser.add_argument("--max-member-mib", type=int, default=128)
    parser.add_argument("--max-image-pixels", type=int, default=64_000_000)
    parser.add_argument("--reserve-disk-gib", type=float, default=4)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--max-archives", type=int, help="Process at most this many additional tars, then publish a snapshot")
    parser.add_argument("--verify-shards", action="store_true", help="Also hash every completed compact tar")
    parser.add_argument("--finalize-only", action="store_true", help="Rebuild training index from completed output tars, without downloads")
    parser.add_argument("--dry-run", action="store_true", help="Show plan; no auth, downloads, processing or scratch deletion")
    args = parser.parse_args()
    if (args.pixels < 1 or not 1 <= args.quality <= 100 or args.retries < 0 or args.max_member_mib < 1
            or args.max_image_pixels < 1 or args.reserve_disk_gib < 0 or not np.isfinite(args.reserve_disk_gib)
            or (args.max_archives is not None and args.max_archives < 1)):
        parser.error("Invalid size, quality, retries, reserve or archive limit")
    # Must precede imports of repository training utilities/Hub SDK.
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    os.environ.setdefault("HF_XET_CACHE", str(args.output.resolve() / ".hf-xet"))
    os.environ.setdefault("HF_XET_CHUNK_CACHE_SIZE_BYTES", "0")  # Also avoids chunk caches with older hf_xet.
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
    try:
        run(args)
    except KeyboardInterrupt:
        print("Stopped. Completed tars are safe; the unfinished tar is processed again next time.", flush=True)
        raise SystemExit(130) from None


if __name__ == "__main__":
    main()
