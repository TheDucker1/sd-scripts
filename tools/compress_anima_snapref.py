"""Create/reuse an FP32 SVD initializer from SnapRef weights, without a base model."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from tqdm import tqdm

from library.anima_lora_compression import compress_snapref_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Source SnapRef adapter safetensors")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ref-rank", type=int, default=32)
    parser.add_argument("--tgt-rank", type=int, default=32)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if min(args.ref_rank, args.tgt_rank, args.threads) < 1:
        parser.error("Ranks and threads must be positive")
    torch.set_num_threads(args.threads)
    with tqdm(desc="Compress SnapRef", unit="projection") as bar:
        def progress(done, total):
            bar.total = total
            bar.update(done - bar.n)
        receipt, reused = compress_snapref_checkpoint(args.input, args.output, args.ref_rank, args.tgt_rank, progress)
    print(f"{'Verified existing' if reused else 'Saved'} FP32 initializer: {args.output}", flush=True)
    for branch, label in (("ref", "B/reference"), ("tgt", "A/target")):
        modules = [item for item in receipt["modules"] if item["branch"] == branch]
        total = sum(item["total_energy"] for item in modules)
        retained = sum(item["retained_energy"] for item in modules)
        fraction = retained / total if total else 1.0
        print(f"{label}: rank {modules[0]['source_rank']} -> {modules[0]['rank']}; "
              f"retained update energy {fraction:.2%}", flush=True)


if __name__ == "__main__":
    main()
