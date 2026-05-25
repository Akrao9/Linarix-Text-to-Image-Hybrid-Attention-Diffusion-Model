"""Inspect cached Boomer latent records."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from boomer.datasets.latent_shards import LatentCacheDataset, collate_latent_records


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect Boomer latent cache records.")
    parser.add_argument("paths", nargs="+", help="Cache files, directories, or glob patterns.")
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()

    dataset = LatentCacheDataset(args.paths)
    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_latent_records)
    batch = next(iter(loader))

    print(f"records: {len(dataset)}")
    print(f"image_latent: {tuple(batch['image_latent'].shape)} {batch['image_latent'].dtype}")
    print(f"text_embedding: {tuple(batch['text_embedding'].shape)} {batch['text_embedding'].dtype}")
    print(f"attention_mask: {tuple(batch['attention_mask'].shape)} {batch['attention_mask'].dtype}")
    print(f"first_caption: {batch['caption'][0]}")
    print(f"first_path: {batch['path'][0]}")


if __name__ == "__main__":
    main()
