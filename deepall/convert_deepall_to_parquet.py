#!/usr/bin/env python3
"""
Convert DeepAll JSONL data to Parquet format for NanoChat training
"""

import json
import os
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq

print("=" * 70)
print("Converting DeepAll JSONL to Parquet")
print("=" * 70)
print()

# Paths (relative to repo root)
REPO_ROOT = Path(__file__).resolve().parents[1]
data_dir = REPO_ROOT / "deepall" / "nanogpt-pytorch-deepall-v1" / "data jsonl"
output_dir = REPO_ROOT / "base_data"
output_dir.mkdir(parents=True, exist_ok=True)

# Step 1: Load all JSONL data
print("Step 1: Loading JSONL data...")
all_texts = []
for jsonl_file in sorted(data_dir.glob("*.jsonl")):
    print(f"  Reading {jsonl_file.name}...")
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                prompt = data.get('prompt', '')
                completion = data.get('completion', '')
                combined = prompt + completion
                if combined.strip():  # Only add non-empty texts
                    all_texts.append(combined)
            except json.JSONDecodeError:
                pass

print(f"✓ Loaded {len(all_texts)} texts")
print()

# Step 2: Create Parquet file
print("Step 2: Creating Parquet file...")
table = pa.table({'text': all_texts})
output_file = output_dir / "shard_00000.parquet"
pq.write_table(table, str(output_file))
print(f"✓ Wrote {len(all_texts)} texts to {output_file}")
print()

# Step 3: Verify
print("Step 3: Verifying...")
pf = pq.ParquetFile(str(output_file))
print(f"  Parquet file has {pf.num_row_groups} row groups")
print(f"  Total rows: {pf.metadata.num_rows}")
print()

print("=" * 70)
print("✓ Conversion complete!")
print("=" * 70)
print()
print(f"Data ready for training at: {output_dir}")
print()
print("Next: Run training with:")
print("  python -m scripts.base_train --depth=4 --max_seq_len=256 --device_batch_size=2 --num_iterations=100")
print()

