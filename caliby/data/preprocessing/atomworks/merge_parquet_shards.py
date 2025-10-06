#!/usr/bin/env python3
import argparse
import glob
from pathlib import Path

import pandas as pd


def main(out_dir: str):
    out = Path(out_dir)
    shard_files = sorted(glob.glob(str(out / "shards" / "metadata_shard_*.parquet")))
    if not shard_files:
        raise SystemExit(f"No shard files found in {out/'shards'}")
    print(f"Found {len(shard_files)} shard files, concatenating...")
    dfs = [pd.read_parquet(p) for p in shard_files]
    df = pd.concat(dfs, ignore_index=True)

    # Write combined metadata
    meta_path = out / "metadata.parquet"
    df.to_parquet(meta_path)
    print(f"Wrote {len(df)} rows to {meta_path}")

    # Unique-by-pdb_id parquet for caching examples in downstream steps
    df_cache = df.groupby("pdb_id", as_index=False).first()
    cache_path = out / "metadata_for_caching.parquet"
    df_cache.to_parquet(cache_path)
    print(f"Wrote {len(df_cache)} rows to {cache_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    main(args.out_dir)
