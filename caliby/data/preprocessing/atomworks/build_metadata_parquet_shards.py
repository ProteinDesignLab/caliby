#!/usr/bin/env python3
import glob
import itertools
from pathlib import Path

import hydra
import pandas as pd
import yaml
from atomworks.ml.common import generate_example_id
from atomworks.ml.preprocessing.get_pn_unit_data_from_structure import DataPreprocessor
from omegaconf import DictConfig, OmegaConf
from p_tqdm import p_umap
from tqdm import tqdm

from caliby.data.preprocessing.atomworks.sharding_utils import take_shard, use_sharding


@hydra.main(
    config_path="../../../configs/data/preprocessing/atomworks",
    config_name="build_metadata_parquet_shards",
    version_base="1.3.2",
)
def main(cfg: DictConfig):
    """
    Process a set of mmCIFs using AtomWorks.
    This script supports sharding by setting:
      - num_shards (int) and shard_id (int)
    """
    # Create dataset directory + shard dir
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_dir = out_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    # Only one shard writes the canonical config.yaml to avoid races
    if not use_sharding(cfg.shard_id, cfg.num_shards) or (cfg.shard_id == 0):
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        with open(out_dir / "config.yaml", "w") as f:
            yaml.safe_dump(cfg_dict, f)

    # Setup
    use_parallel = cfg.num_workers > 1
    dataset_name = Path(cfg.out_dir).stem

    # Get all CIF paths then take this shard's slice
    cif_paths_all = get_cif_paths(cfg.mmcif_dir, cfg.max_file_size)
    cif_paths = take_shard(cif_paths_all, shard_id=cfg.shard_id, num_shards=cfg.num_shards)
    print(f"Shard {cfg.shard_id}/{cfg.num_shards}: {len(cif_paths)} mmCIFs.")

    # Initialize data preprocessor
    processor = DataPreprocessor(**{**cfg.cif_parser_args, **cfg.data_preprocessor_cfg})

    def _process_cif(cif_path: str):
        """
        Wrapper around the data preprocessor to handle errors.
        """
        try:
            return processor.get_rows(cif_path)
        except Exception as e:
            print(f"Error processing {cif_path}: {e}")
            return []

    ### Process each CIF and save to parquet ###
    if len(cif_paths) == 0:
        print(f"Shard {cfg.shard_id}: no files to process, exiting.")
        return

    if use_parallel:
        df = p_umap(_process_cif, cif_paths, num_cpus=cfg.num_workers, desc=f"Processing mmCIFs (shard {cfg.shard_id})")
    else:
        df = [_process_cif(cif_path) for cif_path in tqdm(cif_paths)]

    df = pd.DataFrame(itertools.chain(*df))  # flatten list of lists

    # If nothing produced, skip writing
    if df.empty:
        print(f"Shard {cfg.shard_id}: produced 0 rows, skipping parquet write.")
        return

    # Write one parquet per shard
    shard_out = shard_dir / f"metadata_shard_{cfg.shard_id:05d}.parquet"
    save_to_parquet(df, dataset_name, cfg.mmcif_dir, str(shard_out))
    print(f"Shard {cfg.shard_id}: wrote {len(df)} rows to {shard_out}")


def get_cif_paths(mmcif_dir: str, max_file_size: int | None = None) -> list[str]:
    cif_paths = glob.glob(f"{mmcif_dir}/**/*.cif", recursive=True)
    if max_file_size is not None:
        original_len = len(cif_paths)
        cif_paths = [path for path in cif_paths if Path(path).stat().st_size <= max_file_size]
        print(f"Excluded {original_len - len(cif_paths)} files due to size.")
    print(f"Found {len(cif_paths)} mmCIFs.")
    return cif_paths


def save_to_parquet(df: pd.DataFrame, dataset_name: str, pdb_in_dir: str, out_path: str):
    """
    Save a dataframe to parquet.
    Also adds an example_id column based on the name of the dataset.
    """
    # Convert all object columns to string to save to parquet
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str)

    # Add example_id based on the name of this dataset
    df["example_id"] = df.apply(
        lambda x: generate_example_id([dataset_name], x["pdb_id"], x["assembly_id"], [x["q_pn_unit_iid"]]), axis=1
    )

    # Add in relative path
    df["rel_path"] = df["path"].apply(lambda x: str(Path(x).relative_to(pdb_in_dir)))

    df.to_parquet(out_path)
    print(f"Saved {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()
