#!/usr/bin/env python3
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import hydra
import torch
import yaml
from atomworks.ml.datasets.datasets import StructuralDatasetWrapper
from omegaconf import DictConfig, OmegaConf, open_dict
from tqdm import tqdm

from caliby.data.preprocessing.atomworks.sharding_utils import take_shard, use_sharding
from caliby.data.transform.preprocess import preprocess_transform

# tame BLAS threads inside each worker
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


@hydra.main(
    config_path="../../../configs/data/preprocessing/atomworks", config_name="preprocess_examples", version_base="1.3.2"
)
def main(cfg: DictConfig):
    """
    Process a set of mmCIFs using AtomWorks.
    """
    # Create dataset directory
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)

    # Preserve the original config
    if not use_sharding(cfg.shard_id, cfg.num_shards) or (cfg.shard_id == 0):
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        with open(Path(cfg.out_dir, "config.yaml"), "w") as f:
            yaml.safe_dump(cfg_dict, f)

    # Setup
    use_parallel = cfg.num_workers > 1
    dataset_name = Path(cfg.out_dir).stem

    ### Cache features ###
    cached_structure_dir = f"{cfg.out_dir}/cached_structures"
    Path(cached_structure_dir).mkdir(parents=True, exist_ok=True)
    with open_dict(cfg):
        cfg.dataset.cif_parser_args["cache_dir"] = cached_structure_dir
        cfg.dataset.dataset.name = dataset_name

    cached_example_dir = f"{cfg.out_dir}/cached_examples"
    Path(cached_example_dir).mkdir(parents=True, exist_ok=True)
    cache_fn = partial(_cache_examples, cached_example_dir=cached_example_dir)

    # iterate over the dataset, and the caching will happen automatically
    struct_dataset = hydra.utils.instantiate(cfg.dataset, transform=preprocess_transform())
    indices = list(range(len(struct_dataset)))
    indices = take_shard(indices, shard_id=cfg.shard_id, num_shards=cfg.num_shards)

    if use_parallel:
        with ProcessPoolExecutor(
            max_workers=cfg.num_workers,
            mp_context=mp.get_context("forkserver"),
            initializer=_init_dataset,
            initargs=(cfg.dataset,),
        ) as executor:
            for _ in tqdm(executor.map(cache_fn, indices), total=len(indices), desc="Caching examples"):
                pass
    else:
        for idx in tqdm(indices, desc="Caching examples"):
            cache_fn(idx, dataset=struct_dataset)


def _cache_examples(idx: int, cached_example_dir: str, *, dataset: StructuralDatasetWrapper | None = None) -> str:
    try:
        feats = (
            dataset[idx] if dataset is not None else _DATASET[idx]
        )  # indexing the dataset triggers structure caching
    except Exception as e:
        print(f"Error caching example {idx}: {e}")
        return None

    # save feats to disk
    pdb_id = feats["extra_info"]["pdb_id"]
    torch.save(feats, f"{cached_example_dir}/{pdb_id}.pt")
    return pdb_id


# Initialize the dataset in each worker so that the dataset is not pickled
_DATASET: StructuralDatasetWrapper | None = None


def _init_dataset(dataset_cfg: DictConfig):
    global _DATASET
    _DATASET = hydra.utils.instantiate(dataset_cfg, transform=preprocess_transform())


if __name__ == "__main__":
    main()
