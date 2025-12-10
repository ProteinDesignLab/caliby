import glob
import math
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import wandb
from natsort import natsorted


def get_pdb_files(
    pdb_dir: str,
    pdb_name_list: str | None,
    pdb_name_ext: str | None = None,
    n_subsample: int | None = None,
    # slurm array parameters for parallelization
    array_id: int | None = None,
    num_arrays: int | None = None,
    skip_pdb_names: list[str] | None = None,
) -> list[str]:
    """
    Retrieve a list of PDB files from a directory, either by specifying a list of pdb_names or by getting all files.

    Args:
        pdb_dir: Directory containing PDB files
        pdb_name_list: Optional path to a file containing PDB keys (one per line)
        pdb_name_ext: Optional extension to append to each key when pdb_name_list is provided
        array_id: Set by Slurm array job. Null means run all.
        num_arrays: Number of total arrays. If array_id is null, this can remain 1.
        skip_pdb_names: List of PDB names to skip

        # if providing a pdb manifest, set options here
        manifest_kwargs:
            pdb_manifest_csv: Optional path to a CSV file containing PDB keys and other metadata


    Returns:
        List of PDB file paths, naturally sorted if retrieving all files

    Raises:
        ValueError: If no PDB files are found in the directory when pdb_name_list is None
    """
    # Read in PDB files from directory or list of PDB names
    if pdb_name_list is not None:
        # get PDBs with keys in the list
        with open(pdb_name_list, "r") as f:
            pdb_names = f.read().splitlines()
        if pdb_name_ext:
            # replace extension with pdb_name_ext
            pdb_names = [f"{Path(name).with_suffix(pdb_name_ext)}" for name in pdb_names]
        pdb_files = [f"{pdb_dir}/{name}" for name in pdb_names]
        print(f"Found {len(pdb_files)} PDB files from key list")
    else:
        # get all PDBs in the directory
        pdb_files = natsorted(list(glob.glob(f"{pdb_dir}/*")))
        print(f"Found {len(pdb_files)} PDB files in {pdb_dir}")
        if len(pdb_files) == 0:
            raise ValueError(f"No PDB files found in directory {pdb_dir}")

    # Skip existing PDBs
    if skip_pdb_names is not None:
        skip_pdb_names = set(skip_pdb_names)
        pdb_files = [f for f in pdb_files if Path(f).name not in skip_pdb_names]

    # Parallelization: split PDB files into chunks based on array id
    if array_id is not None:
        array_id = array_id
        num_arrays = num_arrays
        chunk_size = math.ceil(len(pdb_files) / num_arrays)

        start_idx = array_id * chunk_size
        end_idx = min(start_idx + chunk_size, len(pdb_files))
        pdb_files = pdb_files[start_idx:end_idx]

    # Optionally take a random subset, preserving order
    if n_subsample is not None:
        n_subsample = min(n_subsample, len(pdb_files))
        chosen_indices = sorted(np.random.choice(len(pdb_files), n_subsample, replace=False))
        pdb_files = [pdb_files[i] for i in chosen_indices]

    print(f"Using {len(pdb_files)} PDB files")

    return pdb_files


def get_training_checkpoints(
    denoiser_train_dir: str,
    model_type: str,
    eval_every_n_ckpts: int = 1,
    start_step: int | None = None,
    end_step: int | None = None,
) -> list[str]:
    """
    Get model checkpoints from a training directory, preferring EMA checkpoints if available.

    Args:
        denoiser_train_dir: Path to the denoiser training directory
        model_type: Either "atom_denoiser" or "seq_denoiser"
        eval_every_n_ckpts: Only evaluate every nth checkpoint
        start_step: Optional starting step to filter checkpoints (skip checkpoints before this step)
        end_step: Optional ending step to filter checkpoints (skip checkpoints after this step)

    Returns:
        List of checkpoint paths, sorted by step/epoch
    """
    # Map model type to checkpoint prefix
    prefix_map = {"atom_denoiser": "ad", "seq_denoiser": "sd"}
    prefix = prefix_map.get(model_type)
    if prefix is None:
        raise ValueError(f"Invalid model_type: {model_type}. Must be 'atom_denoiser' or 'seq_denoiser'")

    # Check for EMA checkpoints
    ema_ckpt_dir = f"{denoiser_train_dir}/checkpoints/ema"
    if Path(ema_ckpt_dir).exists():
        # Use EMA checkpoints if they exist
        print(f"Using EMA checkpoints from {ema_ckpt_dir}")
        pattern = re.compile(f"{prefix}-step(\\d+)-epoch(\\d+)-ema(\\d+\\.\\d+)\\.ckpt$")
        ckpts = glob.glob(f"{ema_ckpt_dir}/*.ckpt")
    else:
        print(f"Using non-EMA checkpoints from {denoiser_train_dir}/checkpoints")
        pattern = re.compile(f"{prefix}-step(\\d+)-epoch(\\d+)\\.ckpt$")
        ckpts = glob.glob(f"{denoiser_train_dir}/checkpoints/*.ckpt")

    # Filter and sort checkpoints
    ckpts = natsorted([ckpt for ckpt in ckpts if pattern.search(Path(ckpt).name)])[::eval_every_n_ckpts]

    # Filter by start_step and end_step if provided
    if start_step is not None or end_step is not None:
        filtered_ckpts = []
        for ckpt in ckpts:
            match = pattern.search(Path(ckpt).name)
            if match:
                global_step = int(match.group(1))
                if (start_step is None or global_step >= start_step) and (end_step is None or global_step <= end_step):
                    filtered_ckpts.append(ckpt)
            else:
                raise ValueError(f"Unexpected checkpoint filename: {Path(ckpt).name}")
        ckpts = filtered_ckpts

    return ckpts, pattern


def wandb_setup(
    base_out_dir: str,
    no_wandb: bool,
    project: str | None,
    wandb_id: str | None,
    group: str | None,
    exp_name: str | None,
    cfg_dict: dict = None,
) -> str:
    """
    Set up Weights & Biases (wandb) tracking and return the log directory.
    Log directory is set to base_out_dir/exp_name.

    Args:
        no_wandb: If True, disable wandb logging
        project: wandb project name
        wandb_id: wandb entity ID
        group: Group name for the experiment
        exp_name: Name of the experiment
        base_out_dir: Base output directory for logs
        cfg_dict: Configuration dictionary to log

    Returns:
        Path: Log directory path
    """
    if exp_name is None:
        exp_name = "debug"

    # Set up log directory
    log_dir = str(Path(base_out_dir, exp_name))
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    # Initialize wandb
    if not no_wandb:
        # Create wandb dir
        wandb_dir = str(Path(base_out_dir, "wandb"))
        Path(wandb_dir).mkdir(parents=True, exist_ok=True)

        # Set wandb cache directory
        wandb_cache_dir = str(Path(base_out_dir, "cache", "wandb"))
        os.environ["WANDB_CACHE_DIR"] = wandb_cache_dir

        wandb.init(
            project=project,
            entity=wandb_id,
            group=group,
            name=exp_name,
            config=cfg_dict,
            dir=wandb_dir,
        )

    return log_dir


def get_conformer_dirs(
    conformer_dir: str,
    pdb_name_list: str | list[str] | None,
    # slurm array parameters for parallelization
    array_id: int | None,
    num_arrays: int | None,
    # other options
    skip_pdb_names: list[str] | None = None,
) -> list[str]:
    """
    Get a list of conformer directories from a directory, either by specifying a list of pdb_names or by getting all files.

    pdb_name_list can be a list of pdb names or a path to a file containing a list of pdb names.
    """
    if pdb_name_list is not None:
        # get conformer directories corresponding to pdb_names in the list
        if isinstance(pdb_name_list, str):
            with open(pdb_name_list, "r") as f:
                pdb_names = f.read().splitlines()
        else:
            pdb_names = pdb_name_list
        conformer_dirs = [f"{conformer_dir}/{Path(pdb_name).stem}" for pdb_name in pdb_names]
    else:
        # get all directories in the conformer_dir
        conformer_dirs = natsorted(list(glob.glob(f"{conformer_dir}/*")))
        conformer_dirs = [conformer_dir for conformer_dir in conformer_dirs if Path(conformer_dir).is_dir()]

    # Skip PDBs
    if skip_pdb_names is not None:
        skip_pdb_names = set(skip_pdb_names)
        conformer_dirs = [
            conformer_dir for conformer_dir in conformer_dirs if Path(conformer_dir).name not in skip_pdb_names
        ]

    # Parallelization: split PDB files into chunks based on array id
    if array_id is not None:
        array_id = array_id
        num_arrays = num_arrays
        chunk_size = math.ceil(len(conformer_dirs) / num_arrays)

        start_idx = array_id * chunk_size
        end_idx = min(start_idx + chunk_size, len(conformer_dirs))
        conformer_dirs = conformer_dirs[start_idx:end_idx]

    print(f"Using {len(conformer_dirs)} conformer directories")
    return conformer_dirs


def process_conformer_dirs(
    conformer_dirs: list[str],
    *,
    max_num_conformers: int | None,
    include_primary_conformer: bool,
    ignore_missing_primary_conformer: bool = False,
) -> dict[str, list[str]]:
    """
    Process PDB/CIF structures in all conformer directories.

    If max_num_conformers is None, we will include all conformers found in the conformer directories.
    If include_primary_conformer is True, we will also include the primary conformer, which must share the same PDB name as the conformer directory (either .pdb or .cif).

    For each conformer directory, we will grab all PDB/CIF files, natsort them, and take until we have max_num_conformers files (including the primary conformer if include_primary_conformer is True).

    Returns:
        List of processed structure files, one per conformer directory.
    """
    # First, collect a list of conformers for each PDB
    pdb_to_conformer_list = defaultdict(list)  # maps from a given pdb name to its conformer structure files
    for conformer_dir in conformer_dirs:
        pdb_name = Path(conformer_dir).name
        all_conformers = natsorted(glob.glob(f"{conformer_dir}/*.pdb") + glob.glob(f"{conformer_dir}/*.cif"))

        if max_num_conformers is None:
            max_num_conformers = len(all_conformers)

        # Try to find primary conformer with either .cif or .pdb extension
        primary_conformer_cif = f"{conformer_dir}/{pdb_name}.cif"
        primary_conformer_pdb = f"{conformer_dir}/{pdb_name}.pdb"

        if Path(primary_conformer_cif).exists():
            primary_conformer = primary_conformer_cif
        elif Path(primary_conformer_pdb).exists():
            primary_conformer = primary_conformer_pdb
        elif ignore_missing_primary_conformer:
            print(
                f"Warning: Primary conformer not found for {pdb_name}, defaulting to first conformer in natsorted list {all_conformers[0]}"
            )
            primary_conformer = None
        else:
            raise FileNotFoundError(
                f"Primary conformer not found for {pdb_name}. Expected either {primary_conformer_cif} or {primary_conformer_pdb}"
            )

        if primary_conformer is not None:
            all_conformers.remove(primary_conformer)

        # Then, take the first max_num_conformers conformers (including the primary conformer if include_primary_conformer is True)
        if include_primary_conformer and primary_conformer is not None:
            conformers = [primary_conformer] + all_conformers[: max_num_conformers - 1]
        else:
            conformers = all_conformers[:max_num_conformers]
        pdb_to_conformer_list[pdb_name].extend(conformers)

    return pdb_to_conformer_list


def get_ensemble_constraint_df(
    pos_constraint_df: pd.DataFrame,
    pdb_to_processed_conformers: dict[str, list[str]],
) -> pd.DataFrame:
    """
    Expand a pos_constraint_df to include all conformers
    """
    # expand pdb_key to all conformers
    conformer_dfs = []
    for pdb_key in pos_constraint_df["pdb_key"].unique():
        if pdb_key not in pdb_to_processed_conformers:
            continue
        conformer_df = pos_constraint_df[pos_constraint_df["pdb_key"] == pdb_key]
        conformer_df = pd.concat([conformer_df] * len(pdb_to_processed_conformers[pdb_key]), ignore_index=True)
        conformer_df["pdb_key"] = [Path(x).stem for x in pdb_to_processed_conformers[pdb_key]]
        conformer_dfs.append(conformer_df)
    pos_constraint_df = pd.concat(conformer_dfs)
    return pos_constraint_df
