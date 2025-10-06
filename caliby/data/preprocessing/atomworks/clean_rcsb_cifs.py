"""
Preprocess mmCIF files from the RCSB using AtomWorks.
This is helpful for e.g. building the first bioassembly for other tools to use (e.g. Protpardelle-1c).
"""

from pathlib import Path

import hydra
from atomworks.io.utils.io_utils import to_cif_string
from joblib import Parallel, delayed
from omegaconf import DictConfig
from tqdm import tqdm

from caliby.eval.eval_utils.eval_setup_utils import get_pdb_files
from caliby.eval.eval_utils.seq_des_utils import preprocess_pdb


@hydra.main(
    config_path="../../../configs/data/preprocessing/atomworks", config_name="clean_rcsb_cifs", version_base="1.3.2"
)
def main(cfg: DictConfig):
    """
    Clean mmCIF files from the RCSB using AtomWorks.
    """
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
    use_parallel = cfg.num_workers > 1

    # Load in PDB files.
    pdb_files = get_pdb_files(**cfg.input_cfg)

    # Clean the PDB files.
    if use_parallel:
        parallel = Parallel(n_jobs=cfg.num_workers)
        jobs = [delayed(_clean_rcsb_cif)(pdb_path, cfg.out_dir) for pdb_path in pdb_files]
        list(parallel(tqdm(jobs, total=len(jobs), desc="Cleaning RCSB mmCIF files")))
    else:
        for pdb_path in tqdm(pdb_files, total=len(pdb_files), desc="Cleaning RCSB mmCIF files"):
            _clean_rcsb_cif(pdb_path, cfg.out_dir)


def _clean_rcsb_cif(pdb_path: str, out_dir: str):
    # Preprocess the PDB file.
    example = preprocess_pdb(pdb_path, None)
    atom_array = example["atom_array"]

    # Map each unique pair of (chain_id, transformation_id) to a sequential label 'A', 'B', ..., 'Z', 'AA', ...
    unique_pairs = []
    for c, t in zip(atom_array.chain_id, atom_array.transformation_id):
        key = (str(c), str(t))
        if key not in unique_pairs:
            unique_pairs.append(key)

    pair_to_label = {pair: _pair_index_to_label(i) for i, pair in enumerate(unique_pairs)}
    new_chain_ids = [pair_to_label[(str(c), str(t))] for c, t in zip(atom_array.chain_id, atom_array.transformation_id)]
    atom_array.chain_id = new_chain_ids

    # Write the PDB file.
    out_file = f"{out_dir}/{Path(pdb_path).stem}.cif"
    with open(out_file, "w") as f:
        f.write(to_cif_string(atom_array, include_nan_coords=False))


def _pair_index_to_label(idx: int) -> str:
    """
    Convert a zero-based index to chain label:
      0 -> 'A', 1 -> 'B', ..., 25 -> 'Z',
      26 -> 'AA', 27 -> 'AB', ...
    This is the usual "excel-style" base-26 without a zero digit.
    """
    if idx < 0:
        raise ValueError("idx must be >= 0")
    letters = []
    n = idx + 1  # convert to 1-based for the math
    while n > 0:
        n, rem = divmod(n - 1, 26)
        letters.append(chr(ord('A') + rem))
    return ''.join(reversed(letters))


if __name__ == "__main__":
    main()
