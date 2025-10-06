"""
Make PDB name lists according to various filters.
For boltz_test_cifs, we used the cleaned_cifs directory for compatibility with Protpardelle-1c/Chroma/ProteinMPNN.
"""

import shutil
from pathlib import Path
from typing import Any

import atomworks.enums as aw_enums
import biotite.structure as struct
import hydra
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from omegaconf import DictConfig
from tqdm import tqdm

from caliby.data import const
from caliby.eval.eval_utils.eval_setup_utils import get_pdb_files
from caliby.eval.eval_utils.seq_des_utils import preprocess_pdb


def _get_metadata_from_cif(cif_path: str) -> dict[str, Any]:
    """
    Get metadata from a CIF file.
    """
    # Preprocess the PDB file.
    example = preprocess_pdb(cif_path, None)
    atom_array = example["atom_array"]

    # Get metadata from the atom array.
    is_protein = np.isin(atom_array.chain_type, aw_enums.ChainTypeInfo.PROTEINS)
    is_standard_aa = np.isin(atom_array.res_name, const.STANDARD_AA)
    prot_atom_array = atom_array[is_protein]
    num_prot_res = struct.get_residue_count(prot_atom_array)
    num_prot_chains = struct.get_chain_count(prot_atom_array)

    return {
        "example_id": example["example_id"],
        "num_prot_res": num_prot_res,
        "num_prot_chains": num_prot_chains,
        "is_protein_only": is_protein.all() & is_standard_aa.all(),
    }


@hydra.main(
    config_path="../../../configs/data/preprocessing/atomworks", config_name="make_pdb_name_lists", version_base="1.3.2"
)
def main(cfg: DictConfig):
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
    use_parallel = cfg.num_workers > 1

    # Load in PDB files.
    pdb_files = get_pdb_files(**cfg.input_cfg)

    # Preprocess the PDB files to get metadata.
    if use_parallel:
        parallel = Parallel(n_jobs=cfg.num_workers)
        jobs = [delayed(_get_metadata_from_cif)(pdb_path) for pdb_path in pdb_files]
        metadata = list(parallel(tqdm(jobs, total=len(jobs), desc="Cleaning RCSB mmCIF files")))
    else:
        metadata = []
        for pdb_path in tqdm(pdb_files, total=len(pdb_files), desc="Cleaning RCSB mmCIF files"):
            metadata.append(_get_metadata_from_cif(pdb_path))

    # Save metadata.
    metadata_df = pd.DataFrame(metadata).set_index("example_id")
    metadata_df["pdb_name"] = metadata_df.index + ".cif"
    metadata_df.to_parquet(f"{cfg.out_dir}/metadata.parquet")

    # Apply filters.
    for pdb_name_list in cfg.pdb_name_lists:
        pdb_name_list_df = metadata_df.copy()
        for filter in pdb_name_list["filters"]:
            pdb_name_list_df = pdb_name_list_df.query(filter)
        if cfg.save_subset_cifs:
            subset_cif_path = f"{cfg.out_dir}/subset_cifs/{pdb_name_list['name']}"
            Path(subset_cif_path).mkdir(parents=True, exist_ok=True)
            for pdb_name in pdb_name_list_df["pdb_name"]:
                shutil.copy(f"{cfg.input_cfg.pdb_dir}/{pdb_name}", f"{subset_cif_path}/{pdb_name}")
        pdb_name_list_df["pdb_name"].to_csv(f"{cfg.out_dir}/{pdb_name_list['name']}.txt", index=False, header=False)


if __name__ == "__main__":
    main()
