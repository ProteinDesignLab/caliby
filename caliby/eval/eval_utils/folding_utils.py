"""Utilities for running AlphaFold2 structure prediction.

All colabdesign imports are lazy so this module can be imported without AF2 installed.
"""

import gc
import shutil
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from caliby.data.io_utils import cif_to_pdb


def _check_af2_installed():
    """Raise a clear error if colabdesign is not installed."""
    try:
        import colabdesign  # noqa: F401
    except ImportError:
        raise ImportError("colabdesign is required for AF2 refolding. Install with: pip install 'caliby[af2]'")


def clear_mem_torch():
    """Clear memory from torch. Use before running jax models."""
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()


def run_af2(
    pdbs: list[str],
    struct_pred_model: dict[str, Any],
    out_dir: str,
) -> list[str]:
    """
    Predict structures from PDBs with AlphaFold2. Automatically converts .cif files to .pdb files.
    Returns a list of output PDB paths (one per input pdb).
    """
    af_model = struct_pred_model["af_model"]
    model_cfg = struct_pred_model["cfg"]["af2"]

    # Set up output directories.
    unique_id = uuid.uuid4().hex
    temp_dir = f"{out_dir}/temp/{unique_id}"
    Path(temp_dir).mkdir(exist_ok=True, parents=True)

    # Predict structures.
    output_files = []
    for pdb in tqdm(pdbs, desc="Predicting structures with AlphaFold2"):
        # Convert .cif files to .pdb files if needed.
        if Path(pdb).suffix == ".cif":
            cif, pdb = pdb, f"{temp_dir}/{Path(pdb).stem}.pdb"
            cif_to_pdb(cif, pdb)

        output_pdb = f"{out_dir}/af2_{Path(pdb).stem}.pdb"
        af_model.prep_inputs(pdb)

        af_model.restart(mode="wt")  # set to pdb sequence
        af_model.predict(
            num_models=model_cfg["num_models"],
            sample_models=model_cfg["sample_models"],
            num_recycles=model_cfg["num_recycles"],
            verbose=False,
        )

        af_model._save_results(save_best=True, best_metric="plddt", verbose=False)

        _save_best_model(af_model, output_pdb)
        output_files.append(output_pdb)

    # Clean up temp dir.
    shutil.rmtree(temp_dir)

    return output_files


def _save_best_model(af_model, filename: str):
    from colabdesign.af.alphafold.common import protein

    aux = af_model._tmp["best"]["aux"]["all"]
    plddt = np.mean(aux["plddt"], axis=-1)
    best_model_idx = np.argmax(plddt)
    p = {k: aux[k][best_model_idx] for k in ["aatype", "residue_index", "atom_positions", "atom_mask"]}
    p["b_factors"] = 100 * p["atom_mask"] * aux["plddt"][best_model_idx][..., None]

    p_str = protein.to_pdb(protein.Protein(**p))
    p_str = "\n".join(p_str.splitlines()[1:-2])
    p_str = p_str + "\nEND\n"
    with open(filename, 'w') as f:
        f.write(p_str)


def get_struct_pred_model(cfg: DictConfig, device: str) -> dict[str, Any]:
    """
    Get structure prediction model components as a dictionary based on config.

    Example config:
    struct_pred_cfg:
        model_name: "af2"
        af2:
            data_dir: # directory containing "params/" with af2 model params
            num_models: 1
            num_recycles: 3
            use_multimer: false
            ...
    """
    _check_af2_installed()
    from colabdesign.af import mk_af_model
    from colabdesign.shared.utils import clear_mem

    model_name = cfg.model_name
    base_cfg = OmegaConf.load(cfg.base_cfg)
    cfg = OmegaConf.merge(base_cfg, cfg)

    struct_pred_model = {"model_name": model_name, "cfg": cfg, "device": device}
    if model_name == "af2":
        from caliby.weights import ensure_af2_params

        clear_mem_torch()
        clear_mem()
        data_dir = ensure_af2_params(cfg.af2.data_dir)
        af_model = mk_af_model(data_dir=data_dir, use_multimer=cfg.af2.use_multimer)
        struct_pred_model["af_model"] = af_model
    else:
        raise ValueError(f"Invalid model name: {model_name}")

    return struct_pred_model
