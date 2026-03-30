"""Evaluation metrics for sidechain packing and self-consistency evaluation."""

import os
import subprocess
import urllib.request
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import biotite.structure as struc
import numpy as np
import torch
from atomworks.io.utils.io_utils import to_cif_string
from atomworks.ml.utils.token import apply_token_wise
from joblib import Parallel, delayed
from torchtyping import TensorType

import caliby.data.const as const
from caliby.data import data
from caliby.eval.eval_utils.seq_des_utils import get_sd_example

# ---------------------------------------------------------------------------
# Self-consistency evaluation
# ---------------------------------------------------------------------------


def run_self_consistency_eval(
    pdbs: list[str], struct_pred_model: dict[str, Any], out_dir: str
) -> dict[str, dict[str, TensorType]]:
    """
    Run self-consistency evaluation on a list of PDBs with designed sequences using the specified model.
    """
    if struct_pred_model["model_name"] == "af2":
        return run_self_consistency_eval_af2(pdbs, struct_pred_model, out_dir)
    else:
        raise ValueError(f"Unsupported model: {struct_pred_model['model_name']}")


def run_self_consistency_eval_af2(
    pdbs: list[str], struct_pred_model: dict[str, Any], out_dir: str
) -> dict[str, dict[str, TensorType]]:
    """
    Run self-consistency evaluation on a list of PDBs with designed sequences using AlphaFold2.
    """
    from caliby.eval.eval_utils.folding_utils import run_af2

    pred_dir = f"{out_dir}/struct_preds"
    pred_pdbs = run_af2(pdbs, struct_pred_model, pred_dir)
    id_to_metrics = {}
    for pred_pdb, design_pdb in zip(pred_pdbs, pdbs):
        example_id = Path(design_pdb).stem
        id_to_metrics[example_id] = compute_self_consistency_metrics(
            pred_pdb=pred_pdb, design_pdb=design_pdb, out_dir=f"{out_dir}/ca_aligned_struct_preds"
        )
    return id_to_metrics


def compute_self_consistency_metrics(
    *,
    pred_pdb: str,
    design_pdb: str,
    out_dir: str,
) -> dict[str, float]:
    """
    Compute self-consistency metrics between a designed structure and its predicted structure.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    metrics = {}

    # Load in each structure.
    pred_example = get_sd_example(pred_pdb, data_cfg=None)
    design_example = get_sd_example(design_pdb, data_cfg=None)

    # Align on CA atoms.
    pred_coords, design_coords = pred_example["coords"], design_example["coords"]  # [N, 3]

    # First, extract CA atoms from both designed and predicted structures.
    design_ca_atom_mask = torch.tensor(design_example["atom_array"].atom_name == "CA")
    pred_ca_atom_mask = torch.tensor(pred_example["atom_array"].atom_name == "CA")

    # Only compute RMSD over resolved CA atoms in the design.
    design_ca_resolved_mask = design_example["atom_resolved_mask"] * design_ca_atom_mask

    # Compute CA RMSD.
    ca_rmsd, aux = data.torch_rmsd_weighted(
        pred_coords[pred_ca_atom_mask].unsqueeze(0),
        design_coords[design_ca_atom_mask].unsqueeze(0),
        design_ca_resolved_mask[design_ca_atom_mask].unsqueeze(0),
        return_aux=True,
    )

    # CA-align predicted coordinates to designed coordinates.
    R, t = aux["transforms"]
    ca_aligned_pred_coords = pred_coords @ R + t

    # Write aligned coords to mmcif.
    pred_example["atom_array"].coord = ca_aligned_pred_coords.squeeze(0).numpy()
    with open(f"{out_dir}/{Path(pred_pdb).stem}.cif", "w") as f:
        f.write(to_cif_string(pred_example["atom_array"]))

    # Compute metrics.
    for metric in ["sc_ca_rmsd", "avg_ca_plddt", "tmalign_score"]:
        if metric == "sc_ca_rmsd":
            metrics[metric] = ca_rmsd.item()

        elif metric == "avg_ca_plddt":
            ca_plddts = pred_example["atom_array"][pred_ca_atom_mask].b_factor
            avg_ca_plddt = ca_plddts.mean().item()
            metrics[metric] = avg_ca_plddt

        elif metric == "tmalign_score":
            tmalign_score, _ = _compute_tmalign_score(pred_pdb, design_pdb)
            metrics[metric] = tmalign_score

    return metrics


def _ensure_tmalign() -> str:
    """Return path to TMalign binary, downloading and compiling from source if needed."""
    tmalign_dir = os.path.join(os.environ["MODEL_PARAMS_DIR"], "software", "tmalign")
    binary = os.path.join(tmalign_dir, "TMalign")
    if os.path.isfile(binary):
        return binary

    print("TMalign not found. Downloading and compiling from source...")
    os.makedirs(tmalign_dir, exist_ok=True)
    src = os.path.join(tmalign_dir, "TMalign.cpp")
    subprocess.run(["wget", "-O", src, "https://zhanggroup.org/TM-align/TMalign.cpp"], check=True)
    subprocess.run(["g++", "-O3", "-ffast-math", "-lm", "-o", binary, src], check=True)
    print(f"TMalign compiled successfully at {binary}")
    return binary


def _compute_tmalign_score(pdb_1: str, pdb_2: str) -> tuple[float, float]:
    """
    Compute TM-score between two PDBs using TM-align.

    Returns:
        - tmalign_score_1: TM-align score normalized by length of pdb_1
        - tmalign_score_2: TM-align score normalized by length of pdb_2
    """
    try:
        tmalign_bin = _ensure_tmalign()
        cmd = [tmalign_bin, pdb_1, pdb_2]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        output = result.stdout

        tmalign_score_1 = None
        tmalign_score_2 = None

        for line in output.splitlines():
            if line.startswith("TM-score="):
                parts = line.strip().split()
                score = float(parts[1])
                if "Chain_1" in line:
                    tmalign_score_1 = score
                elif "Chain_2" in line:
                    tmalign_score_2 = score

        tmalign_score_1 = tmalign_score_1 if tmalign_score_1 is not None else np.nan
        tmalign_score_2 = tmalign_score_2 if tmalign_score_2 is not None else np.nan
    except Exception as e:
        print(f"Error computing TM-align score: {e}")
        tmalign_score_1 = np.nan
        tmalign_score_2 = np.nan

    return tmalign_score_1, tmalign_score_2


# ---------------------------------------------------------------------------
# Sidechain packing metrics
# ---------------------------------------------------------------------------


def run_packing_metrics_eval(
    *,
    input_pdbs: list[str],
    pred_pdbs: list[str],
    out_dir: str,
    num_workers: int = 1,
) -> dict[str, dict[str, float]]:
    """
    Run sidechain packing metrics evaluation on a list of input PDBs and predicted PDBs.
    """
    example_ids = [Path(input_pdb).stem for input_pdb in input_pdbs]
    parallel_context = Parallel(n_jobs=num_workers) if num_workers > 1 else nullcontext()

    aligned_inputs_dir = f"{out_dir}/aligned_inputs"
    Path(aligned_inputs_dir).mkdir(parents=True, exist_ok=True)

    with parallel_context as parallel_pool:
        if parallel_pool is None:
            id_to_metrics = {}
            for example_id, input_pdb, pred_pdb in zip(example_ids, input_pdbs, pred_pdbs):
                id_to_metrics[example_id] = compute_packing_metrics(
                    pdb1=input_pdb, pdb2=pred_pdb, out_dir=aligned_inputs_dir
                )
        else:
            results = parallel_pool(
                delayed(compute_packing_metrics)(pdb1=input_pdb, pdb2=pred_pdb, out_dir=aligned_inputs_dir)
                for input_pdb, pred_pdb in zip(input_pdbs, pred_pdbs)
            )
            id_to_metrics = {example_id: metrics for example_id, metrics in zip(example_ids, results)}

    return id_to_metrics


def compute_packing_metrics(*, pdb1: str, pdb2: str, out_dir: str) -> dict[str, float]:
    """
    Compute sidechain packing metrics between two PDBs.
    """
    metrics = {}

    # Load in each structure.
    example1 = get_sd_example(pdb1, data_cfg=None)
    example2 = get_sd_example(pdb2, data_cfg=None)
    atom_resolved_mask = example1["atom_resolved_mask"] * example2["atom_resolved_mask"]

    # Compute standard protein atom mask.
    standard_prot_mask = example1["is_protein"] & ~example1["is_atomized"]
    atomwise_standard_prot_mask = (
        torch.gather(standard_prot_mask, dim=-1, index=example1["atom_to_token_map"]) * example1["atom_pad_mask"]
    )

    # Align on backbone atoms.
    coords1, coords2 = example1["coords"], example2["coords"]  # [N, 3]

    bb_atom_mask = (
        torch.tensor(example1["atom_array"].is_backbone_atom) * atomwise_standard_prot_mask * atom_resolved_mask
    )
    _, aux = data.torch_rmsd_weighted(
        coords1.unsqueeze(0), coords2.unsqueeze(0), bb_atom_mask.unsqueeze(0), return_aux=True
    )
    bb_aligned_coords1 = aux["aligned_a"].squeeze(0)

    # Write aligned coords to mmcif.
    example1["atom_array"].coord = bb_aligned_coords1.numpy()
    with open(f"{out_dir}/{Path(pdb1).stem}.cif", "w") as f:
        f.write(to_cif_string(example1["atom_array"]))

    # Compute metrics.

    # Sidechain RMSD.
    scn_atom_mask = (
        ~torch.tensor(example1["atom_array"].is_backbone_atom) * atomwise_standard_prot_mask * atom_resolved_mask
    )
    tokenwise_squared_errors = apply_token_wise(
        example1["atom_array"],
        (scn_atom_mask[..., None] * (bb_aligned_coords1 - coords2) ** 2),
        lambda x: x.sum().item(),
    )
    tokenwise_atom_counts = torch.tensor(
        apply_token_wise(example1["atom_array"], scn_atom_mask, lambda x: x.sum().item())
    )
    tokenwise_scn_rmsds = (tokenwise_squared_errors / tokenwise_atom_counts.clamp(min=1)).sqrt()
    metrics["scn_rmsd"] = (
        (tokenwise_scn_rmsds * standard_prot_mask).sum() / standard_prot_mask.sum().clamp(min=1)
    ).item()

    # Compute chi angle errors (requires biotite >= 1.6.0 for dihedral_side_chain).
    if hasattr(struc, "dihedral_side_chain"):
        chi_angles1 = struc.dihedral_side_chain(example1["atom_array"])
        chi_angles2 = struc.dihedral_side_chain(example2["atom_array"])
        chi_mask = np.isfinite(chi_angles1) & np.isfinite(chi_angles2)

        # Get residue names for symmetric chi handling.
        res_names = struc.get_residues(example1["atom_array"])[1]
        symmetric_chi_mask = _get_symmetric_chi_mask(res_names, chi_angles1.shape)

        chi_metrics = _chi_metrics(chi_angles1, chi_angles2, chi_mask, symmetric_chi_mask)

        for i in range(0, 4):
            chi_mae_i = chi_metrics["chi_mae"][:, i]
            chi_acc_i = chi_metrics["chi_acc"][:, i]
            chi_mask_i = chi_mask[:, i]
            metrics[f"chi_{i+1}_mae"] = chi_mae_i[chi_mask_i].mean()
            metrics[f"chi_{i+1}_acc"] = chi_acc_i[chi_mask_i].mean()
    else:
        for i in range(0, 4):
            metrics[f"chi_{i+1}_mae"] = float("nan")
            metrics[f"chi_{i+1}_acc"] = float("nan")

    return metrics


def _get_symmetric_chi_mask(res_names: np.ndarray, shape: tuple) -> np.ndarray:
    """
    Get mask indicating which (residue, chi) pairs have symmetric alternatives.
    """
    mask = np.zeros(shape, dtype=bool)
    for i, res_name in enumerate(res_names):
        if res_name in const.RES_NAME_TO_SYMMETRIC_CHI:
            for chi_idx in const.RES_NAME_TO_SYMMETRIC_CHI[res_name]:
                mask[i, chi_idx] = True
    return mask


# Adapated from FlowPacker https://gitlab.com/mjslee0921/flowpacker/-/blob/main/utils/metrics.py?ref_type=heads
def _angle_ae(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    Compute angle absolute error between two arrays of angles.
    """
    ae = np.abs(pred - target)
    ae_alt = np.abs(ae - 2 * np.pi)
    ae = np.minimum(ae, ae_alt)
    return ae


def _chi_metrics(
    chi_angles1: np.ndarray,
    chi_angles2: np.ndarray,
    chi_mask: np.ndarray,
    symmetric_chi_mask: np.ndarray,
    threshold=20,
) -> dict[str, float]:
    ae = _angle_ae(chi_angles1, chi_angles2)
    # For symmetric sidechains, also consider the 180° rotated alternative.
    ae_alt = _angle_ae(chi_angles1, chi_angles2 + np.pi)
    ae_min = np.where(symmetric_chi_mask, np.minimum(ae, ae_alt), ae) * chi_mask
    ae_min = ae_min * 180 / np.pi
    acc = (ae_min <= threshold) * chi_mask

    return {"chi_mae": ae_min, "chi_acc": acc, "chi_mask": chi_mask}
