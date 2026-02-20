"""Evaluation metrics for sidechain packing."""

from contextlib import nullcontext
from pathlib import Path

import biotite.structure as struc
import numpy as np
import torch
from atomworks.io.utils.io_utils import to_cif_string
from atomworks.ml.utils.token import apply_token_wise
from joblib import Parallel, delayed

import caliby.data.const as const
from caliby.data import data
from caliby.eval.eval_utils.seq_des_utils import get_sd_example


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

    # Compute chi angle errors.
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
