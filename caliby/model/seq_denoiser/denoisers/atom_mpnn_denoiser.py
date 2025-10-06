import copy
from collections import defaultdict
from typing import Any

import numpy as np
import torch
from atomworks.ml.utils.token import spread_token_wise
from biotite.structure import AtomArray
from omegaconf import DictConfig
from torchtyping import TensorType
from tqdm import tqdm

import caliby.data.const as const
import caliby.model.seq_denoiser.denoisers.seq_design.potts as potts
from caliby.data.data import to
from caliby.data.feature.feature_utils import slice_feats
from caliby.model.seq_denoiser.denoisers.denoiser import BaseSeqDenoiser
from caliby.model.seq_denoiser.denoisers.seq_design.atom_mpnn import AtomMPNN
from chroma.layers import complexity


class AtomMPNNDenoiser(BaseSeqDenoiser):
    def __init__(self, cfg: DictConfig, sigma_data: tuple[TensorType[(), float], TensorType[(), float]]):
        super().__init__()

        self.cfg = cfg
        self.bb_sigma_data, self.scn_sigma_data = sigma_data
        self.task = cfg.task

        # Random Gaussian noise.
        self.augment_eps = cfg.augment_eps

        # Sequence design model: AtomMPNN
        self.atom_mpnn = AtomMPNN(cfg.mpnn)

    def forward(
        self,
        batch: dict[str, TensorType["b ..."]],
        is_sampling: bool = False,
        sampling_inputs: dict[str, Any] | None = None,
    ) -> tuple[
        TensorType["b n c", float],  # seq_logits
        dict[str, TensorType["b ..."]],
    ]:
        # Build some helpful masks based on conditioning sequence and atoms.
        batch = self.build_masks(batch)

        # During training, add random noise to input coordinates.
        if not is_sampling:
            batch = self.get_training_random_noise(batch)

        # Run model.
        seq_logits, mpnn_feats = self.atom_mpnn(batch, is_sampling)

        # Outputs.
        aux_preds = {
            "seq_logits": seq_logits,
            "potts_decoder_aux": mpnn_feats.get("potts_decoder_aux", None),
            "seq_cond_mask": batch["seq_cond_mask"],
            "atom_cond_mask": batch["atom_cond_mask"],
            "token_exists_mask": batch["token_exists_mask"],
        }

        return seq_logits, aux_preds

    def build_masks(self, batch: dict[str, TensorType["b ..."]]) -> dict[str, TensorType["b ..."]]:
        """
        Build various masks for AtomMPNN.

        Ensures that the conditioning masks only contain non-pad, resolved entries.
        Also, updates batch (in place) with:
        - atomwise_token_idx: Tensor["b n_atoms", int]: index of the token that the atom belongs to, 0 for pad atoms
        - atomwise_seq_cond_mask: Tensor["b n_atoms", float]: 1 if the atom is part of an unmasked residue type, or 0 otherwise
        - token_exists_mask: Tensor["b n_tokens", float]: 1 if there exists any unmasked atom in the token, or 0 otherwise
        """
        # Ensure the conditioning masks only contain non-pad, resolved entries
        batch["seq_cond_mask"] = batch["seq_cond_mask"] * batch["token_pad_mask"] * batch["token_resolved_mask"]
        batch["atom_cond_mask"] = batch["atom_cond_mask"] * batch["atom_pad_mask"] * batch["atom_resolved_mask"]

        # Create atom-level mask which is 1 if the atom is part of an unmasked residue type, or 0 otherwise
        batch["atomwise_seq_cond_mask"] = batch["seq_cond_mask"].gather(
            dim=-1, index=batch["atom_to_token_map"]
        )  # [b, n_atoms]
        batch["atomwise_seq_cond_mask"] = (
            batch["atomwise_seq_cond_mask"] * batch["atom_pad_mask"]
        )  # re-mask out pad atoms, since atom_to_token_map is 0 for pad atoms

        # Build mask for which tokens to include in the token-level grpah
        ## ensure center atom is present, since graph nodes are the center atom
        batch["token_exists_mask"] = batch[
            "token_resolved_mask"
        ].float()  # [b, n_tokens], "whether the token exists in the residue-level graph"

        ## sometimes, it's helpful to mask out certain tokens from the graph (e.g. for protein-only design)
        token_exists_override = batch.get("token_exists_override", torch.ones_like(batch["token_exists_mask"]))
        batch["token_exists_mask"] = batch["token_exists_mask"] * token_exists_override

        return batch

    def get_training_random_noise(self, batch: dict[str, TensorType["b ..."]]) -> dict[str, TensorType["b ..."]]:
        """
        During training, adds random noise and noise labels for input coordinates.

        Updates batch (in place) with:
        - noise: Tensor["b n_atoms 3", float]: random noise for each atom
        """
        if not self.training or self.augment_eps <= 0:
            # if not training or no noise, and not provided, then we assume no noise
            batch["noise"] = batch.get("noise", None)
            return batch

        ## Training: choose random backbone noise ##
        B, N_atoms = batch["atom_pad_mask"].shape
        device = batch["atom_pad_mask"].device

        # global noise, similar to ProteinMPNN
        # add randomly sampled noise to input
        noise = self.augment_eps * torch.randn((B, N_atoms, 3), device=device)

        batch["noise"] = noise
        return batch

    def potts_sample(
        self, batch: dict[str, TensorType["b ..."]], sampling_inputs: dict[str, Any]
    ) -> tuple[dict[str, list[AtomArray]], dict[str, list[dict]]]:
        """
        Potts sampling for sequence design.

        Returns:
            id_to_atom_arrays: dict[str, list[AtomArray]]: maps from example_id to list of output atom arrays
            id_to_aux: dict[str, list[dict]]: maps from example_id to list of auxiliary outputs for each sample
        """
        aux = {}

        # If specified, use Gaussian noise to generate multiple conformers.
        noise_std = sampling_inputs["gaussian_conformers_cfg"]["noise_std"]
        if noise_std > 0:
            _generate_gaussian_conformers(
                batch=batch,
                noise_std=noise_std,
                n_gaussian_conformers=sampling_inputs["gaussian_conformers_cfg"]["n_conformers"],
            )

        # If specified, condition on sequence only in the Potts model.
        batch["seq_cond_mask_potts"] = batch["seq_cond_mask"].clone()
        if sampling_inputs["potts_sampling_cfg"].get("potts_only_cond", False):
            print("Conditioning on sequence only in the potts model")
            batch["seq_cond_mask"] = torch.zeros_like(
                batch["seq_cond_mask"]
            )  # zero out model-level sequence conditioning mask

        # Compute potts parameters.
        potts_decoder_aux, batch, sampling_inputs = self.compute_potts_params(batch, sampling_inputs)
        aux["potts_decoder_aux"] = to(potts_decoder_aux, "cpu")

        # Set up Potts sampling
        potts_sampling_cfg = sampling_inputs["potts_sampling_cfg"]
        regularization = potts_sampling_cfg["regularization"]
        potts_sweeps = potts_sampling_cfg["potts_sweeps"]
        potts_proposal = potts_sampling_cfg["potts_proposal"]
        potts_temperature = potts_sampling_cfg["potts_temperature"]
        rejection_step = potts_sampling_cfg.get("rejection_step", potts_proposal == "chromatic")

        B, N, _ = batch["restype"].shape
        logits_init = torch.zeros((B, N, const.AF3_ENCODING.n_tokens), device=batch["restype"].device).float()

        # Handle banned amino acids and aatype restrictions.
        ban_S = {"X"}
        omit_aas = sampling_inputs.get("omit_aas", None)
        if omit_aas is not None:
            ban_S = ban_S | set(omit_aas)
        ban_S = const.AF3_ENCODING.encode_aa_seq(ban_S)
        ban_S = ban_S + const.AF3_ENCODING.encode(const.AF3_ENCODING.non_protein_tokens)  # ban all non-protein tokens

        # Initialize random sequence and sampling masks.
        mask_sample = (1 - batch["seq_cond_mask_potts"]) * batch[
            "token_pad_mask"
        ]  # 1 where we can sample, 0 where we can't
        mask_sample, _, S_init = potts.init_sampling_masks(
            logits_init,
            mask_sample=mask_sample,
            S=batch["restype"].argmax(dim=-1),
            ban_S=ban_S,
            pos_restrict_aatype=sampling_inputs.get("pos_restrict_aatype", None),
        )

        # Complexity regularization.
        penalty_func = None
        mask_ij_coloring = None
        edge_idx_coloring = None
        if regularization == "LCP":
            C_complexity = batch["asym_id"] - torch.min(batch["asym_id"]) + 1  # renumber asym_id to have min value of 1
            C_complexity = (
                C_complexity * batch["token_pad_mask"] * batch["token_exists_mask"]
            )  # mask out pad tokens and tokens that don't exist in the graph
            penalty_func = lambda _S: complexity.complexity_lcp(_S, C_complexity)

        # Design sequences.
        S = []  # keep track of sequences for each sample
        aux["U"] = []  # keep track of energies for each sample
        for _ in tqdm(range(sampling_inputs["num_seqs_per_pdb"]), desc="Sampling sequences", leave=False):
            S_sample, U_sample = self.atom_mpnn.decoder_S_potts.sample(
                potts_decoder_aux["h"],
                potts_decoder_aux["J"],
                potts_decoder_aux["edge_idx"],
                potts_decoder_aux["mask_i"],
                potts_decoder_aux["mask_ij"],
                S=S_init,
                mask_sample=mask_sample,
                temperature=potts_temperature,
                num_sweeps=potts_sweeps,
                penalty_func=penalty_func,
                proposal=potts_proposal,
                rejection_step=rejection_step,
                verbose=False,
                edge_idx_coloring=edge_idx_coloring,
                mask_ij_coloring=mask_ij_coloring,
            )
            # Set all tokens that don't exist in the graph to unknown.
            S_sample = torch.where(
                ~batch["token_exists_mask"].bool() & (batch["is_protein"] | batch["is_ligand"]),
                const.AF3_ENCODING.token_to_idx[const.UNKNOWN_AA],
                S_sample,
            )
            S_sample = torch.where(
                ~batch["token_exists_mask"].bool() & batch["is_rna"],
                const.AF3_ENCODING.token_to_idx[const.UNKNOWN_RNA],
                S_sample,
            )
            S_sample = torch.where(
                ~batch["token_exists_mask"].bool() & batch["is_dna"],
                const.AF3_ENCODING.token_to_idx[const.UNKNOWN_DNA],
                S_sample,
            )

            aux["U"].append(U_sample.cpu())
            S.append(S_sample.cpu())

        batch = to(batch, device="cpu")

        # Thread sequences onto atom arrays.
        id_to_atom_arrays = defaultdict(list)
        id_to_aux = defaultdict(list)
        for si in range(len(S)):  # iterate over num_seqs_per_pdb
            atom_arrays = copy.deepcopy(batch["atom_array"])

            for bi in range(len(atom_arrays)):  # iterate over batch size
                token_pad_mask = batch["token_pad_mask"][bi].bool()
                atom_pad_mask = batch["atom_pad_mask"][bi].bool()

                new_restype = S[si][bi][token_pad_mask]
                new_coords = batch["coords"][bi][atom_pad_mask]

                example_id = batch["example_id"][bi]
                atom_array = atom_arrays[bi]
                seq_cond_mask = batch["seq_cond_mask"][bi][token_pad_mask]
                atom_cond_mask = batch["atom_cond_mask"][bi][atom_pad_mask]
                atom_resolved_mask = batch["atom_resolved_mask"][bi][atom_pad_mask]

                # Update resnames.
                update_seq_mask = ~seq_cond_mask.numpy().astype(bool)  # update where seq_cond_mask is False
                atomwise_update_seq_mask = spread_token_wise(atom_array, update_seq_mask)
                atomwise_resnames = spread_token_wise(atom_array, const.AF3_ENCODING.idx_to_token[new_restype])
                atomwise_resnames = np.where(
                    atomwise_update_seq_mask, atomwise_resnames, atom_array.get_annotation("res_name")
                )
                atom_array.set_annotation("res_name", atomwise_resnames)

                # Update coords.
                update_coords_mask = (atom_cond_mask * atom_resolved_mask).numpy().astype(bool)
                atom_array.coord = np.where(update_coords_mask[..., None], new_coords.numpy(), np.nan)

                # Add to id_to_atom_arrays.
                id_to_atom_arrays[example_id].append(atom_array)

                # Add additional auxiliary outputs.
                id_to_aux[example_id].append(
                    {
                        "U": aux["U"][si][bi].cpu().item(),
                        "S": new_restype.cpu(),
                    }
                )

        return id_to_atom_arrays, id_to_aux

    def compute_potts_params(
        self, batch: dict[str, TensorType["b ..."]], sampling_inputs: dict[str, Any]
    ) -> tuple[dict[str, TensorType["b ..."]], dict[str, TensorType["b ..."]], dict[str, Any]]:
        """
        Run model and collect potts parameters over a batch of samples.

        If "tied_sampling_ids" is in batch, we will aggregate potts parameters across tied groups and slice batch
        to representative elements.

        Returns:
            potts_decoder_aux: dict[str, TensorType["b ..."]]: potts parameters
            batch: dict[str, TensorType["b ..."]]: batch with token_exists_mask added
            sampling_inputs: dict[str, Any]: sampling inputs with pos_restrict_aatype sliced to representative elements
        """
        subbatch_size = sampling_inputs["batch_size"]
        B = batch["restype"].shape[0]

        # Run model and collect potts parameters
        potts_decoder_aux = {}  # potts parameters
        token_exists_mask = []  # keep track of the tokens that exist in the graph
        for bi in tqdm(range(0, B, subbatch_size), desc="Computing potts parameters", leave=False):
            subbatch = slice_feats(batch, slice(bi, bi + subbatch_size))

            _, aux_preds_i = self(subbatch, is_sampling=True, sampling_inputs=sampling_inputs)

            for k, v in aux_preds_i["potts_decoder_aux"].items():
                potts_decoder_aux.setdefault(k, []).append(v)
            token_exists_mask.append(aux_preds_i["token_exists_mask"])
        potts_decoder_aux = {k: torch.cat(v, dim=0) for k, v in potts_decoder_aux.items()}
        token_exists_mask = torch.cat(token_exists_mask, dim=0)
        batch["token_exists_mask"] = token_exists_mask  # store in batch for downstream use

        # Handle tied sampling
        if "tied_sampling_ids" in batch:
            tied_sampling_inputs = _construct_tied_sampling_inputs(batch)

            # slice to representative elements
            unique_rep_idxs = tied_sampling_inputs["rep_idx"].unique().tolist()
            batch = slice_feats(batch, unique_rep_idxs)  # get representative batch elements

            if sampling_inputs.get("pos_restrict_aatype", None) is not None:
                sampling_inputs["pos_restrict_aatype"] = [
                    x[unique_rep_idxs] for x in sampling_inputs["pos_restrict_aatype"]
                ]

            # aggregate potts parameters across tied groups
            potts_decoder_aux = _aggregate_potts_params(potts_decoder_aux, tied_sampling_inputs)

        return potts_decoder_aux, batch, sampling_inputs


def _aggregate_potts_params(
    potts_decoder_aux: dict[str, TensorType["b ..."]],
    tied_sampling_inputs: dict[str, Any],
    use_mean: bool = True,
) -> dict[str, TensorType["b ..."]]:
    """
    Aggregate potts parameters across tied groups.

    If use_mean, we take the mean of the potts parameters across the tied groups (equivalent to geometric mean
    in probability space)
    """
    h, J, edge_idx, mask_i, mask_ij = (
        potts_decoder_aux["h"],
        potts_decoder_aux["J"],
        potts_decoder_aux["edge_idx"],
        potts_decoder_aux["mask_i"],
        potts_decoder_aux["mask_ij"],
    )
    inverse, unique_ids = tied_sampling_inputs["inverse"], tied_sampling_inputs["unique_ids"]

    # handle 1D features
    counts = torch.bincount(inverse)
    h_new = h.new_zeros(unique_ids.shape[0], *h.shape[1:]).index_add(0, inverse, h)
    node_counts = mask_i.new_zeros(unique_ids.shape[0], *mask_i.shape[1:]).index_add(0, inverse, mask_i)
    mask_i_new = (
        node_counts == counts.view(-1, 1)
    ).float()  # node i is unmasked only if node i is present across all inputs in the tied group

    # handle 2D features
    n_grp = unique_ids.shape[0]
    B, N, K = edge_idx.shape
    C = J.shape[-1]
    edge_counts = mask_ij.new_zeros(n_grp, N, N)
    J_new = J.new_zeros(n_grp, N, N, C, C)
    for bi in range(B):
        g = inverse[bi]

        edge_indices_flat = (edge_idx[bi] + torch.arange(N, device=edge_idx.device)[:, None] * N).reshape(-1)
        edge_counts[g].view(-1).index_add_(
            0, edge_indices_flat, mask_ij[bi].view(-1)
        )  # count number of edges between each pair of nodes
        J_new[g].view(-1, C, C).index_add_(
            0, edge_indices_flat, J[bi].view(-1, C, C)
        )  # add in the pairwise interactions for this graph

    mask_ij_new = (edge_counts > 0) * (
        mask_i_new[:, :, None] * mask_i_new[:, None, :]
    )  # edge i,j is present only if both nodes are present and there exists some edge between them
    edge_idx_new = (
        torch.arange(N, device=edge_idx.device).expand(1, 1, -1).repeat(n_grp, N, 1)
    )  # new edge indices are given in the full NxN grid

    if use_mean:
        J_new = J_new / counts.view(-1, 1, 1, 1, 1)
        h_new = h_new / counts.view(-1, 1, 1)

    potts_decoder_aux_new = {
        "h": h_new,
        "J": J_new,
        "edge_idx": edge_idx_new,
        "mask_i": mask_i_new,
        "mask_ij": mask_ij_new,
    }

    return potts_decoder_aux_new


def _construct_tied_sampling_inputs(batch: dict[str, TensorType["b ..."]]) -> dict[str, Any]:
    tied_sampling_inputs = {"tied_sampling_ids": batch["tied_sampling_ids"]}
    device = batch["tied_sampling_ids"].device
    tied_sampling_inputs["unique_ids"], tied_sampling_inputs["inverse"] = tied_sampling_inputs[
        "tied_sampling_ids"
    ].unique(return_inverse=True)

    # use first index of each tied group as the representative index
    B = batch["restype"].shape[0]
    batch_idx = torch.arange(B, device=device)
    n_unique_ids = tied_sampling_inputs["unique_ids"].shape[0]
    first_idxs = torch.full((n_unique_ids,), B, device=device)
    first_idxs.scatter_reduce_(0, tied_sampling_inputs["inverse"], batch_idx, reduce="amin", include_self=True)
    tied_sampling_inputs["rep_idx"] = first_idxs[tied_sampling_inputs["inverse"]]
    return tied_sampling_inputs


def _repeat_batch(batch: dict[str, TensorType["b ..."]], n_repeats: int) -> dict[str, TensorType["b ..."]]:
    """
    Repeat (repeat_interleave in torch, repeat in numpy) a batch n_repeats times.
    """
    for k, v in batch.items():
        if v is None:
            batch[k] = v
        elif isinstance(v, torch.Tensor):
            batch[k] = torch.repeat_interleave(v, n_repeats, dim=0)
        elif isinstance(v, np.ndarray):
            batch[k] = np.repeat(v, n_repeats, axis=0)
        elif isinstance(v, list):
            batch[k] = [elem for item in v for elem in [item] * n_repeats]
        else:
            raise ValueError(f"Unsupported input type: expected torch.Tensor, numpy.ndarray, or list, got {type(v)}")
    return batch


def _generate_gaussian_conformers(
    *, batch: dict[str, TensorType["b ..."]], noise_std: float, n_gaussian_conformers: int
) -> dict[str, TensorType["b ..."]]:
    """
    Generate multiple Gaussian conformers for a batch.
    """
    if n_gaussian_conformers is None or n_gaussian_conformers <= 0:
        raise ValueError("n_gaussian_conformers must be greater than 0 if noise_std > 0")

    # If tied_sampling_ids is not in the batch, we tie each sample with its own Gaussian conformers.
    if "tied_sampling_ids" not in batch:
        batch["tied_sampling_ids"] = torch.arange(batch["restype"].shape[0], device=batch["restype"].device)

    # Create multiple conformers.
    batch = _repeat_batch(batch, n_gaussian_conformers)

    # Add noise to each conformer.
    batch["coords"] = batch["coords"] + torch.randn_like(batch["coords"]) * noise_std

    # Mask out pad atoms and resolved atoms.
    batch["coords"] = batch["coords"] * batch["atom_pad_mask"].unsqueeze(-1) * batch["atom_resolved_mask"].unsqueeze(-1)
    return batch
