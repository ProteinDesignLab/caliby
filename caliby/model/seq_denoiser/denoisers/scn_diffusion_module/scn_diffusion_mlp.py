from collections import defaultdict
from functools import partial
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from atomworks.ml.utils.geometry import apply_batched_rigid, invert_rigid, rigid_from_3_points
from einops import rearrange, repeat
from omegaconf import DictConfig
from torchtyping import TensorType

import caliby.data.const as const
from caliby.model.seq_denoiser.denoisers.scn_diffusion_module.dit_utils import (
    DenoisingMLPBlock,
    FinalLayer,
    TimestepEmbedder,
)
from caliby.model.seq_denoiser.denoisers.scn_diffusion_module.edm_interpolant import EDM


class SidechainDiffusionModule(nn.Module):
    def __init__(self, cfg: DictConfig, scn_sigma_data: TensorType[(), float]):
        """
        Sidechain denoising module.
        """
        super().__init__()
        self.cfg = cfg
        self.use_self_conditioning = cfg.use_self_conditioning

        self.scn_interpolant = EDM(cfg.interpolant, sigma_data=scn_sigma_data)

        # Set up denoising model
        self.scn_denoiser = SidechainMLP(cfg.scn_denoiser, self.scn_interpolant)

    def sidechain_diffusion(
        self,
        mpnn_feature_dict: Dict[str, TensorType["b ..."]],
        batch: dict[str, TensorType["b ..."]],
        is_sampling: bool,
        sampling_inputs: dict[str, Any] | None = None,
    ) -> Tuple[TensorType["b n a 3", float], Dict[str, TensorType["b ...", float]]]:
        h_V = mpnn_feature_dict["h_V"]
        B, N, _ = h_V.shape
        diffusion_aux = defaultdict(lambda: None)
        aatype = batch["restype"].argmax(dim=-1)

        if not is_sampling:
            # === Training === #
            # Get ground truth sidechains for diffusion
            x_scn_gt = batch["encoded_xyz"][..., const.AF2_SCN_IDXS, :]
            atom_mask_gt = batch["encoded_mask"]  # use ground truth atom mask for ghost / missing atoms

            # Transform sidechains from ground truth to local frame
            x_scn_local_gt, bb_frames_exists = transform_sidechain_frame(
                x_scn_gt,
                batch["encoded_xyz"][..., const.AF2_BB_IDXS, :],
                atom_mask_gt[..., const.AF2_SCN_IDXS],
                atom_mask_gt[..., const.AF2_BB_IDXS],
                to_local=True,
            )

            # Repeat inputs for batch multiplier
            M = self.cfg.training_batch_size_mult
            x_scn_local_gt_batched = repeat(x_scn_local_gt, "b n a x -> (m b) n a x", m=M, b=B)
            bb_frames_exists_batched = repeat(bb_frames_exists, "b n -> (m b) n", m=M, b=B)
            h_V_batched = repeat(h_V, "b n h -> (m b) n h", m=M, b=B)
            aatype_batched = repeat(aatype, "b n -> (m b) n", m=M, b=B)
            seq_mask_batched = repeat(batch["token_pad_mask"], "b n -> (m b) n", m=M, b=B)

            # Evaluate at specific timesteps (for validation)
            t_sd_batched = None
            if batch.get("t_scd", None) is not None:
                t_sd_batched = torch.full((M * B,), batch["t_scd"], device=x_scn_local_gt_batched.device)

            # Noise the ground truth local sidechains
            interpolant_out = self.scn_interpolant({"x": x_scn_local_gt_batched}, t=t_sd_batched)
            xt_scn_local_batched = interpolant_out["x_noised"]
            x_scn_target_batched = interpolant_out["x_target"]
            t_batched = interpolant_out["t"]
            loss_weight_t_batched = interpolant_out["loss_weight_t"]

            # Run small denoising MLP
            denoiser_fn = self.scn_denoiser
            if self.use_self_conditioning and (np.random.uniform() < self.cfg.self_cond_p):
                # Apply self-conditioning
                with torch.no_grad():
                    x1_scn_local_batched, aux_preds = denoiser_fn(
                        xt_scn_local_batched, aatype_batched, t_batched, h_V_batched, seq_mask=seq_mask_batched
                    )
                torch.clear_autocast_cache()  # Sidestep AMP bug (PyTorch issue #65766)
                denoiser_fn = partial(denoiser_fn, x_scn_self_cond=x1_scn_local_batched)

            x1_scn_local_batched, aux_preds = denoiser_fn(
                xt_scn_local_batched, aatype_batched, t_batched, h_V_batched, seq_mask=seq_mask_batched
            )

            # Cache intermediates for computing loss
            diffusion_aux["scn_pred"] = x1_scn_local_batched
            diffusion_aux["scn_target"] = (
                x_scn_target_batched  # diffusion target; for edm this is just the ground truth coordinates
            )
            diffusion_aux["loss_weight_t"] = loss_weight_t_batched
            diffusion_aux["bb_frames_exists"] = bb_frames_exists_batched  # don't compute loss if bb frame doesn't exist

        else:
            # === Sampling === #
            # Sample sidechains from prior
            A = len(const.AF2_SCN_IDXS)
            x0_scn_local = self.scn_interpolant.sample_prior((B, N, A, 3), h_V.device)

            # Extract sampling parameters
            scn_packing_cfg = sampling_inputs["scn_packing_cfg"]
            S_scd = scn_packing_cfg["num_steps"]
            step_scale = scn_packing_cfg["step_scale"]
            timesteps = torch.linspace(0.0, 1.0, S_scd + 1, device=h_V.device)[None].expand(B, S_scd + 1)

            denoiser_fn = partial(self.scn_denoiser, aatype=aatype, h_V=h_V, seq_mask=batch["token_pad_mask"])
            # Run integration steps
            xt_scn_local = x0_scn_local
            for i in range(S_scd):
                t = timesteps[:, i]
                t_next = timesteps[:, i + 1]
                xt_scn_local, aux_preds = self.scn_interpolant.euler_step(
                    denoiser_fn, xt_scn_local, t=t, t_next=t_next, step_scale=step_scale
                )

                if self.use_self_conditioning:
                    # Apply self-conditioning
                    denoiser_fn = partial(denoiser_fn, x_scn_self_cond=aux_preds["x1_pred"])

            # Finalize outputs.

            # First, we transform the denoised sidechains back to global coordinates.
            x_bb = batch["encoded_xyz"][..., const.AF2_BB_IDXS, :]

            # 0 denotes missing backbone atoms in input backbone.
            atom_mask_bb = batch["encoded_atom_or_ghost_mask"][..., const.AF2_BB_IDXS]

            # All sidechain atoms are present for the packed aatype, since we sampled them.
            atom_mask_scn = batch["encoded_standard_atom_mask"][..., const.AF2_SCN_IDXS]

            # Transform denoised sidechains back to global coordinates.
            x1_scn, _ = transform_sidechain_frame(
                xt_scn_local, x_bb, atom_mask_scn, atom_mask_bb, to_local=False
            )  # if backbone frame doesn't exist, we predict all sidechain atoms at CA

            # Update coords and mask with packed sidechains.
            encoded_xyz_packed = batch["encoded_xyz"].clone()
            encoded_xyz_packed[..., const.AF2_SCN_IDXS, :] = x1_scn
            encoded_mask_packed = batch["encoded_mask"].clone()
            encoded_mask_packed[..., const.AF2_SCN_IDXS] = atom_mask_scn.bool()
            diffusion_aux["encoded_xyz_packed"] = encoded_xyz_packed
            diffusion_aux["encoded_mask_packed"] = encoded_mask_packed

        return diffusion_aux


class SidechainMLP(nn.Module):
    def __init__(self, cfg: DictConfig, scn_interpolant: EDM):
        """
        MLP for per-token sidechain diffusion conditioned on MPNN sequence embeddings.
        """
        super().__init__()

        self.cfg = cfg
        self.scn_interpolant = scn_interpolant

        # Set up MLP model
        self.use_self_conditioning = cfg.use_self_conditioning
        self.in_channels = 33 * 3  # 33 * 3; input sidechain atoms
        self.in_channels += const.AF3_ENCODING.n_tokens  # concatenate one-hot encoded amino acid type

        self.out_channels = 33 * 3  # 33 * 3; output all sidechain atoms
        if self.use_self_conditioning:
            self.in_channels += self.out_channels  # concatenate input with output from previous timestep

        # Conditioning
        self.timestep_embedder = TimestepEmbedder(cfg.hidden_size)

        # node embedding conditioning
        self.h_V_embedder = nn.Linear(cfg.c_h_V, cfg.hidden_size)

        # Blocks
        self.x_embedder = nn.Linear(self.in_channels, cfg.hidden_size)

        # Blocks
        self.blocks = nn.ModuleList(
            [
                DenoisingMLPBlock(cfg.hidden_size, mlp_dropout=cfg.mlp_dropout, mlp_ratio=cfg.mlp_ratio)
                for _ in range(cfg.depth)
            ]
        )
        self.final_layer = FinalLayer(cfg.hidden_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.timestep_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.timestep_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(
        self,
        x_scn: TensorType["b n a_scn 3", float],  # noisy sidechain atoms
        aatype: TensorType["b n", float],  # aatype to condition on (predicted during inference; GT during training)
        t: TensorType["b n", float],  # timestep
        h_V: TensorType["b n h", float],  # conditioning latent
        seq_mask: TensorType["b n", float],
        x_scn_self_cond: Optional[TensorType["b n a_scn 3", float]] = None,  # self-conditioning input
    ) -> Tuple[TensorType["b n a 3", float], Dict[str, TensorType["b ..."]]]:
        aux_preds = {}

        # Preconditioning
        precondition_in, precondition_out = self.scn_interpolant.setup_preconditioning(x_scn, x_scn_self_cond, t)
        x_scn, x_scn_self_cond, t = precondition_in()  # input preconditioning

        # Concatenate self-conditioning
        if self.use_self_conditioning:
            if x_scn_self_cond is None:
                x_scn_self_cond = torch.zeros_like(x_scn)
            x_scn = torch.cat([x_scn, x_scn_self_cond], dim=-1)

        x = rearrange(x_scn, "b n a x -> b n (a x)")

        # Concatenate one-hot sequence conditioning
        aatype_oh = F.one_hot(
            aatype, num_classes=const.AF3_ENCODING.n_tokens
        ).float()  # aatype is ground truth during training
        x = torch.cat([x, aatype_oh], dim=-1)

        # Begin MLP forward pass
        x = self.x_embedder(x)

        # Conditioning
        # embed timestep
        c = self.timestep_embedder(t).unsqueeze(1)

        # add conditioning from h_V
        h_V = self.h_V_embedder(h_V)
        c = c + h_V
        x = x + h_V

        # MLP blocks
        for block in self.blocks:
            x = block(x, c)

        # Final output
        x = self.final_layer(x, c, per_token_conditioning=True)
        x = x * seq_mask[..., None]  # zero out padding positions

        # Reshape back to coordinates
        x = rearrange(x, "b n (a x) -> b n a x", x=3)
        x_scn = precondition_out(x)  # output preconditioning on sidechains

        return x_scn, aux_preds


def backbone_coords_to_frames(
    x_bb: TensorType["... 4 3", float], atom_mask: TensorType["... 4", float]
) -> tuple[tuple[TensorType["... 3 3", float], TensorType["... 3", float]], TensorType["... 4", float]]:
    """
    Convert backbone coordinates to local frames (rotation + translation) for each residue.
    """
    base_atom_names = ["C", "CA", "N"]
    rigid_group_base_atom37_idx = torch.tensor([const.AF2_BB_ATOM_ORDER[atom] for atom in base_atom_names])
    base_atom_pos = x_bb[..., rigid_group_base_atom37_idx, :]
    rigid = rigid_from_3_points(
        x1=base_atom_pos[..., 0, :],
        x2=base_atom_pos[..., 1, :],
        x3=base_atom_pos[..., 2, :],
    )
    gt_exists = torch.min(atom_mask[..., rigid_group_base_atom37_idx], dim=-1)[0]

    return rigid, gt_exists


def transform_sidechain_frame(
    x_scn: TensorType["b n 33 3", float],
    x_bb: TensorType["b n 4 3", float],
    atom_mask_scn: TensorType["b n 33", float],
    atom_mask_bb: TensorType["b n 4", float],
    to_local: bool,
) -> Tuple[TensorType["b n 33 3", float], TensorType["b n", float]]:
    """
    Transform sidechain coordinates based on the backbone frame.
    If to_local, transform from global to local frame. Otherwise, transform from local to global frame.
    """
    rigid, bb_frames_exists = backbone_coords_to_frames(x_bb, atom_mask_bb)
    B, N, A, _ = x_scn.shape

    rigid = (
        rigid[0].unsqueeze(2).expand(B, N, A, 3, 3).reshape(B * N * A, 3, 3),
        rigid[1].unsqueeze(2).expand(B, N, A, 3).reshape(B * N * A, 3),
    )

    if to_local:
        # Transform from global to local frame, ghost atom value is at 0
        x_scn = apply_batched_rigid(invert_rigid(rigid), x_scn.reshape(B * N * A, 1, 3)).reshape(B, N, A, 3)
        ghost_atom_value = 0
    else:
        # Transform from local to global frame, ghost atom value is at CA
        x_scn = apply_batched_rigid(rigid, x_scn.reshape(B * N * A, 1, 3)).reshape(B, N, A, 3)
        ca_idx = const.AF2_BB_ATOM_ORDER["CA"]
        ghost_atom_value = x_bb[..., ca_idx : ca_idx + 1, :]

    # "Zero out" ghost atoms and missing atoms.
    x_scn = torch.where(atom_mask_scn[..., None].bool(), x_scn, ghost_atom_value)

    # "Zero out" sidechain atoms where backbone frame does not exist.
    x_scn = torch.where(bb_frames_exists[..., None, None].bool(), x_scn, ghost_atom_value)

    return x_scn, bb_frames_exists
