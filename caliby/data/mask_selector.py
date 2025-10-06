from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from omegaconf import DictConfig, OmegaConf
from torchtyping import TensorType

from caliby.data import const


class MaskSelector:
    def __init__(self, cfg: DictConfig):
        """
        Handles selecting masks for training the sequence design model.
        """
        super().__init__()
        self.cfg = cfg

        self.restype_masking_schedule = cfg.restype_masking_schedule
        self.restype_masking_cfg = OmegaConf.to_container(
            cfg.restype_masking_cfg[self.restype_masking_schedule], resolve=True
        )  # to dict to avoid dataloader issues?

    def sample_seq_cond_mask(
        self, batch: dict[str, TensorType["b ..."]], t: TensorType["b", float] | None = None
    ) -> TensorType["b n_tokens", float]:
        """
        Create a mask denoting which restypes to mask out.
        0 if we should mask, 1 if we should keep. Non-protein restypes are always kept (1).
        """
        B, N = batch["token_pad_mask"].shape
        device = batch["token_pad_mask"].device

        if t is None:
            # Sample timestep
            t = self._sample_t(B, device)

        # Create mask based on timestep
        seq_cond_mask = torch.rand(B, N, device=device) < rearrange(t, "b -> b 1")

        # Non-protein and non-standard restypes are always kept
        standard_prot_mask = batch["is_protein"] & ~batch["is_atomized"]
        seq_cond_mask = torch.where(~standard_prot_mask, torch.ones_like(seq_cond_mask), seq_cond_mask)

        seq_cond_mask = seq_cond_mask * batch["token_pad_mask"]  # mask out padding
        return seq_cond_mask

    def sample_atom_cond_mask(self, batch: dict[str, TensorType["b ..."]]) -> TensorType["b n_atoms", float]:
        """
        Create a mask denoting which atoms to mask out.
        0 if we should mask, 1 if we should keep.
        """
        B, _ = batch["atom_to_token_map"].shape
        device = batch["atom_resolved_mask"].device

        atom_cond_mask = batch["atom_resolved_mask"].clone()  # [n_atoms]
        prot_bb_atom_mask = batch["prot_bb_atom_mask"] * atom_cond_mask

        # Mask out the sidechain of a token with probability p in U[0, 1]  # TODO: try different masking schemes
        tok_keep_scn_p = torch.rand(B, device=device)
        tok_keep_scn_mask = torch.rand_like(batch["seq_cond_mask"]) < rearrange(tok_keep_scn_p, "b -> b 1")
        tok_keep_scn_mask = (
            tok_keep_scn_mask * batch["seq_cond_mask"]
        )  # sidechains should be masked where seq is masked
        standard_prot_mask = batch["is_protein"] & ~batch["is_atomized"]
        tok_keep_scn_mask = torch.where(
            ~standard_prot_mask,  # non-protein tokens should be kept
            torch.ones_like(tok_keep_scn_mask),
            tok_keep_scn_mask,
        )

        ## convert to atomwise mask
        atomwise_tok_keep_scn_mask = (
            tok_keep_scn_mask.gather(dim=-1, index=batch["atom_to_token_map"]) * batch["atom_pad_mask"]
        )  # [b, n_atoms]

        atom_cond_mask = torch.where(atomwise_tok_keep_scn_mask.bool(), atom_cond_mask, prot_bb_atom_mask)

        return atom_cond_mask

    def _sample_t(self, B: int, device: torch.device) -> TensorType["b", float]:
        """
        Sample a timestep from the masking schedule.
        t = probability of keeping the restype unmasked
        """
        if self.restype_masking_schedule == "constant_t":
            t = torch.ones(B, device=device) * self.restype_masking_cfg["t"]
        elif self.restype_masking_schedule.startswith("uniform"):
            # sample time from uniform distribution
            t_min, t_max = self.restype_masking_cfg["t_min"], self.restype_masking_cfg["t_max"]
            t = torch.rand(B, device=device) * (t_max - t_min) + t_min

            # apply transformation to t
            if self.restype_masking_schedule == "uniform_t":
                t = t
            elif self.restype_masking_schedule == "uniform_squared_t":
                t = t**2
            elif self.restype_masking_schedule == "uniform_cubed_t":
                t = t**3
            elif self.restype_masking_schedule == "uniform_cosine_t":
                t = 1 - torch.cos(t * np.pi / 2)
            elif self.restype_masking_schedule == "uniform_sqrt_t":
                t = t**0.5
            elif self.restype_masking_schedule == "uniform_cbrt_t":
                t = t ** (1 / 3)

        return t


##################### Atom-level motif selection #####################
def select_all_atoms(feats: dict[str, torch.Tensor]) -> TensorType["n_atoms", float]:
    """
    Selects all atoms for the AtomDenoiser.
    """
    return feats["atom_resolved_mask"]


def select_protein_sidechain_atoms(feats: dict[str, torch.Tensor]) -> TensorType["n_atoms", float]:
    """
    Selects protein sidechain atoms for the AtomDenoiser.
    """
    return feats["prot_scn_atom_mask"] * feats["atom_resolved_mask"]


def select_protein_backbone_atoms(feats: dict[str, torch.Tensor]) -> TensorType["n_atoms", float]:
    """
    Selects protein backbone atoms for the AtomDenoiser.
    """
    return feats["prot_bb_atom_mask"] * feats["atom_resolved_mask"]


def get_mask_selector(cfg: Optional[DictConfig]) -> Optional[MaskSelector]:
    """
    Get the mask selector specified in the config.
    """
    if cfg.name == "mask_selector":
        return MaskSelector(cfg)
    else:
        raise ValueError(f"Unknown mask selector: {cfg.name}")
