import copy
from typing import Any

import torch
import torch.nn as nn
from biotite.structure import AtomArray
from omegaconf import DictConfig
from torchtyping import TensorType

from caliby.data.mask_selector import MaskSelector
from caliby.model.seq_denoiser.denoisers.atom_mpnn_denoiser import AtomMPNNDenoiser
from caliby.model.seq_denoiser.denoisers.denoiser import BaseSeqDenoiser
from caliby.model.seq_denoiser.denoisers.seq_design.potts import compute_potts_energy


class SeqDenoiser(nn.Module):
    """
    Sequence denoiser model.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.task = cfg.task

        # Data scaling parameters.
        self.register_buffer("bb_std", torch.tensor(1.0))
        self.register_buffer("bb_mean", torch.tensor(0.0))

        self.register_buffer("scn_mean", torch.tensor(0.0))
        self.register_buffer("scn_std", torch.tensor(1.0))

        self.sigma_data = (self.bb_std, self.scn_std)

        self.denoiser = get_denoiser(cfg.denoiser, self.sigma_data)

        # Mask selector
        self.mask_selector = MaskSelector(cfg.mask_selector)

    def setup(self):
        # Initialize denoiser pre-trained weights if needed.
        self.denoiser.setup()

    def forward(
        self, batch: dict[str, TensorType["b ..."]], t: TensorType["b", float] | None = None
    ) -> dict[str, TensorType["b ..."]]:
        outputs = {}

        # Copy batch to avoid modifying the original.
        batch = copy.deepcopy(batch)

        with torch.no_grad():
            # Sample sequence and atom conditioning masks.
            batch["seq_cond_mask"] = self.mask_selector.sample_seq_cond_mask(
                batch, t
            )  # 1 if we should condition on the restype, 0 otherwise
            batch["atom_cond_mask"] = self.mask_selector.sample_atom_cond_mask(
                batch
            )  # 1 if we should condition on the atom, 0 otherwise

            # Ensure the conditioning masks only contain non-pad, resolved entries.
            batch["seq_cond_mask"] = batch["seq_cond_mask"] * batch["token_pad_mask"] * batch["token_resolved_mask"]
            batch["atom_cond_mask"] = batch["atom_cond_mask"] * batch["atom_pad_mask"] * batch["atom_resolved_mask"]

        # Denoise sequence.
        _, aux_preds = self.denoiser(batch)

        # Additional outputs for computing loss.
        outputs.update(aux_preds)

        return outputs

    def set_scale_factors(self, scale_factors: dict[str, tuple[float, float]]):
        bb_mean, bb_std = scale_factors["bb"]
        self.bb_mean.data = torch.tensor(bb_mean)
        self.bb_std.data = torch.tensor(bb_std)
        print(f"Setting bb_mean: {bb_mean}, bb_std: {bb_std}")

        scn_mean, scn_std = scale_factors["scn"]
        self.scn_mean.data = torch.tensor(scn_mean)
        self.scn_std.data = torch.tensor(scn_std)
        print(f"Setting scn_mean: {scn_mean}, scn_std: {scn_std}")

    @torch.no_grad()
    def sample(
        self, batch: dict[str, TensorType["b ..."]], sampling_inputs: dict[str, Any]
    ) -> tuple[dict[str, list[AtomArray]], dict[str, list[dict]]]:
        # Run Potts sampling.
        batch["noise"] = None  # No Gaussian noise for sampling.
        return self.denoiser.potts_sample(batch, sampling_inputs)

    @torch.no_grad()
    def score_samples(self, batch: dict[str, TensorType["b ..."]], sampling_inputs: dict[str, Any]) -> dict[str, Any]:
        """
        Score samples using Potts parameters computed from input backbones.
        """
        batch["noise"] = None

        potts_decoder_aux, batch, sampling_inputs = self.denoiser.compute_potts_params(
            batch, sampling_inputs=sampling_inputs
        )

        # Get sequences.
        res_type_to_score = batch["restype"]
        S = res_type_to_score.argmax(dim=-1)

        # Score sequences for each sample.
        U, U_i = compute_potts_energy(
            S,
            potts_decoder_aux["h"],
            potts_decoder_aux["J"],
            potts_decoder_aux["edge_idx"],
            potts_decoder_aux["mask_i"],
            potts_decoder_aux["mask_ij"],
        )

        # Store outputs.
        id_to_aux = {
            batch["example_id"][bi]: {
                "atom_array": batch["atom_array"][bi],
                "U": U[bi].cpu().item(),
                "U_i": U_i[bi][batch["token_pad_mask"][bi].bool()].cpu(),
            }
            for bi in range(len(batch["example_id"]))
        }

        return id_to_aux


def get_denoiser(cfg: DictConfig, sigma_data: TensorType[(), float]) -> BaseSeqDenoiser:
    """
    Get the denoiser specified in the config.
    """
    if cfg.name == "atom_mpnn":
        return AtomMPNNDenoiser(cfg, sigma_data)
    else:
        raise ValueError(f"Unknown denoiser: {cfg.name}")
