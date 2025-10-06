import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from torchtyping import TensorType

import caliby.model.seq_denoiser.denoisers.seq_design.potts as potts
from caliby.data import const

logger = logging.getLogger(__name__)


class SDLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.task = cfg.task
        self.use_seq_pred = self.task in ["seq_des"]

        # Parse loss_weights
        self.loss_weights = {}

        for k in cfg.loss_weights.keys():
            if isinstance(cfg.loss_weights[k], DictConfig):
                for k2 in cfg.loss_weights[k].keys():
                    self.loss_weights[f"{k}/{k2}"] = cfg.loss_weights[k][k2]
            else:
                self.loss_weights[k] = cfg.loss_weights[k]

        # Define losses based on task
        if self.task == "seq_des":
            self.loss_keys = {"seq_loss", "potts_composite_loss"}
        else:
            raise ValueError(f"Unrecognized task: {self.task}")

    def forward(self, outputs, batch, eval_seq: bool = True, eval_total: bool = True, return_aux: bool = False):
        """
        Compute losses for the sequence denoiser
        """
        aux = {}  # losses
        aux_monitor = {}  # monitor other metrics that do not contribute to the loss

        if self.use_seq_pred and eval_seq:
            # compute sequence loss from sequence design module
            target_restype = batch["restype"].argmax(dim=-1)
            seq_loss_mask = outputs["token_exists_mask"] * (
                1 - outputs["seq_cond_mask"]
            )  # compute loss only on masked tokens
            seq_loss_mask = seq_loss_mask * (
                target_restype != const.AF3_ENCODING.token_to_idx["UNK"]
            )  # mask out UNK tokens from loss

            # DEBUG: ensure that we're only computing over protein tokens
            if (~batch["is_protein"][seq_loss_mask.bool()]).any():
                logger.warning("WARNING: seq_loss is being computed over non-protein tokens")

            aux["seq_loss"] = masked_cross_entropy(
                outputs["seq_logits"], target_restype, seq_loss_mask, seq_loss_cfg=self.cfg.seq_loss
            )
            aux_monitor["seq_acc"] = (
                masked_seq_accuracy(outputs["seq_logits"], target_restype, seq_loss_mask).mean().detach().clone()
            )

            if outputs.get("potts_decoder_aux") is not None:
                potts_decoder_aux = outputs["potts_decoder_aux"]
                aux["potts_composite_loss"] = potts_composite_loss(
                    target_restype, potts_decoder_aux, self.cfg.potts.label_smoothing, self.cfg.potts.per_token_avg
                )

        # Aggregate losses
        total_loss = 0
        for loss_name, loss in aux.items():
            aux[loss_name] = loss.mean().detach().clone()

            # Average over batch
            loss = loss.mean()

            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"Loss {loss_name} is NaN or Inf, skipping...")
                logger.warning(f"Examples: {batch['example_id']}")
                loss = loss.new_tensor(0.0, requires_grad=True)

            if eval_total:
                if loss_name in self.loss_keys:
                    # Only allow losses that are in the loss_keys to contribute to the total loss
                    total_loss += loss * self.loss_weights[loss_name]  # apply manual per-loss loss weighting

        if eval_total:
            aux["total_loss"] = total_loss

        # Monitor other metrics
        aux.update(aux_monitor)

        if return_aux:
            return total_loss, aux

        return total_loss


def masked_cross_entropy(
    logits: TensorType["b n c", float],
    target: TensorType["b n", int],
    mask: TensorType["b n", float],
    seq_loss_cfg: DictConfig,
) -> TensorType["b", float]:
    """
    Compute cross entropy loss with masking.

    seq_loss_cfg has the following keys:
    - label_smoothing: float, label smoothing factor
    - per_token_avg: bool, whether to average loss per token (false will divide by fixed_size)
    """
    n_classes = const.AF3_ENCODING.n_tokens
    target_oh = F.one_hot(target, num_classes=n_classes).float()

    # Unpack seq_loss_cfg
    label_smoothing = seq_loss_cfg.label_smoothing
    per_token_avg = seq_loss_cfg.per_token_avg

    # Label smoothing
    target_oh = target_oh + label_smoothing / n_classes
    target_oh = target_oh / target_oh.sum(dim=-1, keepdim=True)

    # Compute cross entropy loss
    logprobs = F.log_softmax(logits, dim=-1)
    cel = -(logprobs * target_oh).sum(dim=-1)

    if per_token_avg:
        # average loss per token
        loss = (cel * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1e-8)
    else:
        # divide by constant N to get loss on roughly the same scale as per_token_avg
        N = mask.shape[1]
        loss = (cel * mask).sum(dim=-1) / N

    return loss


def masked_seq_accuracy(
    logits: TensorType["b n k", float], target: TensorType["b n", int], mask: TensorType["b n", float]
) -> TensorType["b", float]:
    """
    Compute sequence accuracy with masking.
    """
    pred = logits.argmax(dim=-1)
    correct = (pred == target).float()
    return (correct * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1e-8)


def potts_composite_loss(
    S: TensorType["b n", int],
    potts_decoder_aux: dict[str, TensorType["b ...", float]],
    label_smoothing: float,
    per_token_avg: bool,
) -> TensorType["b", float]:
    # Log composite likelihood
    logp_ij, mask_p_ij = potts.log_composite_likelihood(
        S,
        potts_decoder_aux["h"],
        potts_decoder_aux["J"],
        potts_decoder_aux["edge_idx"],
        potts_decoder_aux["mask_i"],
        potts_decoder_aux["mask_ij"],
        smoothing_alpha=label_smoothing,
    )

    # Map into approximate local likelihoods
    logp_i = (
        potts_decoder_aux["mask_i"]
        * torch.sum(mask_p_ij * logp_ij, dim=-1)
        / (2.0 * torch.sum(mask_p_ij, dim=-1) + 1e-3)
    )

    # Get loss per sample
    mask = potts_decoder_aux["mask_i"]
    if per_token_avg:
        # average loss per token
        loss = (-logp_i * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1e-8)
    else:
        # divide by constant N to get loss on roughly the same scale as per_token_avg
        N = mask.shape[1]
        loss = -(logp_i * mask).sum(dim=-1) / N

    return loss
