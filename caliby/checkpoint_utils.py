from pathlib import Path
from typing import Any, Union

import lightning as L
import torch
from omegaconf import DictConfig, OmegaConf, open_dict


def migrate_legacy_state_dict(state_dict: dict[str, Any], cfg: DictConfig) -> dict[str, Any]:
    """Remap legacy checkpoint keys to their current names."""
    state_dict_migrations: dict[str, str] = {}  # old_prefix -> new_prefix

    # Legacy config settings.
    if cfg.model.denoiser.get("model_version", 0) in [0, 1]:
        state_dict_migrations = {
            "denoiser.atom_mpnn.": "denoiser.mpnn.",
        }

    # Migrate state dict keys.
    out = {}
    for key, value in state_dict.items():
        new_key = key
        for old_prefix, new_prefix in state_dict_migrations.items():
            if old_prefix in key:
                new_key = key.replace(old_prefix, new_prefix, 1)
                break
        out[new_key] = value
    return out


def migrate_legacy_cfg(cfg: DictConfig) -> DictConfig:
    """Backfill missing keys and apply renames to legacy configs."""

    # Config key renames: (old_dotpath, new_dotpath).
    cfg_key_renames: list[tuple[str, str]] = []

    # Set configs if keys didn't exist in old configs: dotpath -> default_value.
    cfg_defaults: dict[str, Any] = {}

    # Legacy config settings.
    if cfg.model.denoiser.get("model_version", 0) == 0:
        cfg_defaults = {
            "model.denoiser.model_version": 0,
            "model.denoiser.mpnn.name": "caliby_mpnn",
        }
    elif cfg.model.denoiser.model_version == 1:
        cfg_defaults = {
            "model.denoiser.mpnn.name": "caliby_mpnn",
        }

    # Migrate config.
    _SENTINEL = object()
    with open_dict(cfg):
        # Rename keys.
        for old_path, new_path in cfg_key_renames:
            val = OmegaConf.select(cfg, old_path, default=_SENTINEL)
            if val is not _SENTINEL:
                OmegaConf.update(cfg, new_path, val)

        # Set default values.
        for dotpath, default in cfg_defaults.items():
            if OmegaConf.select(cfg, dotpath, default=_SENTINEL) is _SENTINEL:
                parts = dotpath.rsplit(".", 1)
                parent_path, key = parts[0], parts[1]
                parent = OmegaConf.select(cfg, parent_path, default=_SENTINEL)
                if parent is not _SENTINEL:
                    OmegaConf.update(parent, key, default)
    return cfg


def get_cfg_from_ckpt(
    ckpt_path: str, return_as_dict: bool = False
) -> tuple[Union[DictConfig, dict[str, Any]], dict[str, Any]]:
    """
    Load the config directly from the cfg arg passed into the model during training.

    Also returns the model checkpoint.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg_dict = ckpt["hyper_parameters"]["cfg"]
    cfg = OmegaConf.create(cfg_dict)

    if return_as_dict:
        return cfg_dict, ckpt
    return cfg, ckpt


def resume_ckpt_cfg(current_cfg: DictConfig) -> DictConfig:
    """
    Handles logic for obtaining a config for resuming training from a checkpoint.

    resume_opts.use_current_cfg: If True, the current config is used instead of the checkpoint config
    resume_opts.overrides: a dict of overrides to apply to the cfg

    Parameters:
    - current_cfg (DictConfig): The active configuration.

    Returns:
    - DictConfig: The updated configuration for resuming training
    """
    resume_opts = current_cfg.resume

    lit_model_cfg, lit_model_ckpt = get_cfg_from_ckpt(resume_opts.ckpt_path)

    # use the current cfg instead of the checkpoint cfg
    cfg = current_cfg if resume_opts.use_current_cfg else lit_model_cfg

    with open_dict(cfg):
        # retain resume info in new cfg
        cfg.resume = resume_opts

        if ("train" in resume_opts.overrides) and (resume_opts.overrides.train.get("compile_model") is False):
            # get rid of _orig_mod. prefix in saved compiled models
            lit_model_ckpt["state_dict"] = repair_state_dict(lit_model_ckpt["state_dict"])

        # if trying to override optimizer, throw an error
        if "optim" in resume_opts.overrides:
            raise ValueError(
                "Cannot override optimizer when resuming training, we probably have to switch away from pure Lightning to do this better..."
            )

    # apply specific overrides
    cfg = OmegaConf.merge(cfg, OmegaConf.create(resume_opts.overrides))

    return cfg, lit_model_ckpt


def repair_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
    """
    Repair the state dict to avoid issues with loading the model checkpoint due to torch.compile().

    https://github.com/pytorch/pytorch/issues/101107

    Parameters:
    - state_dict (dict[str, Any]): The model state dict.

    Returns:
    - dict[str, Any]: The repaired state dict.
    """
    pairings = [(src_key, src_key.replace("_orig_mod.", "")) for src_key in state_dict.keys()]
    out_state_dict = {}
    for src_key, dest_key in pairings:
        out_state_dict[dest_key] = state_dict[src_key]

    return out_state_dict


class EMATrackerCheckpoint(L.Callback):
    def __init__(self, save_dir, save_freq_steps=None, save_freq_epochs=None):
        """
        Args:
            save_dir (str): Directory where EMA tracker checkpoints will be saved.
            save_freq_steps (int): Save EMA tracker every N steps.
            save_freq_epochs (int): Save EMA tracker every N epochs.
        """
        super().__init__()
        self.save_dir = save_dir
        self.save_freq_steps = save_freq_steps
        self.save_freq_epochs = save_freq_epochs
        assert (save_freq_steps is not None) or (
            save_freq_epochs is not None
        ), "Either save_freq_steps or save_freq_epochs must be provided."

    def setup(self, trainer, pl_module, stage):
        Path(self.save_dir).mkdir(parents=True, exist_ok=True)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.save_freq_steps is not None:
            global_step = trainer.global_step
            if (global_step > 0) and (global_step % self.save_freq_steps == 0):
                self.save_ema_tracker(trainer, pl_module, global_step=global_step)

    def on_train_epoch_end(self, trainer, pl_module):
        if self.save_freq_epochs is not None:
            current_epoch = trainer.current_epoch
            if (current_epoch > 0) and (current_epoch % self.save_freq_epochs) == 0:
                self.save_ema_tracker(trainer, pl_module, epoch=current_epoch)

    def save_ema_tracker(self, trainer, pl_module, global_step=None, epoch=None):
        ema_state = pl_module.ema_tracker.state_dict()
        # Construct the filename
        if global_step is not None:
            filename = f"ema_tracker_step_{global_step}.ckpt"
        elif epoch is not None:
            filename = f"ema_tracker_epoch_{epoch}.ckpt"
        else:
            filename = "ema_tracker.ckpt"

        # Prepare the checkpoint dictionary
        checkpoint = {
            'ema_state': ema_state,
            'global_step': trainer.global_step,
            'epoch': trainer.current_epoch,
        }

        # Save the checkpoint
        save_path = f"{self.save_dir}/{filename}"
        torch.save(checkpoint, save_path)

        # Optionally, log that the EMA tracker has been saved
        pl_module.log("ema_tracker_saved", True, on_step=False, on_epoch=True, sync_dist=True)
