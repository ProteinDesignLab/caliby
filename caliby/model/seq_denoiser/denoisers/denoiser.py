from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torchtyping import TensorType

from caliby.checkpoint_utils import repair_state_dict


class BaseSeqDenoiser(nn.Module, ABC):
    """
    Generic sequence denoiser.
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(
        self,
        x_noised: TensorType["b n a 3", float],
        aatype_noised: Optional[TensorType["b n", int]],
        t: TensorType["b", float],  # possibly a tuple (t_seq, t_scn)
        residue_index: TensorType["b n", int],
        seq_mask: TensorType["b n", float],
        cond_labels_in: dict[str, TensorType["b", int]] = {},
        aux_inputs: Optional[
            Dict
        ] = None,  # stores additional inputs for the model (different for training and sampling)
        is_sampling: bool = False,
    ) -> tuple[
        TensorType["b n a 3", float],  # x1 pred
        TensorType["b n", int],  # aatype pred
        dict[str, TensorType["b ..."]],  # aux_preds
    ]:
        pass

    def setup(self, **kwargs):
        """
        Setup function is only called at the start of training. Useful for loading pre-trained modules only at the start of training.
        """
        pass

    def load_pretrained_module(self, ckpt_path: str, module_name: str, freeze: bool = False):
        print(f"Loading pre-trained {module_name} from {ckpt_path}...")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = repair_state_dict(ckpt["state_dict"])
        state_dict = {
            k.replace(f"model.denoiser.{module_name}.", ""): v
            for k, v in state_dict.items()
            if f"model.denoiser.{module_name}" in k
        }  # remove module prefix
        load_result = getattr(self, module_name).load_state_dict(state_dict, strict=False)

        # Warn about missing or unexpected keys
        if load_result.missing_keys:
            print("Missing keys:")
            for key in load_result.missing_keys:
                print(f"  {key}")

        if load_result.unexpected_keys:
            print("Unexpected keys:")
            for key in load_result.unexpected_keys:
                print(f"  {key}")

        if freeze:
            print(f"Freezing the {module_name}...")
            # Freeze the module
            for param in getattr(self, module_name).parameters():
                param.requires_grad = False
