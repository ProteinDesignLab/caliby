"""Helpers for converting between AF3 token ids and the Potts token alphabet.

Sequences passed into Potts utilities use AF3 token ids. Potts parameters and
Potts-local outputs use the reduced alphabet defined by `const.POTTS_TOKENS`.
These helpers keep the vocabulary conversion logic in one place.
"""

import torch

import caliby.data.const as const


def _potts_state_idx(device: torch.device) -> torch.Tensor:
    """Return AF3 token ids kept in `const.POTTS_TOKENS` as a `(C_potts,)` tensor."""
    return torch.as_tensor(const.POTTS_AF3_TOKEN_IDXS, device=device, dtype=torch.long)


def uses_full_af3_state_space(h: torch.Tensor, J: torch.Tensor) -> bool:
    """Return `True` when `h` and `J` use the full AF3 token axis.

    Args:
        h: Field tensor with shape `(..., C)`.
        J: Coupling tensor with shape `(..., C, C)`.
    """
    n_states_full = const.AF3_ENCODING.n_tokens
    return h.shape[-1] == n_states_full and J.shape[-2:] == (n_states_full, n_states_full)


def uses_potts_state_space(h: torch.Tensor, J: torch.Tensor) -> bool:
    """Return `True` when `h` and `J` already use `const.POTTS_TOKENS`.

    Args:
        h: Field tensor with shape `(..., C)`.
        J: Coupling tensor with shape `(..., C, C)`.
    """
    n_states_potts = len(const.POTTS_TOKENS)
    return h.shape[-1] == n_states_potts and J.shape[-2:] == (n_states_potts, n_states_potts)


def uses_full_af3_token_tensor(x: torch.Tensor) -> bool:
    """Return `True` when the last axis of `x` has size `const.AF3_ENCODING.n_tokens`."""
    return x.shape[-1] == const.AF3_ENCODING.n_tokens


def uses_potts_token_tensor(x: torch.Tensor) -> bool:
    """Return `True` when the last axis of `x` has size `len(const.POTTS_TOKENS)`."""
    return x.shape[-1] == len(const.POTTS_TOKENS)


def removed_af3_state_idx(device: torch.device) -> torch.Tensor:
    """Return AF3 token ids removed from `const.POTTS_TOKENS` as a `(C_removed,)` tensor."""
    mask = torch.ones(const.AF3_ENCODING.n_tokens, device=device, dtype=torch.bool)
    mask[_potts_state_idx(device)] = False
    return mask.nonzero(as_tuple=False).flatten()


def slice_af3_token_tensor(x: torch.Tensor) -> torch.Tensor:
    """Slice the last axis of an AF3-token tensor down to `const.POTTS_TOKENS`.

    Args:
        x: Tensor with shape `(..., C_af3)`.

    Returns:
        Tensor with shape `(..., C_potts)`.
    """
    return x.index_select(-1, _potts_state_idx(x.device))


def slice_potts_params(h: torch.Tensor, J: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Slice Potts parameters from AF3 token space to `const.POTTS_TOKENS`.

    Args:
        h: Field tensor with shape `(..., C)`.
        J: Coupling tensor with shape `(..., C, C)`.

    Returns:
        `h_reduced` with shape `(..., C_potts)` and `J_reduced` with shape
        `(..., C_potts, C_potts)`. If `h/J` are already reduced, they are
        returned unchanged.
    """
    if uses_potts_state_space(h, J):
        return h, J
    if not uses_full_af3_state_space(h, J):
        raise ValueError(f"Unexpected Potts state space: h={h.shape}, J={J.shape}")
    potts_state_idx = _potts_state_idx(h.device)
    h = h.index_select(-1, potts_state_idx)
    J = J.index_select(-2, potts_state_idx).index_select(-1, potts_state_idx)
    return h, J


def expand_potts_token_tensor(x: torch.Tensor, fill_value: float | int = 0.0) -> torch.Tensor:
    """Expand a reduced-token tensor back into AF3 token space.

    Args:
        x: Tensor with shape `(..., C_potts)`.
        fill_value: Value used for AF3 states that are not part of
            `const.POTTS_TOKENS`.

    Returns:
        Tensor with shape `(..., C_af3)`.
    """
    potts_state_idx = _potts_state_idx(x.device)
    x_full = x.new_full((*x.shape[:-1], const.AF3_ENCODING.n_tokens), fill_value)
    x_full.index_copy_(-1, potts_state_idx, x)
    return x_full


def map_af3_indices_to_potts(
    S: torch.Tensor, fill_removed_with: str = const.UNKNOWN_AA
) -> torch.Tensor:
    """Map AF3 token ids to Potts token ids elementwise.

    Args:
        S: Integer tensor of AF3 token ids with shape `(...,)`.
        fill_removed_with: AF3 token name to use when `S` contains a token that
            is not part of `const.POTTS_TOKENS`.

    Returns:
        Integer tensor `S_reduced` with shape `(...,)` indexed by
        `const.POTTS_TOKENS`.
    """
    af3_to_potts = torch.as_tensor(const.AF3_TO_POTTS_TOKEN_IDX, device=S.device, dtype=torch.long)
    S_potts = af3_to_potts[S]
    fill_idx = af3_to_potts[const.AF3_ENCODING.token_to_idx[fill_removed_with]]
    return torch.where(S_potts >= 0, S_potts, torch.full_like(S_potts, fill_idx))


def map_potts_indices_to_af3(S: torch.Tensor) -> torch.Tensor:
    """Map Potts token ids back to AF3 token ids.

    Args:
        S: Integer tensor with shape `(...,)` indexed by `const.POTTS_TOKENS`.

    Returns:
        Integer tensor with the same shape as `S`, now using AF3 token ids.
    """
    return _potts_state_idx(S.device)[S]


def check_removed_af3_states_are_unused(
    S: torch.Tensor | None,
    *,
    mask_i: torch.Tensor | None,
    mask_S: torch.Tensor | None = None,
):
    """Check that AF3-only states are masked out before Potts reduction.

    Args:
        S: AF3-indexed sequence tensor with shape `(B, N)`, or `None`.
        mask_i: Node mask with shape `(B, N)`, or `None`.
        mask_S: Optional sampling mask with shape `(B, N, C_af3)` when present.
    """
    device = S.device if S is not None else mask_i.device if mask_i is not None else mask_S.device
    removed_state_idx = removed_af3_state_idx(device)
    if removed_state_idx.numel() == 0:
        return

    if S is not None:
        active_mask = mask_i.bool() if mask_i is not None else torch.ones_like(S, dtype=torch.bool)
        removed_in_S = torch.isin(S, removed_state_idx)
        if torch.any(active_mask & removed_in_S):
            raise ValueError("AF3 states outside the Potts vocabulary must be masked out before Potts reduction.")
    else:
        if mask_i is None:
            raise ValueError("mask_i is required when validating sampling masks without an explicit sequence.")
        active_mask = mask_i.bool()

    if mask_S is None or mask_S.dim() != 3:
        return

    removed_allowed = mask_S.index_select(-1, removed_state_idx)
    active_mask = active_mask[..., None].expand_as(removed_allowed)
    if torch.any(active_mask & removed_allowed.bool()):
        raise ValueError("AF3 states outside the Potts vocabulary must be banned during Potts sampling.")


def normalize_potts_inputs(
    *,
    S: torch.Tensor | None,
    h: torch.Tensor,
    J: torch.Tensor,
    mask_i: torch.Tensor | None,
    mask_S: torch.Tensor | None = None,
) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Convert Potts inputs to reduced-state tensors.

    Args:
        S: Optional AF3-indexed sequence tensor with shape `(B, N)`.
        h: Field tensor with shape `(B, N, C)` where `C` is either
            `const.AF3_ENCODING.n_tokens` or `len(const.POTTS_TOKENS)`.
        J: Coupling tensor with shape `(B, N, K, C, C)` using the same state
            dimension as `h`.
        mask_i: Optional node mask with shape `(B, N)`.
        mask_S: Optional sampling mask. A 2D mask has shape `(B, N)`. A 3D mask
            has shape `(B, N, C)` and may use either the AF3 token axis or the
            reduced Potts token axis.

    Returns:
        `S_reduced`: `S` mapped into Potts token ids with shape `(B, N)`, or
            `None` if `S` was not provided.
        `h_reduced`: Tensor with shape `(B, N, C_potts)`.
        `J_reduced`: Tensor with shape `(B, N, K, C_potts, C_potts)`.
        `mask_S_reduced`: The input sampling mask with the same rank as `mask_S`.
            A 3D mask is sliced to shape `(B, N, C_potts)`. A 2D mask is returned
            unchanged.
    """
    if uses_full_af3_state_space(h, J):
        h_reduced, J_reduced = slice_potts_params(h, J)
    elif uses_potts_state_space(h, J):
        h_reduced, J_reduced = h, J
    else:
        raise ValueError(f"Unexpected Potts state space: h={h.shape}, J={J.shape}")

    if S is not None:
        check_removed_af3_states_are_unused(S, mask_i=mask_i)
        S_reduced = map_af3_indices_to_potts(S)
    else:
        S_reduced = None

    if mask_S is not None and mask_S.dim() == 3:
        if uses_full_af3_token_tensor(mask_S):
            check_removed_af3_states_are_unused(S=None, mask_i=mask_i, mask_S=mask_S)
            mask_S_reduced = slice_af3_token_tensor(mask_S)
        elif not uses_potts_token_tensor(mask_S):
            raise ValueError(f"Unexpected sampling mask state space: mask_S={mask_S.shape}")
        else:
            mask_S_reduced = mask_S
    else:
        mask_S_reduced = mask_S

    return S_reduced, h_reduced, J_reduced, mask_S_reduced


def slice_potts_decoder_aux(potts_decoder_aux: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Slice `potts_decoder_aux["h"]` and `potts_decoder_aux["J"]` to Potts space.

    Args:
        potts_decoder_aux: Dictionary containing at least
            `h` with shape `(B, N, C)` and `J` with shape `(B, N, K, C, C)`.

    Returns:
        Copy of `potts_decoder_aux` where `h/J` use `const.POTTS_TOKENS`.
    """
    h, J = slice_potts_params(potts_decoder_aux["h"], potts_decoder_aux["J"])
    return {
        **potts_decoder_aux,
        "h": h,
        "J": J,
    }


def expand_potts_decoder_aux(potts_decoder_aux: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Expand reduced `potts_decoder_aux["h"]` and `potts_decoder_aux["J"]` to AF3 space."""
    if uses_full_af3_state_space(potts_decoder_aux["h"], potts_decoder_aux["J"]):
        return potts_decoder_aux
    if not uses_potts_state_space(potts_decoder_aux["h"], potts_decoder_aux["J"]):
        raise ValueError(
            f"Unexpected Potts state space: h={potts_decoder_aux['h'].shape}, J={potts_decoder_aux['J'].shape}"
        )

    potts_state_idx = _potts_state_idx(potts_decoder_aux["h"].device)
    n_states_full = const.AF3_ENCODING.n_tokens

    h_full = potts_decoder_aux["h"].new_zeros(*potts_decoder_aux["h"].shape[:-1], n_states_full)
    h_full.index_copy_(-1, potts_state_idx, potts_decoder_aux["h"])

    J_last_full = potts_decoder_aux["J"].new_zeros(*potts_decoder_aux["J"].shape[:-1], n_states_full)
    J_last_full.index_copy_(-1, potts_state_idx, potts_decoder_aux["J"])
    J_full = potts_decoder_aux["J"].new_zeros(*potts_decoder_aux["J"].shape[:-2], n_states_full, n_states_full)
    J_full.index_copy_(-2, potts_state_idx, J_last_full)

    return {
        **potts_decoder_aux,
        "h": h_full,
        "J": J_full,
    }
