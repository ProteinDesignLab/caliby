import torch


def unbatch_feats(batched: dict[str, torch.Tensor]) -> list[dict[str, torch.Tensor]]:
    """
    Turn dict[B, …] → list[dict[…]]  (keeps non‑tensor, non-list entries verbatim).
    """
    B = next(v for v in batched.values() if isinstance(v, torch.Tensor)).shape[0]
    out: list[dict[str, torch.Tensor]] = []
    for b in range(B):
        slice_b: dict[str, torch.Tensor] = {}
        for k, v in batched.items():
            if isinstance(v, torch.Tensor):
                slice_b[k] = v[b]
            elif isinstance(v, list):
                slice_b[k] = v[b]
            else:
                slice_b[k] = v
        out.append(slice_b)
    return out


def slice_feats(feats: dict[str, torch.Tensor], indices: slice | list[int]) -> dict[str, torch.Tensor]:
    """
    Slice a dictionary of features by a list of indices.
    Keeps non-tensor, non-list entries verbatim.
    """
    sliced_feats = {}
    for k, v in feats.items():
        if isinstance(v, torch.Tensor):
            sliced_feats[k] = v[indices]
        elif isinstance(v, list):
            if isinstance(indices, slice):
                sliced_feats[k] = v[indices]
            else:
                sliced_feats[k] = [v[i] for i in indices]
        else:
            sliced_feats[k] = v
    return sliced_feats
