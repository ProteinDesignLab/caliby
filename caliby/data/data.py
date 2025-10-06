import torch
from torchtyping import TensorType

from caliby.data import const


def torch_kabsch(a: TensorType["b n x"], b: TensorType["b n x"]) -> TensorType["b x x"]:
    """
    get alignment matrix for two sets of coordinates using PyTorch

    adapted from: https://github.com/sokrypton/ColabDesign/blob/ed4b01354928b60cd1347f570e9b248f78f11c6d/colabdesign/shared/protein.py#L128
    """
    with torch.autocast(device_type=a.device.type, enabled=False):
        ab = a.transpose(-1, -2) @ b
        u, s, vh = torch.linalg.svd(ab, full_matrices=False)
        flip = torch.det(u @ vh) < 0
        u_ = torch.where(flip, -u[..., -1].T, u[..., -1].T).T
    u = torch.cat([u[..., :-1], u_[..., None]], dim=-1)
    return u @ vh


def torch_rmsd_weighted(
    a: TensorType["b n x", float],
    b: TensorType["b n x", float],
    weights: TensorType["b n", float] | None,
    return_aligned: bool = False,
) -> TensorType["b", float]:
    """
    Compute weighted RMSD of coordinates after weighted alignment. Batched.

    For masked RMSD, set weights to 0 for masked atoms.

    Aligns a to b using Kabsch algorithm, then computes RMSD.
    If return_aligned is True, returns the aligned structures as well.

    Adapted from: https://github.com/sokrypton/ColabDesign/blob/main/colabdesign/af/loss.py#L445
    """
    assert a.dim() == 3 and b.dim() == 3, "Input tensors must be 3D (batch, num_atoms, 3)"

    if weights is None:
        weights = torch.ones(a.shape[:-1], device=a.device, dtype=a.dtype)
    weights = weights / weights.sum(dim=-1, keepdim=True)  # normalize weights

    # Align
    W = weights[..., None]
    a_mu = (a * W).sum(dim=-2, keepdim=True)
    b_mu = (b * W).sum(dim=-2, keepdim=True)

    R = torch_kabsch((a - a_mu) * W, b - b_mu)
    aligned_a = (a - a_mu) @ R + b_mu

    weighted_msd = (W * ((aligned_a - b) ** 2)).sum(dim=(-1, -2))
    weighted_rmsd = torch.sqrt(weighted_msd + 1e-8)

    if return_aligned:
        return weighted_rmsd, (aligned_a, b)
    return weighted_rmsd


def to(obj, device: torch.device):
    """
    Move object to device. Source: https://github.com/pytorch/pytorch/issues/69431
    """
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: to(v, device) for k, v in obj.items()}
    if isinstance(obj, tuple):
        return tuple(to(v, device) for v in obj)
    if isinstance(obj, list):
        return [to(v, device) for v in obj]
    return obj


def get_seq_from_res_type(res_type: TensorType["n k", int]) -> str:
    """
    Get sequence from res_type.
    """
    return "".join([const.prot_token_to_letter[const.tokens[x]] for x in res_type.argmax(dim=-1)])


def batched_gather(data, inds, dim=0, no_batch_dims=0):
    ranges = []
    for i, s in enumerate(data.shape[:no_batch_dims]):
        r = torch.arange(s)
        r = r.view(*(*((1,) * i), -1, *((1,) * (len(inds.shape) - i - 1))))
        ranges.append(r)

    remaining_dims = [slice(None) for _ in range(len(data.shape) - no_batch_dims)]
    remaining_dims[dim - no_batch_dims if dim >= 0 else dim] = inds
    ranges.extend(remaining_dims)
    return data[ranges]
