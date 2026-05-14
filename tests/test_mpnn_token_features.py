import torch
from omegaconf import OmegaConf

from caliby.model.seq_denoiser.denoisers.seq_design.atom_mpnn import TokenFeatures


def _make_token_features(use_multichain_encoding: bool) -> TokenFeatures:
    cfg = OmegaConf.create(
        {
            "ca_only": True,
            "k_neighbors": 4,
            "num_rbf": 16,
            "num_positional_embeddings": 16,
            "edge_n_channel": 16,
            "use_multichain_encoding": use_multichain_encoding,
        }
    )
    return TokenFeatures(cfg, model_version=1)


def _make_batch() -> dict[str, torch.Tensor]:
    coords = torch.zeros((1, 4, 14, 3), dtype=torch.float32)
    coords[0, :, 1, 0] = torch.arange(4, dtype=torch.float32)
    return {
        "atom14_coords": coords,
        "atom14_cond_mask": torch.ones((1, 4, 14), dtype=torch.bool),
        "noise": torch.zeros_like(coords),
        "residue_index": torch.tensor([[0, 1, 0, 1]], dtype=torch.int64),
        "asym_id": torch.tensor([[0, 0, 1, 1]], dtype=torch.int64),
        "token_bonds": torch.zeros((1, 4, 4), dtype=torch.float32),
        "is_ligand": torch.zeros((1, 4), dtype=torch.bool),
    }


def _capture_positional_inputs(model: TokenFeatures, batch: dict[str, torch.Tensor]):
    captured = {}

    def fake_embeddings(offset, mask):
        captured["offset"] = offset.detach().clone()
        captured["mask"] = mask.detach().clone()
        return torch.zeros((*offset.shape, model.num_positional_embeddings), dtype=torch.float32)

    model.embeddings.forward = fake_embeddings
    _, E_idx, _ = model(batch)
    return captured, E_idx


def _gather_expected_offsets(residue_index: torch.Tensor, E_idx: torch.Tensor) -> torch.Tensor:
    pair_offsets = residue_index[:, :, None] - residue_index[:, None, :]
    return pair_offsets[:, torch.arange(residue_index.shape[1]).unsqueeze(-1), E_idx[0]]


def test_non_multichain_token_features_gap_residue_indices_between_chains():
    model = _make_token_features(use_multichain_encoding=False)
    batch = _make_batch()

    captured, E_idx = _capture_positional_inputs(model, batch)

    effective_residue_index = torch.tensor([[0, 1, 102, 103]], dtype=torch.int64)
    assert torch.equal(captured["offset"], _gather_expected_offsets(effective_residue_index, E_idx))
    assert captured["mask"].eq(1).all()


def test_multichain_token_features_keep_raw_residue_indices_and_chain_mask():
    model = _make_token_features(use_multichain_encoding=True)
    batch = _make_batch()

    captured, E_idx = _capture_positional_inputs(model, batch)

    assert torch.equal(captured["offset"], _gather_expected_offsets(batch["residue_index"], E_idx))
    source_asym = batch["asym_id"][0, :, None].expand_as(E_idx[0])
    neighbor_asym = batch["asym_id"][0][E_idx[0]]
    assert captured["mask"][0][source_asym != neighbor_asym].eq(0).all()
