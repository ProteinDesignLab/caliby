from functools import partial

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.nn import functional as F
from torchtyping import TensorType

import caliby.data.const as const
import caliby.model.seq_denoiser.denoisers.seq_design.potts as potts
from caliby.data.data import batched_gather
from caliby.model.seq_denoiser.denoisers.seq_design.mpnn_utils import (
    cat_neighbors_nodes,
    gather_edges,
    gather_nodes,
)


class CalibyMPNN(nn.Module):
    """MPNN model for Caliby sequence design."""

    def __init__(self, cfg: DictConfig, model_version: int = 0):
        super().__init__()
        self.cfg = cfg
        self.model_version = model_version
        self.task = cfg.get("task", "seq_des")
        self.node_features = cfg.n_channel
        self.edge_features = cfg.n_channel
        self.hidden_dim = cfg.n_channel
        self.num_encoder_layers = cfg.n_layers
        self.num_decoder_layers = cfg.n_layers
        self.k_neighbors = cfg.k_neighbors
        self.n_tokens = const.AF3_ENCODING.n_tokens

        self.token_features = TokenFeatures(cfg.token_features, model_version=self.model_version)
        self.W_e = nn.Linear(self.edge_features, self.hidden_dim, bias=False)
        self.W_s = nn.Linear(self.n_tokens, self.hidden_dim, bias=False)
        self.decoder_in = self.hidden_dim * 3  # concat of h_E, h_S, h_V

        self.dropout = nn.Dropout(cfg.dropout_p)

        # Encoder layers
        self.encoder_layers = nn.ModuleList(
            [
                EncLayer(self.hidden_dim, self.hidden_dim * 2, dropout=cfg.dropout_p)
                for _ in range(self.num_encoder_layers)
            ]
        )

        # Decoder layers
        # No need to update edges in the last layer for sidechain packing.
        update_edges_in_last_layer = self.task not in ["scn_pack"]
        self.decoder_layers = nn.ModuleList(
            [
                DecLayer(
                    self.hidden_dim,
                    self.decoder_in,
                    dropout=cfg.dropout_p,
                    update_edges=update_edges_in_last_layer if (i == self.num_decoder_layers - 1) else True,
                )
                for i in range(self.num_decoder_layers)
            ]
        )

        # Potts decoder
        self.use_potts = cfg.potts.use_potts and (self.task != "scn_pack")
        if self.use_potts:
            self.k_neighbors_potts = cfg.potts.get("k_neighbors_potts", None)
            self.max_dist_potts = cfg.potts.get("max_dist_potts", None)
            self.parameterization = cfg.potts.parameterization
            self.num_factors = cfg.potts.num_factors

            potts_init = partial(
                potts.GraphPotts,
                dim_nodes=self.node_features,
                dim_edges=self.decoder_in,
                num_states=self.n_tokens,
                parameterization=self.parameterization,
                num_factors=self.num_factors,
                symmetric_J=cfg.potts.symmetric_J,
                dropout=cfg.dropout_p,
            )
            self.decoder_S_potts = potts_init()

        # Output layers
        self.W_out = nn.Linear(self.hidden_dim, self.n_tokens, bias=True)

        # Initialize weights
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, batch: dict[str, TensorType["b ..."]]):
        # Get token-level features.
        B, N, C = batch["restype"].shape
        h_V = torch.zeros((B, N, self.node_features), device=batch["restype"].device)

        # First, mask out residues using gap token.
        B, N, C = batch["restype"].shape
        masked = F.one_hot(
            torch.full((B, N), const.AF3_ENCODING.token_to_idx["<G>"], device=batch["restype"].device), num_classes=C
        ).float()
        restype = torch.where(batch["seq_cond_mask"].unsqueeze(-1).bool(), batch["restype"], masked)
        h_S = self.W_s(restype)

        # Build graph and get edge features.
        h_E, E_idx, D_neighbors = self.token_features(batch)

        # Pass through encoder layers.
        h_V = h_V + h_S
        h_E = self.W_e(h_E)
        token_mask = batch["token_exists_mask"]
        token_mask_2d = gather_nodes(token_mask.unsqueeze(-1), E_idx).squeeze(-1)
        token_mask_2d = token_mask.unsqueeze(-1) * token_mask_2d
        for layer in self.encoder_layers:
            h_V, h_E = layer(h_V, h_E, E_idx, token_mask, token_mask_2d)

        # Pass through decoder layers.
        h_ES = cat_neighbors_nodes(h_S, h_E, E_idx)
        h_ESV = cat_neighbors_nodes(h_V, h_ES, E_idx)
        for layer in self.decoder_layers:
            h_V, h_ESV = layer(h_V, h_ESV, token_mask, E_idx)

        # Potts model.
        if self.use_potts:
            if self.max_dist_potts is not None:
                token_mask_2d = token_mask_2d * (
                    D_neighbors <= self.max_dist_potts
                )  # mask out edges that are too far away

            if self.k_neighbors_potts is not None:
                # Truncate to k_neighbors_potts.
                h_ESV = h_ESV[:, :, : self.k_neighbors_potts]
                E_idx = E_idx[:, :, : self.k_neighbors_potts]
                token_mask_2d = token_mask_2d[:, :, : self.k_neighbors_potts]

            h, J = self.decoder_S_potts(h_V, h_ESV, E_idx, token_mask, token_mask_2d)
            potts_decoder_aux = {
                "h": h,
                "J": J,
                "edge_idx": E_idx,
                "mask_i": token_mask,
                "mask_ij": token_mask_2d,
            }

        logits = self.W_out(h_V)

        # Output features.
        mpnn_feature_dict = {"h_V": h_V, "h_ESV": h_ESV, "E_idx": E_idx}
        if self.use_potts:
            mpnn_feature_dict["potts_decoder_aux"] = potts_decoder_aux

        return logits, mpnn_feature_dict


class TokenFeatures(nn.Module):
    def __init__(self, cfg: DictConfig, model_version: int = 0):
        """
        Extract token-level edge features and build KNN graph.
        """
        super().__init__()
        self.cfg = cfg
        self.model_version = model_version

        # Parameters
        self.ca_only = cfg.get("ca_only", True)  # backwards compatibility
        self.k_neighbors = cfg.k_neighbors
        self.num_rbf = cfg.num_rbf
        self.num_positional_embeddings = cfg.num_positional_embeddings
        self.edge_n_channel = cfg.edge_n_channel
        self.use_multichain_encoding = cfg.get("use_multichain_encoding", False)

        # Layers
        self.embeddings = PositionalEncodings(self.num_positional_embeddings)
        if self.model_version == 0:
            num_pairwise_dists = 1 if self.ca_only else const.MAX_NUM_ATOMS**2
        else:
            # 5*5 for backbone + virtual CB; 4*10 for backbone-sidechain pairs
            num_pairwise_dists = 1 if self.ca_only else (5 * 5) + (4 * 10)

        edge_in = self.num_positional_embeddings + self.num_rbf * num_pairwise_dists + 1
        self.edge_embedding = nn.Linear(edge_in, self.edge_n_channel, bias=False)
        self.norm_edges = nn.LayerNorm(self.edge_n_channel)

    def forward(self, batch: dict[str, TensorType["b ..."]]):
        """
        Extract token-level edge features and build KNN graph.
        """
        X, X_cond_mask = batch["atom14_coords"], batch["atom14_cond_mask"]
        X = X + batch["noise"]  # Add noise to coords.
        X = torch.where(X_cond_mask.unsqueeze(-1).bool(), X, X[..., 1:2, :])  # Replace masked atoms with noised CA.
        CA = X[..., 1, :]
        D_neighbors, E_idx = self._dist(CA, X_cond_mask[..., 1].float())

        # Get RBF features
        if self.ca_only:
            RBF_all = self._rbf(D_neighbors)
        else:
            if self.model_version == 0:
                # Backwards compatibility.
                RBF_all = []
                B, N, _, _ = X.shape
                dummy_vals = CA[..., None, :].expand(-1, -1, 23 - 14, -1)
                X = torch.cat([X, dummy_vals], dim=-2)
                for i in range(X.shape[-2]):
                    for j in range(X.shape[-2]):
                        RBF_all.append(self._get_rbf(X[..., i, :], X[..., j, :], E_idx))
                RBF_all = torch.cat(RBF_all, dim=-1)
            else:
                RBF_all = self._get_rbf_all(X, E_idx)

        # Positional encodings
        residue_index = batch["residue_index"]
        offset = residue_index[:, :, None] - residue_index[:, None, :]
        offset = gather_edges(offset[:, :, :, None], E_idx)[:, :, :, 0]  # [B, L, K]

        chain_labels = torch.zeros_like(batch["asym_id"])
        if self.use_multichain_encoding:
            # Only use multichain encoding if the model has been trained with it.
            chain_labels = batch["asym_id"]

        # Find self vs non-self interaction.
        d_chains = ((chain_labels[:, :, None] - chain_labels[:, None, :]) == 0).long()
        E_chains = gather_edges(d_chains[:, :, :, None], E_idx)[:, :, :, 0]
        E_positional = self.embeddings(offset.long(), E_chains)

        # AF3 token_bond feature.
        token_bonds = batch["token_bonds"]

        # Remove polymer-polymer bonds if present.
        token_bonds_mask = batch["is_ligand"]  # [B, L]
        token_bonds_mask = token_bonds_mask[:, :, None] | token_bonds_mask[:, None, :]  # [B, L, L]
        token_bonds = (token_bonds * token_bonds_mask)[..., None]  # [B, L, L, 1]

        token_bonds = gather_edges(token_bonds, E_idx)

        # Concatenate edge features and embed.
        E = torch.cat((E_positional, RBF_all, token_bonds), -1)
        E = self.edge_embedding(E)
        E = self.norm_edges(E)
        return E, E_idx, D_neighbors

    def _dist(self, X, mask, eps=1e-6):
        mask_2D = torch.unsqueeze(mask, 1) * torch.unsqueeze(mask, 2)
        dX = torch.unsqueeze(X, 1) - torch.unsqueeze(X, 2)
        D = mask_2D * torch.sqrt(torch.sum(dX**2, 3) + eps)
        D_max, _ = torch.max(D, -1, keepdim=True)
        D_adjust = D + (1.0 - mask_2D) * D_max
        D_neighbors, E_idx = torch.topk(
            D_adjust, np.minimum(self.k_neighbors, X.shape[1]), dim=-1, sorted=True, largest=False
        )
        return D_neighbors, E_idx

    def _rbf(self, D):
        device = D.device
        D_min, D_max, D_count = 2.0, 22.0, self.num_rbf
        D_mu = torch.linspace(D_min, D_max, D_count, device=device)
        D_mu = D_mu.view([1, 1, 1, -1])
        D_sigma = (D_max - D_min) / D_count
        D_expand = torch.unsqueeze(D, -1)
        RBF = torch.exp(-(((D_expand - D_mu) / D_sigma) ** 2))
        return RBF

    def _get_rbf(self, A, B, E_idx):
        D_A_B = torch.sqrt(torch.sum((A[:, :, None, :] - B[:, None, :, :]) ** 2, -1) + 1e-6)  # [B, L, L]
        D_A_B_neighbors = gather_edges(D_A_B[:, :, :, None], E_idx)[:, :, :, 0]  # [B,L,K]
        RBF_A_B = self._rbf(D_A_B_neighbors)
        return RBF_A_B

    def _get_rbf_all(self, X_all, E_idx):
        """
        Get RBF features for all backbone atom pairs (including a virtual CB)
        and all pairs between backbone and sidechain atoms.
        """
        RBF_all = []

        # Compute virtual CB atom.
        b = X_all[:, :, 1, :] - X_all[:, :, 0, :]
        c = X_all[:, :, 2, :] - X_all[:, :, 1, :]
        a = torch.cross(b, c, dim=-1)
        Cb = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + X_all[:, :, 1, :]

        # Get all backbone atom pairs (including a virtual CB)
        bb_atoms = [X_all[..., i, :] for i in range(4)] + [Cb]  # [N, CA, C, O, virtual CB]
        for i in range(len(bb_atoms)):
            for j in range(len(bb_atoms)):
                RBF_all.append(self._get_rbf(bb_atoms[i], bb_atoms[j], E_idx))

        # Get all pairs between backbone and sidechain atoms.
        for i in range(4):
            for j in range(4, 14):
                RBF_all.append(self._get_rbf(X_all[..., i, :], X_all[..., j, :], E_idx))

        RBF_all = torch.cat(RBF_all, dim=-1)
        return RBF_all


class PositionWiseFeedForward(torch.nn.Module):
    def __init__(self, num_hidden, num_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.W_in = torch.nn.Linear(num_hidden, num_ff, bias=True)
        self.W_out = torch.nn.Linear(num_ff, num_hidden, bias=True)
        self.act = torch.nn.GELU()

    def forward(self, h_V):
        h = self.act(self.W_in(h_V))
        h = self.W_out(h)
        return h


class PositionalEncodings(torch.nn.Module):
    def __init__(self, num_embeddings, max_relative_feature=32):
        super(PositionalEncodings, self).__init__()
        self.num_embeddings = num_embeddings
        self.max_relative_feature = max_relative_feature
        self.linear = torch.nn.Linear(2 * max_relative_feature + 1 + 1, num_embeddings)

    def forward(self, offset, mask):
        d = torch.clip(offset + self.max_relative_feature, 0, 2 * self.max_relative_feature) * mask + (1 - mask) * (
            2 * self.max_relative_feature + 1
        )
        d_onehot = torch.nn.functional.one_hot(d, 2 * self.max_relative_feature + 1 + 1)
        E = self.linear(d_onehot.float())
        return E


class DecLayer(nn.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1, num_heads=None, scale=30, update_edges=True):
        super(DecLayer, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.update_edges = update_edges
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(self.num_hidden)
        self.norm2 = nn.LayerNorm(self.num_hidden)

        self.W1 = nn.Linear(self.num_hidden + num_in, self.num_hidden, bias=True)
        self.W2 = nn.Linear(self.num_hidden, self.num_hidden, bias=True)
        self.W3 = nn.Linear(self.num_hidden, self.num_hidden, bias=True)

        if self.update_edges:
            self.W11 = nn.Linear(num_hidden * 2 + num_in, num_hidden, bias=True)  # nh * 2 for vi AND vj
            self.W12 = nn.Linear(num_hidden, num_hidden, bias=True)
            self.W13 = nn.Linear(num_hidden, num_in, bias=True)  # num_in is hidden dim of edges h_E
            self.norm3 = nn.LayerNorm(num_in)
            self.dropout3 = nn.Dropout(dropout)

        self.act = torch.nn.GELU()
        self.dense = PositionWiseFeedForward(self.num_hidden, num_hidden * 4)

    def forward(self, h_V, h_E, mask_V=None, E_idx=None, mask_attend=None):
        """Parallel computation of full transformer layer"""

        # Concatenate h_V_i to h_E_ij
        h_V_expand = h_V.unsqueeze(-2).expand(-1, -1, h_E.size(-2), -1)
        h_EV = torch.cat([h_V_expand, h_E], -1)
        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))

        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message

        dh = torch.sum(h_message, -2) / self.scale
        h_V = self.norm1(h_V + self.dropout1(dh))

        # Position-wise feedforward
        dh = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout2(dh))

        if mask_V is not None:
            mask_V = mask_V.unsqueeze(-1)
            h_V = mask_V * h_V

        if self.update_edges:
            h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
            h_V_expand = h_V.unsqueeze(-2).expand(-1, -1, h_EV.size(-2), -1)
            h_EV = torch.cat([h_V_expand, h_EV], -1)
            h_message = self.W13(self.act(self.W12(self.act(self.W11(h_EV)))))
            h_E = self.norm3(h_E + self.dropout3(h_message))

        return h_V, h_E


class EncLayer(nn.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1, scale=30, is_last_layer=False):
        super(EncLayer, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.is_last_layer = is_last_layer

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(num_hidden)
        self.norm2 = nn.LayerNorm(num_hidden)

        self.W1 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = torch.nn.GELU()
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

        if not is_last_layer:
            # only initialize if not last layer to avoid unused parameters
            self.W11 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
            self.W12 = nn.Linear(num_hidden, num_hidden, bias=True)
            self.W13 = nn.Linear(num_hidden, num_hidden, bias=True)
            self.norm3 = nn.LayerNorm(num_hidden)

    def forward(self, h_V, h_E, E_idx, mask_V=None, mask_attend=None):
        """Parallel computation of full transformer layer"""

        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
        h_V_expand = h_V.unsqueeze(-2).expand(-1, -1, h_EV.size(-2), -1)
        h_EV = torch.cat([h_V_expand, h_EV], -1)
        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))

        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message
        dh = torch.sum(h_message, -2) / self.scale
        h_V = self.norm1(h_V + self.dropout1(dh))

        dh = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout2(dh))
        if mask_V is not None:
            mask_V = mask_V.unsqueeze(-1)
            h_V = mask_V * h_V

        if not self.is_last_layer:
            # Edge updates
            h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
            h_V_expand = h_V.unsqueeze(-2).expand(-1, -1, h_EV.size(-2), -1)
            h_EV = torch.cat([h_V_expand, h_EV], -1)
            h_message = self.W13(self.act(self.W12(self.act(self.W11(h_EV)))))
            h_E = self.norm3(h_E + self.dropout3(h_message))

        return h_V, h_E


def atom37_to_atom14(
    atom37_feats: TensorType["... n 37 ...", float],
    aatype: TensorType["... n", int],
) -> TensorType["... n 14 ...", float]:
    """Convert Atom37 positions to Atom14 positions."""
    device = atom37_feats.device
    num_batch_dims = len(aatype.shape)

    # Convert atom37 feature to atom14.
    restype_atom14_to_atom37 = torch.from_numpy(const.AF2_RESTYPE_ATOM14_TO_ATOM37).to(device=device, dtype=torch.long)
    atom14_feats = batched_gather(
        atom37_feats,
        restype_atom14_to_atom37[aatype],
        dim=num_batch_dims,
        no_batch_dims=num_batch_dims,
    )

    # Mask out positions that don't exist.
    standard_atom14_mask = torch.from_numpy(const.AF2_RESTYPE_ATOM14_MASK_WITH_X).to(device=device, dtype=torch.bool)
    atom14_mask = standard_atom14_mask[aatype]
    num_feat_dims = len(atom14_feats.shape) - len(atom14_mask.shape)
    for _ in range(num_feat_dims):
        atom14_mask = atom14_mask.unsqueeze(-1)
    atom14_feats = atom14_feats * atom14_mask
    return atom14_feats


def add_atom14_feats(batch: dict[str, TensorType["b ..."]]) -> dict[str, TensorType["b ..."]]:
    """
    Add atom14 feats to the batch.
    """
    B, N_atoms = batch["atom_cond_mask"].shape
    mask = batch["atom_pad_mask"].bool()
    batch_idx = torch.arange(B, device=mask.device).unsqueeze(1).expand(B, N_atoms)

    # Create coords and cond mask in AF2 atom37 format.
    atom37_coords = torch.zeros_like(batch["encoded_xyz"])
    atom37_coords[
        batch_idx[mask], batch["atom_to_token_map"][mask], batch["encoded_atom_to_within_token_map"][mask]
    ] = batch["coords"][mask]

    atom37_cond_mask = torch.zeros_like(batch["encoded_mask"])
    atom37_cond_mask[
        batch_idx[mask], batch["atom_to_token_map"][mask], batch["encoded_atom_to_within_token_map"][mask]
    ] = batch["atom_cond_mask"][mask].to(dtype=atom37_cond_mask.dtype)

    # Convert to AF2 atom14 format.
    aatype = batch["encoded_seq"]  # [B, N]
    batch["atom14_coords"] = atom37_to_atom14(atom37_coords, aatype)
    batch["atom14_cond_mask"] = atom37_to_atom14(atom37_cond_mask, aatype)
