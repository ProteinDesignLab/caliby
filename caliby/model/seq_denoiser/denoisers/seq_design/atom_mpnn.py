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


class AtomMPNN(nn.Module):
    """Modified ProteinMPNN network to predict sequence from full atom structure."""

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.node_features = cfg.n_channel
        self.edge_features = cfg.n_channel
        self.hidden_dim = cfg.n_channel
        self.num_encoder_layers = cfg.n_layers
        self.num_decoder_layers = cfg.n_layers
        self.k_neighbors = cfg.k_neighbors
        self.n_tokens = const.AF3_ENCODING.n_tokens

        self.token_features = TokenFeatures(cfg.token_features)
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
        self.decoder_layers = nn.ModuleList(
            [DecLayer(self.hidden_dim, self.decoder_in, dropout=cfg.dropout_p) for _ in range(self.num_decoder_layers)]
        )

        # Potts decoder
        self.use_potts = cfg.potts.use_potts
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

    def forward(self, batch: dict[str, TensorType["b ..."]], is_sampling: bool):
        # If provided, add noise to input coordinates.
        batch["coords"] = self._add_noise(batch)

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

    def _add_noise(self, batch: dict[str, TensorType["b ..."]]) -> TensorType["b n_atoms 3", float]:
        """
        If provided, add noise to input coordinates.
        """
        if batch["noise"] is None:
            return batch["coords"]

        # Add noise to input coordinates.
        noised_coords = batch["coords"] + batch["noise"]
        return noised_coords


class TokenFeatures(nn.Module):
    def __init__(self, cfg: DictConfig):
        """
        Extract token-level edge features and build KNN graph.
        """
        super().__init__()
        self.cfg = cfg

        # Parameters
        self.ca_only = cfg.get("ca_only", True)  # backwards compatibility
        self.k_neighbors = cfg.k_neighbors
        self.num_rbf = cfg.num_rbf
        self.num_positional_embeddings = cfg.num_positional_embeddings
        self.edge_n_channel = cfg.edge_n_channel
        self.use_multichain_encoding = cfg.get("use_multichain_encoding", False)

        # Layers
        self.embeddings = PositionalEncodings(self.num_positional_embeddings)
        num_pairwise_dists = 1 if self.ca_only else const.MAX_NUM_ATOMS**2
        edge_in = self.num_positional_embeddings + self.num_rbf * num_pairwise_dists + 1
        self.edge_embedding = nn.Linear(edge_in, self.edge_n_channel, bias=False)
        self.norm_edges = nn.LayerNorm(self.edge_n_channel)

    def forward(self, batch: dict[str, TensorType["b ..."]]):
        """
        Extract token-level edge features and build KNN graph.
        """
        X = self._get_token_coords(batch)
        D_neighbors, E_idx = self._dist(X, batch["token_exists_mask"].float())

        # Get RBF features
        if self.ca_only:
            RBF_all = self._rbf(D_neighbors)
        else:
            X_all, tokenwise_atom_cond_mask = get_tokenwise_coords(batch)
            X_all = torch.where(
                tokenwise_atom_cond_mask.unsqueeze(-1).bool(), X_all, X[..., None, :]
            )  # replace all masked atoms with center atom for the residue

            RBF_all = []
            for i in range(X_all.shape[-2]):
                for j in range(X_all.shape[-2]):
                    RBF_all.append(self._get_rbf(X_all[..., i, :], X_all[..., j, :], E_idx))
            RBF_all = torch.cat(RBF_all, dim=-1)

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

    def _get_token_coords(self, batch: dict[str, TensorType["b ..."]]) -> TensorType["b n 3", float]:
        """
        Get token-level coordinates as an average over all known, resolved atoms in the token.
        """
        B, N, _ = batch["coords"].shape
        X = batch["coords"][
            torch.arange(B).unsqueeze(-1), batch["token_to_center_atom"]
        ]  # get center atom for each token
        X = X * batch["token_exists_mask"].unsqueeze(-1)  # mask out padding and unresolved atoms
        return X

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
    def __init__(self, num_hidden, num_in, dropout=0.1, num_heads=None, scale=30):
        super(DecLayer, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(self.num_hidden)
        self.norm2 = nn.LayerNorm(self.num_hidden)

        self.W1 = nn.Linear(self.num_hidden + num_in, self.num_hidden, bias=True)
        self.W2 = nn.Linear(self.num_hidden, self.num_hidden, bias=True)
        self.W3 = nn.Linear(self.num_hidden, self.num_hidden, bias=True)
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

        # edge updates
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


def get_tokenwise_coords(
    batch: dict[str, TensorType["b ..."]],
) -> tuple[TensorType["b n_tokens 23 3", float], TensorType["b n_tokens 23"]]:
    """
    Get token-level coordinates (padded to max_num_atoms per token). Batched version of pad_atom_feats_to_tokenwise
    for just coords.
    """
    B = batch["coords"].shape[0]
    device = batch["coords"].device

    # Build padded atom idxs.
    N_tokens = batch["token_pad_mask"].shape[1]
    n_atoms_per_token = (
        F.one_hot(batch["atom_to_token_map"], num_classes=N_tokens) * batch["atom_pad_mask"][..., None]
    ).sum(dim=-2)
    atom_idxs = torch.cat([torch.zeros((B, 1), device=device), n_atoms_per_token.cumsum(dim=-1)[:, :-1]], dim=-1).long()
    padded_atom_idxs = atom_idxs[..., None] + torch.arange(const.MAX_NUM_ATOMS, device=device)[None, None]
    pad_mask = torch.arange(const.MAX_NUM_ATOMS, device=device)[None, None, :] < n_atoms_per_token[..., None]
    padded_atom_idxs = padded_atom_idxs * pad_mask  # mask out ghost atoms

    # Gather coords.
    B, N, _ = padded_atom_idxs.shape
    X_all = batched_gather(batch["coords"], padded_atom_idxs, dim=1, no_batch_dims=1) * pad_mask.view(
        B, N, const.MAX_NUM_ATOMS, 1
    )
    tokenwise_atom_cond_mask = batched_gather(
        batch["atom_cond_mask"], padded_atom_idxs, dim=1, no_batch_dims=1
    ) * pad_mask.view(B, N, const.MAX_NUM_ATOMS)

    X_all = X_all * tokenwise_atom_cond_mask.unsqueeze(-1)  # zero out masked atoms
    return X_all, tokenwise_atom_cond_mask
