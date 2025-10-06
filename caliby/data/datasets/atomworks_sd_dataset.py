import json
import logging
import random
from pathlib import Path
from typing import Literal, override

import atomworks.enums as aw_enums
import atomworks.ml.preprocessing.constants as aw_const
import lightning as L
import numpy as np
import pandas as pd
import torch
from atomworks.ml.common import generate_example_id
from atomworks.ml.datasets.datasets import BaseDataset
from atomworks.ml.datasets.parsers import GenericDFParser
from atomworks.ml.utils.io import read_parquet_with_metadata
from omegaconf import DictConfig
from torch.utils import data
from torch.utils.data import DataLoader

from caliby.data.sampler import Sampler
from caliby.data.transform.pad import pad_to_max
from caliby.data.transform import sd_featurizer

logger = logging.getLogger(__name__)


class AtomworksSDDataModule(L.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.pdb_path = cfg.pdb_path
        self._train_set = SDDataset(cfg, phase="train")
        self._val_set = SDDataset(cfg, phase="val")

    def train_dataloader(self) -> DataLoader:
        train_loader = DataLoader(
            self._train_set,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=True,
            collate_fn=sd_collator,
            worker_init_fn=worker_init_fn,
        )

        return train_loader

    def val_dataloader(self) -> DataLoader:
        val_loader = DataLoader(
            self._val_set,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=True,
            collate_fn=sd_collator,
            worker_init_fn=worker_init_fn,
        )

        return val_loader


def worker_init_fn(_):
    """Initialize per-worker global random number generators."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class SDDataset(BaseDataset):
    def __init__(self, cfg: DictConfig, phase: Literal["train", "val"]):
        super().__init__()

        self.cfg = cfg
        self.phase = phase

        # Initialize featurizer
        # NOTE: We remove INFERENCE_ONLY_KEYS to avoid cuda initialization issues during training.
        self.featurizer = sd_featurizer.sd_featurizer(
            **cfg.featurizer_cfg, remove_keys=sd_featurizer.INFERENCE_ONLY_KEYS
        )

        # Read in chain metadata parquet
        self.chain_df = self._process_chain_df()

        # Build interface df from contacts in chain df
        self.interface_df = self._process_interface_df()

        # Parse dfs into a common format and concatenate
        self.parsed_df = self._parse_dfs()

        # Initialize per-worker random number generator
        if phase == "train":
            self._sampler = Sampler(self.get_sampling_weights())
            self._rng, self._samples = None, None

    @override
    def __getitem__(self, idx: int):
        if self.phase == "train":
            # For training, draw from infinite sampler.
            self._ensure_worker_rng()
            idx = next(self._samples)

        example_id = self.idx_to_id(idx)
        parsed_row = self.parsed_df.loc[example_id]

        # Load cached example.
        try:
            example = self._load_cached_example(parsed_row["extra_info"]["pdb_id"])
        except FileNotFoundError:
            logger.warning(
                f"Cached example for {parsed_row['extra_info']['pdb_id']} not found in {self.cfg.pdb_path}/cached_examples, skipping..."
            )
            return self.__getitem__(idx + 1)

        example.update(parsed_row)  # add in query_pn_unit_iids

        # Apply train-time transforms.
        try:
            feats = self.featurizer(example)
        except Exception as e:
            logger.error(f"Error applying train-time transforms to example {example_id}: {e}")
            return self.__getitem__(idx + 1)

        return feats

    def _ensure_worker_rng(self):
        """Ensure that each worker has a unique random number generator."""
        if self._rng is None:
            self._rng = np.random.default_rng(torch.initial_seed() % 2**32)
            self._samples = self._sampler.sample(self._rng)

    def _process_chain_df(self) -> pd.DataFrame:
        """
        Processes the chain dataframe. Adds chain counts info and sampling weights, and applies filters.
        """
        # Read in chain parquet
        chain_df = read_parquet_with_metadata(self.cfg.parquet_path)
        chain_df.set_index("example_id", inplace=True, drop=False, verify_integrity=True)
        chain_df["q_pn_unit_contacting_pn_unit_iids"] = chain_df["q_pn_unit_contacting_pn_unit_iids"].apply(json.loads)

        # Load in validation IDs and hold out based on phase. Case insensitive, no extension.
        with open(self.cfg.validation_ids_txt, "r") as f:
            logger.info(f"Loading in validation IDs from {self.cfg.validation_ids_txt}...")
            val_split = {x.lower() for x in f.read().splitlines()}
        chain_df.loc[~chain_df["pdb_id"].str.lower().isin(val_split), "phase"] = "train"
        chain_df.loc[chain_df["pdb_id"].str.lower().isin(val_split), "phase"] = "val"
        chain_df = chain_df[chain_df["phase"] == self.phase]

        # Exclude IDs
        exclude_ids = set()
        for exclude_ids_txt in self.cfg.exclude_ids_txts:
            with open(exclude_ids_txt, "r") as f:
                logger.info(f"Loading in exclude IDs from {exclude_ids_txt}...")
                exclude_ids.update({x.lower() for x in f.read().splitlines()})
        chain_df = chain_df[~chain_df["pdb_id"].str.lower().isin(exclude_ids)]

        # Add chain counts info and sampling weights
        chain_df = add_chain_counts_info(
            chain_df, chain_type_cols=["q_pn_unit_type"], seq_length_cols=["q_pn_unit_sequence_length"]
        )
        chain_df = add_sampling_weights_info(
            chain_df,
            alphas=self.cfg.sampling_weights["alphas"],
            beta=self.cfg.sampling_weights["betas"]["beta_chain"],
            cluster_cols=["q_pn_unit_cluster_id"],
        )

        # Apply chain filters
        filters = self.cfg.train_filters.chain if self.phase == "train" else self.cfg.val_filters.chain
        chain_df = self._apply_filters(filters, chain_df)

        return chain_df

    def _process_interface_df(self) -> pd.DataFrame:
        """
        Processes the interface dataframe based on the filtered chain dataframe. Adds chain counts info and sampling weights.
        """
        interface_df = build_interface_df(self.chain_df, dataset_name=Path(self.cfg.parquet_path).parent.name)
        interface_df = add_chain_counts_info(
            interface_df,
            chain_type_cols=["q_pn_unit_type_1", "q_pn_unit_type_2"],
            seq_length_cols=["q_pn_unit_sequence_length_1", "q_pn_unit_sequence_length_2"],
        )
        interface_df = add_sampling_weights_info(
            interface_df,
            alphas=self.cfg.sampling_weights["alphas"],
            beta=self.cfg.sampling_weights["betas"]["beta_interface"],
            cluster_cols=["q_pn_unit_cluster_id_1", "q_pn_unit_cluster_id_2"],
        )

        return interface_df

    def _parse_dfs(self) -> pd.DataFrame:
        """
        Parses the chain and interface dataframes into a common format and concatenates them.
        """
        chain_parser = GenericDFParser(pn_unit_iid_colnames=["q_pn_unit_iid"])
        interface_parser = GenericDFParser(pn_unit_iid_colnames=["q_pn_unit_iid_1", "q_pn_unit_iid_2"])

        logger.info(
            f"Final {self.phase} dataset contains {len(self.chain_df)} chains and {len(self.interface_df)} interfaces"
        )

        parsed_df = pd.concat(
            [self.chain_df.apply(chain_parser.parse, axis=1), self.interface_df.apply(interface_parser.parse, axis=1)],
            axis=0,
        )

        return parsed_df

    def get_sampling_weights(self) -> np.ndarray:
        return self.parsed_df.apply(lambda x: x["extra_info"]["sampling_weight"]).to_numpy()

    @override
    def __len__(self) -> int:
        if self.phase == "train":
            return self.cfg.samples_per_epoch
        return len(self.parsed_df)

    @override
    def __contains__(self, example_id: str) -> bool:
        return example_id in self.parsed_df.index

    @override
    def id_to_idx(self, example_id: str) -> int:
        return self.parsed_df.index.get_loc(example_id)

    @override
    def idx_to_id(self, idx: int) -> str:
        return self.parsed_df.index[idx]

    def _load_cached_example(self, pdb_id: str) -> dict[str, torch.Tensor]:
        cached_example_path = f"{self.cfg.pdb_path}/cached_examples/{pdb_id}.pt"
        return torch.load(cached_example_path, map_location="cpu", weights_only=False)

    def _apply_filters(self, filters: list[str] | None, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply filters to the data based on the provided list of query strings.
        For documentation on pandas query syntax, see: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.query.html

        Args:
            filters (list[str]): List of query strings to apply to the data.

        Raises:
            ValueError: If the data is not initialized or if a query removes all rows.
            Warning: If a query does not remove any rows.

        Exampleelse:
            logger.info(
                f"Query '{query}' filtered dataset from {original_num_rows:,} to {filtered_num_rows:,} rows (dropped {original_num_rows - filtered_num_rows:,} rows)"
            ):
            queries = [
                "deposition_date < '2020-01-01'",
                "resolution < 2.5 and ~method.str.contains('NMR')",
                "cluster.notnull()",
                "method in ['X-RAY_DIFFRACTION', 'ELECTRON_MICROSCOPY']"
            ]
        """
        if filters is None:
            return df

        # Apply queries one by one, confirming the impact of each
        for query in filters:
            df = self._apply_query(query, df)

        return df

    def _apply_query(self, query: str, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply a single query to the data.

        Args:
            query (str): A query string to apply to the data.
        """
        # Filter using query and validate impact
        original_num_rows = len(df)
        df = df.query(query)
        filtered_num_rows = len(df)
        self._validate_filter_impact(query, original_num_rows, filtered_num_rows)
        return df

    def _validate_filter_impact(self, query: str, original_num_rows: int, filtered_num_rows: int) -> None:
        """
        Validate the impact of the filter.

        Args:
            query (str): The query string that was applied.
            original_num_rows (int): The number of rows before applying the filter.
            filtered_num_rows (int): The number of rows after applying the filter.

        Raises:
            Warning: If the filter did not remove any rows.
            ValueError: If the filter removed all rows.
        """
        rows_removed = original_num_rows - filtered_num_rows
        percent_removed = (rows_removed / original_num_rows) * 100
        percent_remaining = (filtered_num_rows / original_num_rows) * 100

        if filtered_num_rows == original_num_rows:
            logger.warning(f"Query '{query}' on dataset did not remove any rows.")
        elif filtered_num_rows == 0:
            raise ValueError(f"Query '{query}' on dataset removed all rows.")
        else:
            logger.info(
                f"\n+-------------------------------------------+\n"
                f"Query '{query}' on dataset:\n"
                f"  - Started with: {original_num_rows:,} rows\n"
                f"  - Removed: {rows_removed:,} rows ({percent_removed:.2f}%)\n"
                f"  - Remaining: {filtered_num_rows:,} rows ({percent_remaining:.2f}%)\n"
                f"+-------------------------------------------+\n"
            )


def sd_collator(data: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Collate sequence denoiser features into a batch.

    Parameters
    ----------
    data : list[dict[str, torch.Tensor]]
        The data to collate.

    Returns
    -------
    dict[str, torch.Tensor]
        The collated data.

    """
    # Get the keys
    keys = data[0].keys()

    # Collate the data
    collated = {}
    for key in keys:
        values = [d[key] for d in data]

        if key not in ["example_id", *sd_featurizer.INFERENCE_ONLY_KEYS]:
            # Check if all have the same shape
            shape = values[0].shape
            if not all(v.shape == shape for v in values):
                values, _ = pad_to_max(values, 0)
            else:
                values = torch.stack(values, dim=0)

        # Stack the values
        collated[key] = values

    return collated


def build_interface_df(chain_df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    # Bring example_id into a column if it's the index
    chain_df = chain_df.reset_index(drop=True)

    # Get columns we'll need from the source df
    chain_specific_cols = [
        "q_pn_unit_iid",
        "q_pn_unit_type",
        "q_pn_unit_sequence_length",
        "q_pn_unit_cluster_id",
    ]  # columns we need for each chain
    base_cols = [
        "example_id",
        "pdb_id",
        "assembly_id",
        "path",
        "q_pn_unit_contacting_pn_unit_iids",
        *chain_specific_cols,
    ]
    interface_df = chain_df[base_cols].copy()

    # Explode interface contacts
    interface_df = interface_df.explode("q_pn_unit_contacting_pn_unit_iids", ignore_index=True)
    interface_df = interface_df.dropna(
        subset=["q_pn_unit_contacting_pn_unit_iids"]
    )  # drop pn_units without interface contacts

    # Extract the contacted iid
    interface_df["q_pn_unit_iid_2"] = interface_df["q_pn_unit_contacting_pn_unit_iids"].map(
        lambda d: d.get("pn_unit_iid") if isinstance(d, dict) else None
    )
    interface_df = interface_df.dropna(subset=["q_pn_unit_iid_2"])

    # Join back to get chain info for chain_2
    right = chain_df[["pdb_id", "assembly_id"] + chain_specific_cols].rename(
        columns={f"{c}": f"{c}_2" for c in chain_specific_cols}
    )
    interface_df = interface_df.merge(
        right,
        on=["pdb_id", "assembly_id", "q_pn_unit_iid_2"],
        how="inner",
        validate="many_to_one",  # inner join gets rid of interfaces where chain_2 was not in the input chain df
    )

    # Canonicalize pair ordering to dedupe (A_1, B_1) == (B_1, A_1)
    interface_df = _canonicalize_pair_columns(interface_df, order_by="q_pn_unit_iid", paired_cols=chain_specific_cols)

    # Drop exact duplicate interfaces within (pdb_id, assembly_id)
    interface_df = interface_df.drop_duplicates(
        subset=["pdb_id", "assembly_id", "q_pn_unit_iid_1", "q_pn_unit_iid_2"], keep="first"
    )

    # Build example_id for interfaces by appending 'interfaces' to the source dataset_names
    def _get_interface_example_id(row):
        dataset_names = [dataset_name, "interfaces"]
        query_pn_unit_iids = [row["q_pn_unit_iid_1"], row["q_pn_unit_iid_2"]]
        return generate_example_id(dataset_names, row["pdb_id"], row["assembly_id"], query_pn_unit_iids)

    interface_df["example_id"] = interface_df.apply(_get_interface_example_id, axis=1)

    # Final selection / order of columns
    interface_df = interface_df[
        ["example_id", "pdb_id", "assembly_id", "path"]
        + [f"{c}_1" for c in chain_specific_cols]
        + [f"{c}_2" for c in chain_specific_cols]
    ].reset_index(drop=True)
    interface_df.set_index("example_id", inplace=True, drop=False, verify_integrity=True)

    return interface_df


def _canonicalize_pair_columns(
    df: pd.DataFrame,
    order_by: str = "q_pn_unit_iid",
    paired_cols: list[str] = (
        "q_pn_unit_iid",
        "q_pn_unit_type",
        "q_pn_unit_sequence_length",
        "q_pn_unit_cluster_id",
    ),
    right_suffix: str = "_2",
    out_suffixes: tuple[str, str] = ("_1", "_2"),
) -> pd.DataFrame:
    """
    Create canonical *_1/*_2 columns for each entry in `paired_cols` by
    ordering rows so that order_by_1 <= order_by_2 (lexicographic).
    Does not mutate inputs; returns a new DataFrame with added *_1/*_2 columns.
    """
    out = df.copy()

    # mask: True -> swap left/right
    swap = out[order_by].to_numpy() > out[f"{order_by}{right_suffix}"].to_numpy()

    for base in paired_cols:
        a = out[f"{base}"].to_numpy()
        b = out[f"{base}{right_suffix}"].to_numpy()

        out[f"{base}{out_suffixes[0]}"] = np.where(swap, b, a)
        out[f"{base}{out_suffixes[1]}"] = np.where(swap, a, b)

    return out


def add_chain_counts_info(df: pd.DataFrame, chain_type_cols: list[str], seq_length_cols: list[str]) -> pd.DataFrame:
    """
    Add chain type and sequence length columns to the dataframe.
    Modifies the dataframe in place and returns it.
    """
    # Compute chain type counts
    chain_count_cols = ["n_prot", "n_nuc", "n_ligand", "n_peptide", "is_loi"]
    df["chain_types"] = df[chain_type_cols].apply(lambda x: tuple(x), axis=1)
    df["seq_lengths"] = df[seq_length_cols].apply(lambda x: tuple(x), axis=1)

    def _get_chain_type_counts(row) -> dict[str, int]:
        chain_types: tuple[str] = row["chain_types"]
        seq_lengths: tuple[int] = row["seq_lengths"]
        chain_type_counts = {c: 0 for c in chain_count_cols}

        for t, l in zip(chain_types, seq_lengths):
            if t in aw_enums.ChainTypeInfo.PROTEINS:
                if l < aw_const.PEPTIDE_MAX_RESIDUES:
                    chain_type_counts["n_peptide"] += 1
                else:
                    chain_type_counts["n_prot"] += 1
            elif t in aw_enums.ChainTypeInfo.NUCLEIC_ACIDS:
                chain_type_counts["n_nuc"] += 1
            else:
                chain_type_counts["n_ligand"] += 1

        return pd.Series(chain_type_counts)

    df[chain_count_cols] = df.apply(_get_chain_type_counts, axis=1)

    # Delete intermediate columns
    del df["chain_types"]
    del df["seq_lengths"]
    return df


def add_sampling_weights_info(
    df: pd.DataFrame, alphas: dict[str, float], beta: float, cluster_cols: list[str]
) -> pd.DataFrame:
    """
    Based on the cluster ID in cluster_col and chain counts info, add a sampling weights column to the dataframe.
    Modifies the dataframe in place and returns it.
    """
    assert all(
        col in df.columns for col in ["n_prot", "n_peptide", "n_nuc", "n_ligand", "is_loi"]
    ), "Need to add chain counts info before computing sampling weights"

    # Get cluster size
    df["clusters"] = df[cluster_cols].apply(lambda x: tuple(sorted(tuple(x))), axis=1)  # sort cluster ids to dedupe
    cluster_id_to_size = df["clusters"].value_counts()
    df["cluster_size"] = df["clusters"].map(cluster_id_to_size)

    # Compute weights
    missing_alphas = set(alphas.keys()) - {"a_prot", "a_peptide", "a_nuc", "a_ligand", "a_loi"}
    missing_counts = {"n_prot", "n_peptide", "n_nuc", "n_ligand"} - set(df.columns)

    if missing_alphas:
        logger.warning(f"Missing alphas from configuration file: {missing_alphas}; defaulting to 0")
    if missing_counts:
        logger.warning(f"Missing chain within dataframe counts: {missing_counts}; defaulting to 0")
        logger.warning(f"Columns in dataframe: {df.columns}")

    logger.info(f"Calculating weights for AF-3 examples using alphas={alphas}, beta={beta}")

    weights = (beta / df["cluster_size"]) * (
        alphas.get("a_prot", 0) * df["n_prot"]
        + alphas.get("a_peptide", 0) * df["n_peptide"]
        + alphas.get("a_nuc", 0) * df["n_nuc"]
        + alphas.get("a_ligand", 0) * df["n_ligand"]
        + alphas.get("a_loi", 0) * df["is_loi"]  # always 0 for now
    )

    # Set the weight of missing cluster IDs to 0.
    cluster_id_is_missing = df["clusters"].apply(lambda x: -1 in x)
    if cluster_id_is_missing.any():
        logger.warning(
            f"Setting sampling weights of {cluster_id_is_missing.sum()} examples with missing cluster IDs to 0"
        )
    weights[cluster_id_is_missing] = 0

    df["sampling_weight"] = weights
    return df
