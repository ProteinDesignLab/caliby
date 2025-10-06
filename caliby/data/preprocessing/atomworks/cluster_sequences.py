"""
Adapted from: https://github.com/jwohlwend/boltz/blob/main/scripts/process/cluster.py

Cluster the sequences in the metadata parquet file.
"""

import json
import os
import subprocess
from pathlib import Path

import atomworks.enums as aw_enums
import atomworks.ml.preprocessing.constants as aw_const
import hydra
import numpy as np
import pandas as pd
from atomworks.ml.utils.misc import hash_sequence
from omegaconf import DictConfig
from tqdm import tqdm


@hydra.main(
    config_path="../../../configs/data/preprocessing/atomworks",
    config_name="cluster_sequences",
    version_base="1.3.2",
)
def main(cfg: DictConfig) -> None:
    """
    Cluster the sequences in the metadata parquet file.
    """
    df = pd.read_parquet(f"{cfg.pdb_path}/metadata.parquet")
    clustering_dir = Path(cfg.pdb_path) / "clustering"
    clustering_dir.mkdir(parents=True, exist_ok=True)

    # Get all polymer sequences from metadata df
    proteins = set()
    shorts = set()
    nucleic_acids = set()
    nonpolymer_seqs = set()

    for _, row in tqdm(df.iterrows(), desc="Sorting sequences by type", total=len(df)):
        chain_type = row["q_pn_unit_type"]
        if chain_type in aw_enums.ChainTypeInfo.PROTEINS:
            if len(row["q_pn_unit_processed_entity_canonical_sequence"]) <= aw_const.PEPTIDE_MAX_RESIDUES:
                # short sequence
                shorts.add(row["q_pn_unit_processed_entity_canonical_sequence"])
            else:
                # protein
                proteins.add(row["q_pn_unit_processed_entity_canonical_sequence"])
        elif chain_type in aw_enums.ChainTypeInfo.NUCLEIC_ACIDS:
            # nucleic acid
            nucleic_acids.add(row["q_pn_unit_processed_entity_canonical_sequence"])
        else:
            # non-polymer
            nonpolymer_seqs.add(row["q_pn_unit_non_polymer_res_names"])

    # Run mmseqs on the protein data
    proteins = [f">{hash_sequence(seq)}\n{seq}" for seq in proteins]
    with (clustering_dir / "proteins.fasta").open("w") as f:
        f.write("\n".join(proteins))

    subprocess.run(
        f"{os.environ['SOFTWARE_PATH']}/mmseqs/bin/mmseqs easy-cluster {clustering_dir / 'proteins.fasta'} {clustering_dir / 'clust_prot'} {clustering_dir / 'tmp'} --min-seq-id 0.4",
        shell=True,
        check=True,
    )

    # Load protein clusters
    clustering_path = clustering_dir / "clust_prot_cluster.tsv"
    protein_data = pd.read_csv(clustering_path, sep="\t", header=None)
    clusters = protein_data[0]
    items = protein_data[1]
    clustering = dict(zip(list(items), list(clusters)))

    # Each short sequence is given an id
    for short in shorts:
        short_id = hash_sequence(short)
        clustering[short_id] = short_id

    # Each unique rna sequence is given an id
    for nucl in nucleic_acids:
        nucl_id = hash_sequence(nucl)
        clustering[nucl_id] = nucl_id

    # Each unique sequence of CCD codes is given an id
    for nonpolymer_seq in nonpolymer_seqs:
        nonpolymer_id = hash_sequence(nonpolymer_seq)
        clustering[nonpolymer_id] = nonpolymer_id

    # Assign each hash a cluster ID from 0 to num_clusters - 1
    cluster_hashes = list(set(clustering.values()))
    cluster_ids = {h: i for i, h in enumerate(cluster_hashes)}
    clustering = {k: cluster_ids[v] for k, v in clustering.items()}

    # Save clustering
    with (clustering_dir / "clustering.json").open("w") as handle:
        json.dump(clustering, handle)

    # Add cluster IDs to metadata df
    df["q_pn_unit_cluster_id"] = np.where(
        df["q_pn_unit_is_polymer"],
        df["q_pn_unit_processed_entity_canonical_sequence_hash"].map(clustering),
        df["q_pn_unit_non_polymer_res_names"].apply(hash_sequence).map(clustering),
    )

    # Sanity check that we have no missing values
    if df["q_pn_unit_cluster_id"].isna().any():
        print(f"WARNING: {df['q_pn_unit_cluster_id'].isna().sum()} missing values in q_pn_unit_cluster_id")

    df["q_pn_unit_cluster_id"] = (
        df["q_pn_unit_cluster_id"].fillna(-1).astype(np.int32)
    )  # fill missing cluster IDs with -1
    df.to_parquet(f"{cfg.pdb_path}/metadata_clustered.parquet")
    print(f"Saved clustered metadata to {cfg.pdb_path}/metadata_clustered.parquet")


if __name__ == "__main__":
    main()
