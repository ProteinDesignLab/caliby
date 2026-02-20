from pathlib import Path

import hydra
import lightning as L
import pandas as pd
import torch
import yaml
from omegaconf import DictConfig, OmegaConf

from caliby.eval.eval_utils import eval_metrics
from caliby.eval.eval_utils.eval_setup_utils import get_pdb_files
from caliby.eval.eval_utils.seq_des_utils import get_seq_des_model, run_sidechain_packing


@hydra.main(config_path="../../configs/eval/sampling", config_name="sidechain_pack", version_base="1.3.2")
def main(cfg: DictConfig):
    """
    Script for packing sidechains for multiple PDBs.
    """
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    # Set seeds
    L.seed_everything(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set up output directory
    out_dir = cfg.out_dir
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Preserve config
    with open(Path(out_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg_dict, f)

    # Load in PDB files
    pdb_files = get_pdb_files(**cfg.input_cfg)

    # Set up models (in eval mode)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load in sequence design model
    seq_des_model = get_seq_des_model(cfg.seq_des_cfg, device=device)

    # Run sidechain packing
    outputs = run_sidechain_packing(
        model=seq_des_model["model"],
        data_cfg=seq_des_model["data_cfg"],
        sampling_cfg=seq_des_model["sampling_cfg"],
        pdb_paths=pdb_files,
        device=device,
        out_dir=out_dir,
    )
    del seq_des_model  # delete model to free up memory

    # Compute RMSD between packed sidechains and original sidechains.
    id_to_metrics = eval_metrics.run_packing_metrics_eval(
        input_pdbs=pdb_files,
        pred_pdbs=outputs["out_pdb"],
        out_dir=out_dir,
        num_workers=cfg.num_workers,
    )

    metrics_df = pd.DataFrame([{"example_id": eid, **m} for eid, m in id_to_metrics.items()])

    # Save outputs to CSV.
    output_df = pd.DataFrame(outputs)
    output_df = pd.merge(
        output_df, metrics_df, left_on="example_id", right_on="example_id", how="inner", validate="1:1"
    )
    output_df.to_csv(f"{out_dir}/packing_metrics.csv", index=False)


if __name__ == "__main__":
    main()
