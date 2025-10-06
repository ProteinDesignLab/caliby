from pathlib import Path

import hydra
import lightning as L
import pandas as pd
import torch
import yaml
from omegaconf import DictConfig, OmegaConf

from caliby.eval.eval_utils.eval_setup_utils import get_pdb_files
from caliby.eval.eval_utils.seq_des_utils import get_seq_des_model, run_seq_des


@hydra.main(config_path="../../configs/eval/sampling", config_name="seq_des_multi", version_base="1.3.2")
def main(cfg: DictConfig):
    """
    Script for designing sequences for multiple PDBs.
    """
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    # Set seeds
    L.seed_everything(cfg.seed)
    torch.backends.cudnn.deterministic = True  # nonrandom CUDNN convolution algo, maybe slower
    torch.backends.cudnn.benchmark = False  # nonrandom selection of CUDNN convolution, maybe slower

    # Set up wandb logging / output directory
    out_dir = cfg.out_dir
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Preserve config
    with open(Path(out_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg_dict, f)

    # Load in PDB file to eval on
    pdb_files = get_pdb_files(**cfg.input_cfg)

    # Set up models (in eval mode)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load in sequence design model
    seq_des_model = get_seq_des_model(cfg.seq_des_cfg, device=device)

    # Read in fixed positions
    if cfg.pos_constraint_csv is not None:
        pos_constraint_df = pd.read_csv(cfg.pos_constraint_csv)
    else:
        pos_constraint_df = None

    # Run sequence design model
    outputs = run_seq_des(
        model=seq_des_model["model"],
        data_cfg=seq_des_model["data_cfg"],
        sampling_cfg=seq_des_model["sampling_cfg"],
        pdb_paths=pdb_files,
        device=device,
        pos_constraint_df=pos_constraint_df,
        out_dir=out_dir,
    )
    del seq_des_model  # delete model to free up memory

    # Save outputs to CSV
    output_df = pd.DataFrame(outputs)
    output_df.to_csv(f"{out_dir}/seq_des_outputs.csv", index=False)


if __name__ == "__main__":
    main()
