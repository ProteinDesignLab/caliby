from pathlib import Path

import hydra
import lightning as L
import pandas as pd
import torch
import yaml
from omegaconf import DictConfig, OmegaConf

from caliby.eval.eval_utils.eval_setup_utils import get_conformer_dirs, process_conformer_dirs, wandb_setup
from caliby.eval.eval_utils.seq_des_utils import get_seq_des_model, score_samples_ensemble


@hydra.main(config_path="../../configs/eval/sampling", config_name="score_ensemble", version_base="1.3.2")
def main(cfg: DictConfig):
    """
    Script for scoring sequences from input PDBs against input backbones.
    """
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    # Set seeds
    L.seed_everything(cfg.seed)
    torch.backends.cudnn.deterministic = True  # nonrandom CUDNN convolution algo, maybe slower
    torch.backends.cudnn.benchmark = False  # nonrandom selection of CUDNN convolution, maybe slower

    # Make output directory
    out_dir = cfg.out_dir  # base output directory
    Path(out_dir).mkdir(parents=True, exist_ok=True)  # create output directory

    # Preserve config
    with open(Path(out_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg_dict, f)

    # Load in conformer directories to eval on
    conformer_dirs = get_conformer_dirs(**cfg.input_cfg)

    # Process conformer directories
    pdb_to_conformers = process_conformer_dirs(
        conformer_dirs,
        max_num_conformers=cfg.max_num_conformers,
        include_primary_conformer=cfg.include_primary_conformer,
    )

    # Set up models (in eval mode)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load in sequence design model
    seq_des_model = get_seq_des_model(cfg.seq_des_cfg, device=device)

    # Score sequence of primary conformer against ensemble of backbones.
    outputs = score_samples_ensemble(
        model=seq_des_model["model"],
        data_cfg=seq_des_model["data_cfg"],
        sampling_cfg=seq_des_model["sampling_cfg"],
        pdb_to_conformers=pdb_to_conformers,
        device=device,
    )

    # Parse score outputs into a flattened dataframe
    output_df = pd.DataFrame(
        {
            "example_id": outputs["example_id"],
            "seq": outputs["seq"],
            "U": outputs["U"],
        }
    )

    # Save to csv
    output_df.to_csv(f"{out_dir}/score_outputs.csv", index=False)


if __name__ == "__main__":
    main()
