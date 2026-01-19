from pathlib import Path

import hydra
import lightning as L
import numpy as np
import pandas as pd
import torch
import yaml
from omegaconf import DictConfig, OmegaConf

from caliby.eval.eval_utils.eval_setup_utils import get_pdb_files
from caliby.eval.eval_utils.seq_des_utils import get_seq_des_model, score_samples


@hydra.main(config_path="../../configs/eval/sampling", config_name="score", version_base="1.3.2")
def main(cfg: DictConfig):
    """
    Script for scoring sequences from input PDBs against input backbones.
    """
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    # Set seeds.
    L.seed_everything(cfg.seed)
    torch.backends.cudnn.deterministic = True  # nonrandom CUDNN convolution algo, maybe slower
    torch.backends.cudnn.benchmark = False  # nonrandom selection of CUDNN convolution, maybe slower

    # Make output directory.
    out_dir = cfg.out_dir  # base output directory
    Path(out_dir).mkdir(parents=True, exist_ok=True)  # create output directory

    # Preserve config.
    with open(Path(out_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg_dict, f)

    # Load in PDB file to eval on.
    pdb_files = get_pdb_files(**cfg.input_cfg)

    # Set up models (in eval mode).
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load in sequence design model.
    seq_des_model = get_seq_des_model(cfg.seq_des_cfg, device=device)

    # Score each input backbone.
    outputs = score_samples(
        model=seq_des_model["model"],
        data_cfg=seq_des_model["data_cfg"],
        sampling_cfg=seq_des_model["sampling_cfg"],
        pdb_paths=pdb_files,
        device=device,
    )

    if cfg.save_local_conditionals:
        Path(f"{out_dir}/local_conditionals").mkdir(parents=True, exist_ok=True)
        for example_id, U_i in zip(outputs["example_id"], outputs["U_i"]):
            np.save(f"{out_dir}/local_conditionals/{example_id}.npy", U_i.cpu().numpy())

    # Parse score outputs into a flattened dataframe.
    output_df = pd.DataFrame({"example_id": outputs["example_id"], "seq": outputs["seq"], "U": outputs["U"]})

    # Save to csv.
    output_df.to_csv(f"{out_dir}/score_outputs.csv", index=False)


if __name__ == "__main__":
    main()
