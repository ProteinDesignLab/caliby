import importlib
import os
import shutil
from pathlib import Path

import hydra
import lightning as L
import torch
import yaml
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from caliby.eval.eval_utils.eval_setup_utils import get_pdb_files


def _import_protpardelle(cfg: DictConfig):
    """Import Protpardelle-1c for sampling.

    We set dummy paths since they're not required for partial diffusion.
    """
    os.environ["PROTPARDELLE_OUTPUT_DIR"] = f"{cfg.out_dir}/protpardelle_outputs_temp"
    os.environ["FOLDSEEK_BIN"] = "."
    os.environ["ESMFOLD_PATH"] = "."
    os.environ["PROTEINMPNN_WEIGHTS"] = f"{cfg.model_params_path}/proteinmpnn"
    os.environ["PROTPARDELLE_MODEL_PARAMS"] = f"{cfg.model_params_path}/protpardelle-1c"
    return importlib.import_module("protpardelle.sample")


@hydra.main(config_path="../../configs/eval/sampling", config_name="generate_ensembles", version_base="1.3.2")
def main(cfg: DictConfig):
    """
    Script for generating ensembles with Protpardelle-1c for multiple PDBs.
    """
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    protpardelle_sample = _import_protpardelle(cfg)
    GlobalHydra.instance().clear()

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

    # Load in PDB file to eval on
    pdb_files = get_pdb_files(**cfg.input_cfg)

    # Generate ensembles
    for pdb_file in tqdm(pdb_files, desc="Generating ensembles"):
        all_save_dirs = protpardelle_sample.sample(
            sampling_yaml_path=Path(cfg.sampling_yaml_path),
            motif_pdb=Path(pdb_file),
            batch_size=cfg.batch_size,
            num_samples=cfg.num_samples_per_pdb,
            num_mpnn_seqs=0,
        )

        # Move results from Protpardelle-1c output directories to our output directory
        for save_dir in all_save_dirs:
            out_dir_i = f"{out_dir}/{save_dir.parent.name}"
            Path(out_dir_i).mkdir(parents=True, exist_ok=True)
            shutil.move(save_dir, out_dir_i)


if __name__ == "__main__":
    main()
