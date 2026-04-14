# Python API

Caliby provides a Python API for programmatic use in scripts and notebooks, without needing Hydra or the CLI. Install Caliby as usual (`uv pip install -e .` or `uv pip install "git+https://github.com/ProteinDesignLab/caliby.git"`) and import directly:

```python
from caliby import load_model
```

- [Quick start](#quick-start)
- [Loading a model](#loading-a-model)
- [Sequence design](#sequence-design)
- [Ensemble-conditioned sequence design](#ensemble-conditioned-sequence-design)
  - [Generating ensembles with Protpardelle-1c](#generating-ensembles-with-protpardelle-1c)
  - [Providing your own ensembles](#providing-your-own-ensembles)
- [Scoring](#scoring)
  - [Single-structure scoring](#single-structure-scoring)
  - [Ensemble scoring](#ensemble-scoring)
- [Positional constraints](#positional-constraints)
  - [Fixed positions](#fixed-positions)
  - [Amino acid restrictions](#amino-acid-restrictions)
  - [Sequence overrides](#sequence-overrides)
  - [Symmetry positions](#symmetry-positions)
  - [Constraints for ensemble-conditioned design](#constraints-for-ensemble-conditioned-design)
- [Additional sampling options](#additional-sampling-options)
  - [Omitting amino acids](#omitting-amino-acids)
  - [Sampling temperature](#sampling-temperature)
  - [Advanced overrides](#advanced-overrides)
- [Sidechain packing](#sidechain-packing)
- [Self-consistency evaluation with AlphaFold2](#self-consistency-evaluation-with-alphafold2)
- [One-shot convenience functions](#one-shot-convenience-functions)
- [API reference](#api-reference)

## Quick start

```python
from caliby import load_model

# Load the model once and reuse it.
model = load_model("caliby")

# Design 4 sequences for a PDB file.
results = model.sample(
    ["examples/example_data/native_pdbs/7xhz.cif"],
    num_seqs_per_pdb=4,
    out_dir="outputs/seq_des",
)

# results is a dict with keys: "example_id", "out_pdb", "seq", "U", "input_seq"
for name, seq, energy in zip(results["example_id"], results["seq"], results["U"]):
    print(f"{name}: {seq} (U={energy:.2f})")
```

## Loading a model

Use `load_model()` to load a model checkpoint. The model is returned as a `CalibyModel` object that you can reuse across multiple calls.

```python
from caliby import load_model

# Load by model name (weights are auto-downloaded from HuggingFace on first use).
model = load_model("caliby")              # Default model
model = load_model("soluble_caliby")      # Excludes transmembrane proteins
model = load_model("soluble_caliby_v1")   # Trained on monomers and interfaces

# Load from a custom checkpoint path.
model = load_model("/path/to/custom_model.ckpt")

# Specify device (defaults to "cuda" if available, else "cpu").
model = load_model("caliby", device="cuda:1")
```

Available model names are listed in the [Download model weights](README.md#download-model-weights) section of the README.

## Sequence design

Design sequences for one or more PDB/CIF files:

```python
model = load_model("caliby")

results = model.sample(
    ["protein1.pdb", "protein2.cif"],
    num_seqs_per_pdb=4,       # 4 sequences per structure
    batch_size=4,             # batch size for processing
    out_dir="outputs/",       # directory for output CIF files (temp dir if None)
)

# Access results.
results["example_id"]  # e.g. ["protein1_0", "protein1_1", ...]
results["seq"]         # designed sequences
results["U"]           # Potts energy of each designed sequence
results["out_pdb"]     # paths to output CIF files
results["input_seq"]   # native sequence of the input structure
```

To design sequences for all PDBs in a directory:

```python
from pathlib import Path

pdb_dir = Path("examples/example_data/native_pdbs")
pdb_paths = [str(p) for p in sorted(pdb_dir.glob("*.cif"))]

results = model.sample(pdb_paths, num_seqs_per_pdb=4, out_dir="outputs/seq_des")
```

## Ensemble-conditioned sequence design

Designing on synthetic structural ensembles (rather than a single static structure) produces sequences that are both more diverse and more likely to fold into the target structure.

### Generating ensembles with Protpardelle-1c

Use `generate_ensembles()` to generate conformer ensembles via Protpardelle-1c partial diffusion:

```python
from caliby import generate_ensembles, load_model

# Generate 32 conformers per PDB.
pdb_to_conformers = generate_ensembles(
    ["examples/example_data/native_pdbs/7xhz.cif"],
    out_dir="outputs/ensembles",
    num_samples_per_pdb=32,
    batch_size=8,
)
# pdb_to_conformers: {"7xhz": ["outputs/ensembles/.../sample_0.pdb", ...]}

# Run ensemble-conditioned sequence design.
model = load_model("caliby")
results = model.ensemble_sample(
    pdb_to_conformers,
    num_seqs_per_pdb=4,
    out_dir="outputs/seq_des_ensemble",
)
```

We recommend generating at least 32 conformers per PDB, but 16 or 8 can also give good results.

### Providing your own ensembles

If you have your own ensemble of conformers, pass them as a dict mapping PDB name to a list of file paths. The **first** path in each list is treated as the primary conformer:

```python
pdb_to_conformers = {
    "my_protein": [
        "ensembles/my_protein/my_protein.pdb",    # primary conformer (first)
        "ensembles/my_protein/conformer_1.pdb",
        "ensembles/my_protein/conformer_2.pdb",
        # ... up to max_num_conformers (default 32)
    ],
}

results = model.ensemble_sample(
    pdb_to_conformers,
    num_seqs_per_pdb=4,
    out_dir="outputs/seq_des_ensemble",
)
```

All conformers must have matching residue indices and chain IDs. If you encounter a "Residue index / chain ID mismatch" error, see the [FAQ](README.md#faqs) in the README.

## Scoring

### Single-structure scoring

Score the native sequences of PDB/CIF files:

```python
model = load_model("caliby")

results = model.score(["protein.pdb"])

results["example_id"]  # PDB identifiers
results["seq"]         # native sequences
results["U"]           # global Potts energy
results["U_i"]         # per-residue energy contributions
```

### Ensemble scoring

Score a sequence against an ensemble of conformer backbones:

```python
pdb_to_conformers = {
    "my_protein": [
        "ensembles/my_protein/my_protein.pdb",   # primary conformer
        "ensembles/my_protein/conformer_1.pdb",
        "ensembles/my_protein/conformer_2.pdb",
    ],
}

results = model.score_ensemble(pdb_to_conformers)
```

The sequence from the primary conformer is scored; sequences of additional conformers are ignored.

## Positional constraints

Use `make_constraints()` to build a constraint DataFrame, then pass it to `sample()` or `ensemble_sample()`:

```python
from caliby import make_constraints

constraints = make_constraints({
    "7xhz": {
        "fixed_pos_seq": "A6-15,A20-50",
        "fixed_pos_scn": "A6-15",
    },
})

results = model.sample(
    ["examples/example_data/native_pdbs/7xhz.cif"],
    num_seqs_per_pdb=4,
    pos_constraint_df=constraints,
)
```

You can also pass a `pd.DataFrame` directly with a `pdb_key` column and any subset of the constraint columns described below.

### Fixed positions

Fix certain residue positions so they retain their native sequence during design. Residue positions should use `label_seq_id` (not `auth_seq_id`). In PyMOL, run `set cif_use_auth, off` before loading a PDB to view positions in this numbering.

```python
constraints = make_constraints({
    "7xhz": {
        "fixed_pos_seq": "A6-15,A20-50",   # fix sequence at these positions
        "fixed_pos_scn": "A6-15",           # also fix sidechains (must be subset of fixed_pos_seq)
    },
})
```

### Amino acid restrictions

Restrict which amino acids are allowed at specific positions:

```python
constraints = make_constraints({
    "8huz": {
        "pos_restrict_aatype": "A6:QR,A7:QR,A8:QR,A9:QR,A10:QR,A11:QR",
    },
})
```

### Sequence overrides

Override the sequence at specific positions before conditioning:

```python
constraints = make_constraints({
    "7xz3": {
        "fixed_pos_override_seq": "A36:C,A37:C,A38:C,A39:C,A40:C",
    },
})
```

### Symmetry positions

Tie sampling across residue positions (e.g., for homooligomers):

```python
constraints = make_constraints({
    "my_trimer": {
        "symmetry_pos": "A10,B10,C10|A11,B11,C11|A12,B12,C12",
    },
})
```

Positions separated by commas within a group are tied together. Groups are separated by `|`.

### Combining constraints

Multiple constraint types can be combined for the same PDB, and constraints for multiple PDBs can be specified in a single call:

```python
constraints = make_constraints({
    "7xhz": {
        "fixed_pos_seq": "A6-15",
        "pos_restrict_aatype": "A20:AVG,A21:AVG",
    },
    "8huz": {
        "pos_restrict_aatype": "A6:QR,A7:QR",
    },
})

results = model.sample(pdb_paths, num_seqs_per_pdb=4, pos_constraint_df=constraints)
```

### Constraints for ensemble-conditioned design

When using `ensemble_sample()`, constraints must be expanded so that every conformer in the ensemble gets a matching row. Use `make_ensemble_constraints()` to handle this automatically:

```python
from caliby import make_ensemble_constraints

# pdb_to_conformers from generate_ensembles() or your own ensemble
pdb_to_conformers = {
    "7xhz": ["7xhz.cif", "conformer_0.pdb", "conformer_1.pdb"],
}

constraints = make_ensemble_constraints(
    {"7xhz": {"fixed_pos_seq": "A6-15", "pos_restrict_aatype": "A20:AVG"}},
    pdb_to_conformers,
)

results = model.ensemble_sample(
    pdb_to_conformers,
    num_seqs_per_pdb=4,
    pos_constraint_df=constraints,
)
```

This is equivalent to calling `make_constraints()` and then manually replicating each row for every conformer path, but is less error-prone.

## Additional sampling options

### Omitting amino acids

Globally exclude certain amino acids from all designed positions:

```python
results = model.sample(pdb_paths, omit_aas=["C", "M"])
```

### Sampling temperature

By default, Caliby anneals the sampling temperature from 1.0 to 0.01. Raise the final temperature for more diverse sequences:

```python
results = model.sample(pdb_paths, temperature=0.1)
```

Temperatures from 0.1 to 0.2 also perform reasonably, but this is case-dependent.

### Advanced overrides

Any parameter from the [sampling config](caliby/configs/seq_des/inference.yaml) can be overridden via `sampling_overrides`:

```python
results = model.sample(
    pdb_paths,
    sampling_overrides={
        "potts_sampling_cfg": {
            "potts_sweeps": 1000,
            "regularization": "LCP",
        },
    },
)
```

## Sidechain packing

Pack sidechains onto backbone structures using a diffusion-based packer. Note that sidechain packing uses a different model checkpoint from sequence design:

```python
packer = load_model("caliby_packer_010")  # 0.1A noise (recommended)

results = packer.sidechain_pack(
    ["protein.pdb"],
    out_dir="outputs/packed",
)

results["example_id"]  # PDB identifiers
results["out_pdb"]     # paths to packed output CIF files
```

Available packer models: `caliby_packer_000` (0.0A), `caliby_packer_010` (0.1A, recommended), `caliby_packer_030` (0.3A).

## Self-consistency evaluation with AlphaFold2

Evaluate designed sequences by folding them with single-sequence AlphaFold2 and comparing against the design input. Requires the `af2` extra:

```bash
uv pip install -e ".[af2]"
```

```python
model = load_model("caliby")

# Design sequences.
results = model.sample(pdb_paths, num_seqs_per_pdb=2, out_dir="outputs/designed")

# Fold and evaluate.
sc_results = model.self_consistency_eval(
    results["out_pdb"],
    out_dir="outputs/af2_eval",
    num_models=5,
    num_recycles=3,
)

# sc_results: {"protein_0": {"sc_ca_rmsd": 1.2, "avg_ca_plddt": 85.3, "tmalign_score": 0.95}, ...}
for example_id, metrics in sc_results.items():
    print(f"{example_id}: scRMSD={metrics['sc_ca_rmsd']:.2f}, "
          f"pLDDT={metrics['avg_ca_plddt']:.1f}, TM={metrics['tmalign_score']:.3f}")
```

## One-shot convenience functions

For quick, one-off runs where you don't need to reuse the model, Caliby provides module-level convenience functions that load the model, run the task, and return results:

```python
from caliby import (
    caliby_sample,
    caliby_ensemble_sample,
    caliby_score,
    caliby_score_ensemble,
    caliby_sidechain_pack,
)

# These load the model each time — prefer load_model() for repeated calls.
results = caliby_sample(["protein.pdb"], num_seqs_per_pdb=4)
results = caliby_score(["protein.pdb"])
results = caliby_sidechain_pack(["protein.pdb"], model_name="caliby_packer_010")
```

## API reference

### `load_model(model_name, device, sampling_cfg_path) -> CalibyModel`

Load a Caliby model for reuse across multiple calls.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name` | `str` | `"caliby"` | Model name or path to `.ckpt` file |
| `device` | `str \| None` | `None` | Torch device (defaults to `"cuda"` if available) |
| `sampling_cfg_path` | `str \| None` | `None` | Custom sampling YAML config path |

### `CalibyModel.sample()`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pdb_paths` | `list[str]` | required | Paths to PDB/CIF files |
| `out_dir` | `str \| None` | `None` | Output directory (temp dir if None) |
| `num_seqs_per_pdb` | `int \| None` | `1` | Sequences per structure |
| `batch_size` | `int \| None` | `4` | Batch size |
| `omit_aas` | `list[str] \| None` | `None` | Amino acids to exclude |
| `num_workers` | `int \| None` | `8` | Data loading workers |
| `temperature` | `float \| None` | `0.01` | Final Potts sampling temperature |
| `verbose` | `bool \| None` | `True` | Print constraint info |
| `pos_constraint_df` | `DataFrame \| None` | `None` | Positional constraints |
| `sampling_overrides` | `dict \| None` | `None` | Advanced config overrides |

**Returns:** `dict` with keys `"example_id"`, `"out_pdb"`, `"seq"`, `"U"`, `"input_seq"`.

### `CalibyModel.ensemble_sample()`

Same parameters as `sample()`, plus:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pdb_to_conformers` | `dict[str, list[str]]` | required | Maps PDB name to conformer paths (first is primary) |
| `use_primary_res_type` | `bool` | `True` | Use residue types from the primary conformer |

### `CalibyModel.score()`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pdb_paths` | `list[str]` | required | Paths to PDB/CIF files |
| `batch_size` | `int \| None` | `4` | Batch size |
| `num_workers` | `int \| None` | `8` | Data loading workers |
| `sampling_overrides` | `dict \| None` | `None` | Advanced config overrides |

**Returns:** `dict` with keys `"example_id"`, `"seq"`, `"U"`, `"U_i"`.

### `CalibyModel.score_ensemble()`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pdb_to_conformers` | `dict[str, list[str]]` | required | Maps PDB name to conformer paths |
| `num_workers` | `int \| None` | `8` | Data loading workers |
| `sampling_overrides` | `dict \| None` | `None` | Advanced config overrides |

**Returns:** `dict` with keys `"example_id"`, `"seq"`, `"U"`, `"U_i"`.

### `CalibyModel.sidechain_pack()`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pdb_paths` | `list[str]` | required | Paths to PDB/CIF files |
| `out_dir` | `str \| None` | `None` | Output directory (temp dir if None) |
| `batch_size` | `int \| None` | `4` | Batch size |
| `num_workers` | `int \| None` | `8` | Data loading workers |
| `sampling_overrides` | `dict \| None` | `None` | Advanced config overrides |

**Returns:** `dict` with keys `"example_id"`, `"out_pdb"`.

### `CalibyModel.self_consistency_eval()`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `designed_pdbs` | `list[str]` | required | Paths to designed PDB/CIF files |
| `out_dir` | `str \| None` | `None` | Output directory (temp dir if None) |
| `num_models` | `int` | `5` | Number of AF2 models to sample |
| `sample_models` | `bool` | `True` | Randomly sample from the 5 AF2 models |
| `num_recycles` | `int` | `3` | AF2 recycling iterations |
| `use_multimer` | `bool` | `False` | Use AF2-Multimer |

**Returns:** `dict` mapping `example_id` to `{"sc_ca_rmsd", "avg_ca_plddt", "tmalign_score"}`.

### `make_constraints(constraints) -> DataFrame`

Build a positional constraint DataFrame from a dict.

```python
make_constraints({"pdb_key": {"fixed_pos_seq": "A1-50", ...}})
```

### `make_ensemble_constraints(constraints, pdb_to_conformers) -> DataFrame`

Build a constraint DataFrame expanded across ensemble conformers. Wraps `make_constraints()` and replicates each PDB's constraints for every conformer.

```python
make_ensemble_constraints(
    {"7xhz": {"fixed_pos_seq": "A6-15"}},
    {"7xhz": ["7xhz.cif", "conf_0.pdb", "conf_1.pdb"]},
)
```

### `generate_ensembles()`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pdb_paths` | `list[str]` | required | Paths to input PDB/CIF files |
| `out_dir` | `str` | required | Output directory |
| `num_samples_per_pdb` | `int` | `32` | Conformers per structure |
| `batch_size` | `int` | `8` | Batch size for Protpardelle |
| `model_params_path` | `str \| None` | `None` | Model weights directory |
| `sampling_yaml_path` | `str \| None` | `None` | Protpardelle config path |
| `seed` | `int` | `0` | Random seed |

**Returns:** `dict` mapping PDB stem to list of generated conformer file paths.
