"""Functions for IO of structure files in addition to AtomWorks."""

from atomworks.io.utils.io_utils import load_any, to_pdb_string


def cif_to_pdb(cif_file: str, out_pdb: str) -> str:
    """Convert a CIF file to a PDB file."""
    atom_array = load_any(cif_file, model=1)
    with open(out_pdb, "w") as f:
        f.write(to_pdb_string(atom_array))
