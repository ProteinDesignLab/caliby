import numpy as np
from atomworks.ml.preprocessing.utils.structure_utils import get_inter_pn_unit_bond_mask
from atomworks.ml.transforms.base import Transform
from biotite.structure import AtomArray


class FlagCovalentModifications(Transform):
    """Flag covalent modifications without reassigning PN units or atomizing residues."""

    def forward(self, data: dict) -> dict:
        atom_array: AtomArray = data["atom_array"]

        if "is_covalent_modification" not in atom_array.get_annotation_categories():
            atom_array.set_annotation("is_covalent_modification", np.zeros(len(atom_array), dtype=bool))

        inter_pn_unit_bond_mask = get_inter_pn_unit_bond_mask(atom_array)
        bonds_to_check = atom_array.bonds.as_array()[inter_pn_unit_bond_mask]
        bonds_to_check = bonds_to_check[
            atom_array.is_polymer[bonds_to_check[:, 0]] != atom_array.is_polymer[bonds_to_check[:, 1]]
        ]

        for bond in bonds_to_check:
            atom_a, atom_b = atom_array[bond[0]], atom_array[bond[1]]
            polymer_atom = atom_a if atom_a.is_polymer else atom_b

            polymer_residue_mask = (
                (atom_array.res_id == polymer_atom.res_id)
                & (atom_array.chain_id == polymer_atom.chain_id)
                & (atom_array.pn_unit_iid == polymer_atom.pn_unit_iid)
            )
            atom_array.is_covalent_modification[polymer_residue_mask] = True

        data["atom_array"] = atom_array
        return data
