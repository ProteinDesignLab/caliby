import numpy as np
from atomworks.constants import AF3_EXCLUDED_LIGANDS, STANDARD_AA, STANDARD_DNA, STANDARD_RNA
from atomworks.ml.preprocessing.utils.structure_utils import get_inter_pn_unit_bond_mask
from atomworks.ml.transforms.atom_array import (
    AddGlobalAtomIdAnnotation,
    AddGlobalTokenIdAnnotation,
    AddWithinChainInstanceResIdx,
    AddWithinPolyResIdxAnnotation,
    ComputeAtomToTokenMap,
)
from atomworks.ml.transforms.atomize import AtomizeByCCDName, FlagNonPolymersForAtomization
from atomworks.ml.transforms.base import (
    AddData,
    Compose,
    ConditionalRoute,
    ConvertToTorch,
    Identity,
    RandomRoute,
    SubsetToKeys,
    Transform,
)
from atomworks.ml.transforms.bfactor_conditioned_transforms import SetOccToZeroOnBfactor
from atomworks.ml.transforms.bonds import AddAF3TokenBondFeatures
from atomworks.ml.transforms.encoding import EncodeAF3TokenLevelFeatures, EncodeAtomArray
from atomworks.ml.transforms.featurize_unresolved_residues import (
    MaskPolymerResiduesWithUnresolvedFrameAtoms,
    PlaceUnresolvedTokenAtomsOnRepresentativeAtom,
    PlaceUnresolvedTokenOnClosestResolvedTokenInSequence,
)
from atomworks.ml.transforms.filters import (
    FilterToSpecifiedPNUnits,
    HandleUndesiredResTokens,
    RemoveHydrogens,
    RemoveNucleicAcidTerminalOxygen,
    RemovePolymersWithTooFewResolvedResidues,
    RemoveTerminalOxygen,
    RemoveUnresolvedPNUnits,
)
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


def preprocess_transform(
    # Preprocessing
    undesired_res_names: list[str] = AF3_EXCLUDED_LIGANDS,
    b_factor_min: float | None = None,
    b_factor_max: float | None = None,
) -> Transform:
    """
    Build a transform pipeline for featurizing a structure parsed by the AtomWorks CIF parser.
    """
    # Preprocesing
    preprocessing_transforms = [
        RemoveHydrogens(),
        # filter to non-clashing PN units
        FilterToSpecifiedPNUnits(extra_info_key_with_pn_unit_iids_to_keep="all_pn_unit_iids_after_processing"),
        RemoveTerminalOxygen(),
        SetOccToZeroOnBfactor(b_factor_min, b_factor_max),
        RemoveUnresolvedPNUnits(),
        RemovePolymersWithTooFewResolvedResidues(min_residues=4),
        MaskPolymerResiduesWithUnresolvedFrameAtoms(),
        # NOTE: For inference, we must keep UNL to support ligands that are not in the CCD
        HandleUndesiredResTokens(undesired_res_tokens=undesired_res_names),  # e.g., non-standard residues
        FlagCovalentModifications(),
        FlagNonPolymersForAtomization(),
        AddGlobalAtomIdAnnotation(allow_overwrite=True),
        AtomizeByCCDName(
            atomize_by_default=True,
            res_names_to_ignore=STANDARD_AA + STANDARD_RNA + STANDARD_DNA,
            move_atomized_part_to_end=False,
            validate_atomize=False,
        ),
        RemoveNucleicAcidTerminalOxygen(),
        AddWithinChainInstanceResIdx(),
        AddWithinPolyResIdxAnnotation(),
    ]

    return Compose(preprocessing_transforms)
