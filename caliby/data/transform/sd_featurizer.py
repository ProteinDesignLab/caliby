from typing import Any, override

import atomworks.constants as aw_const
import atomworks.enums as aw_enums
import numpy as np
import torch
from atomworks.ml.encoding_definitions import AF2_ATOM37_ENCODING
from atomworks.ml.transforms.atom_array import (
    AddGlobalTokenIdAnnotation,
    AddWithinChainInstanceResIdx,
    AddWithinPolyResIdxAnnotation,
    ComputeAtomToTokenMap,
)
from atomworks.ml.transforms.base import (
    Compose,
    ConvertToTorch,
    Identity,
    RandomRoute,
    RemoveKeys,
    SubsetToKeys,
    Transform,
)
from atomworks.ml.transforms.bonds import AddAF3TokenBondFeatures
from atomworks.ml.transforms.crop import CropContiguousLikeAF3, CropSpatialLikeAF3
from atomworks.ml.transforms.encoding import EncodeAF3TokenLevelFeatures
from atomworks.ml.transforms.featurize_unresolved_residues import (
    MaskResiduesWithSpecificUnresolvedAtoms,
    PlaceUnresolvedTokenAtomsOnRepresentativeAtom,
    PlaceUnresolvedTokenOnClosestResolvedTokenInSequence,
)
from atomworks.ml.transforms.filters import (
    FilterToProteins,
    RemoveUnresolvedTokens,
    RemoveUnsupportedChainTypes,
    filter_to_specified_pn_units,
)
from atomworks.ml.utils.geometry import masked_center, random_rigid_augmentation
from atomworks.ml.utils.token import (
    apply_token_wise,
    get_af3_token_center_idxs,
)

import caliby.data.const as const
import caliby.data.transform.encoding as encoding
from caliby.data.transform.pad import pad_dim

# Keep track of the token/atom dimensions of the features for padding & cropping
FEAT_TO_TOKEN_DIM = {
    # Maps feature name to the token dimension
    # token features
    "residue_index": [0],
    "token_index": [0],
    "asym_id": [0],
    "entity_id": [0],
    "sym_id": [0],
    "restype": [0],
    "is_protein": [0],
    "is_rna": [0],
    "is_dna": [0],
    "is_ligand": [0],
    "is_atomized": [0],
    "token_bonds": [0, 1],
    "token_to_center_atom": [0],
    "token_pad_mask": [0],
    "token_resolved_mask": [0],
    # optional features that might not be present
    "seq_cond_mask": [0],
    "token_exists_mask": [0],
    # AF2-encoded features
    "encoded_seq": [0],
    "encoded_xyz": [0],
    "encoded_mask": [0],
    "encoded_standard_atom_mask": [0],
    "encoded_atom_or_ghost_mask": [0],
    "encoded_token_is_atom": [0],
}

FEAT_TO_ATOM_DIM = {
    # Maps feature name to the atom dimension
    # atom features
    "coords": [0],
    "atom_pad_mask": [0],
    "atom_resolved_mask": [0],
    "atom_to_token_map": [0],
    "prot_bb_atom_mask": [0],
    "prot_scn_atom_mask": [0],
    "encoded_atom_to_within_token_map": [0],
    # optional features that might not be present
    "atom_cond_mask": [0],
}

# Keep track of data dict keys only included at inference time
INFERENCE_ONLY_KEYS = ["crop_info", "atom_array", "feat_metadata"]


def sd_featurizer(
    # se3 augmentation
    apply_random_augmentation: bool = False,
    translation_scale: float = 0.0,
    # cropping
    max_tokens: int | None = None,
    max_atoms: int | None = None,
    crop_center_cutoff_distance: float = 15.0,
    crop_spatial_p: float = 0.0,
    remove_keys: list[str] = [],
) -> Transform:
    """
    Build a transform pipeline that transforms a featurized structure into a training example (including cropping).

    Defaults are used at inference time.
    """
    # Featurization that must be done before cropping
    featurization_transforms_pre_crop = [
        MaskResiduesWithSpecificUnresolvedAtoms(
            chain_type_to_atom_names={
                aw_enums.ChainTypeInfo.PROTEINS: aw_const.PROTEIN_FRAME_ATOM_NAMES,
                aw_enums.ChainTypeInfo.NUCLEIC_ACIDS: aw_const.NUCLEIC_ACID_FRAME_ATOM_NAMES,
            }
        ),
        FilterToProteins(),
        FilterToQueryPNUnits(),
        RemoveUnresolvedTokens(),
        MaskAtomizedTokens(),
        RemoveUnsupportedChainTypes(),
        ErrIfAllUnresolved(),
        AddWithinChainInstanceResIdx(),
        AddWithinPolyResIdxAnnotation(),
    ]

    # Cropping
    crop_contiguous_p = 1.0 - crop_spatial_p
    cropping_transform = Identity()
    if max_tokens is not None:
        cropping_transform = RandomRoute(
            transforms=[
                CropContiguousLikeAF3(
                    crop_size=max_tokens,
                    keep_uncropped_atom_array=True,
                    max_atoms_in_crop=max_atoms,
                ),
                CropSpatialLikeAF3(
                    crop_size=max_tokens,
                    crop_center_cutoff_distance=crop_center_cutoff_distance,
                    keep_uncropped_atom_array=True,
                    max_atoms_in_crop=max_atoms,
                ),
            ],
            probs=[crop_contiguous_p, crop_spatial_p],
        )

    # Featurization
    # NOTE: for now, we ignore ref pos features because they are too slow to compute
    featurization_transforms_post_crop = [
        AddGlobalTokenIdAnnotation(),  # required for reference molecule features and TokenToAtomMap
        EncodeAF3TokenLevelFeatures(sequence_encoding=const.AF3_ENCODING),
        ComputeAtomToTokenMap(),
        AddAF3TokenBondFeatures(),
        ConvertToTorch(keys=["encoded", "feats"]),
        # Handle missing atoms and tokens
        PlaceUnresolvedTokenAtomsOnRepresentativeAtom(annotation_to_update="coord"),
        PlaceUnresolvedTokenOnClosestResolvedTokenInSequence(annotation_to_update="coord", annotation_to_copy="coord"),
        # Add features from the atom_array
        FeaturizeCoordsAndMasks(),
        CenterRandomAugmentation(
            apply_random_augmentation=apply_random_augmentation,
            translation_scale=translation_scale,
            update_atom_array=True,
        ),
        # Featurize atom37 coordinates.
        encoding.EncodeAtomArrayWithMapping(encoding=AF2_ATOM37_ENCODING, default_coord=0.0, extra_annotations=[]),
        FeaturizeEncodedMasks(),
        ConvertToTorch(keys=["encoded"]),
    ]

    transforms = [
        *featurization_transforms_pre_crop,
        cropping_transform,
        *featurization_transforms_post_crop,
        PadSDFeats(max_tokens=max_tokens, max_atoms=max_atoms),
        SubsetToKeys(keys=["example_id", "feats", "encoded", *INFERENCE_ONLY_KEYS]),
        FlattenSDFeats(),
        RemoveKeys(keys=remove_keys),
    ]

    return Compose(transforms)


class FeaturizeCoordsAndMasks(Transform):
    """Add coordinates and atom masks to feats."""

    @override
    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        atom_array = data["atom_array"]
        feats = data["feats"]

        # Get coordinates
        feats["coords"] = torch.tensor(atom_array.coord)

        # Get token and atom resolved masks
        feats["token_resolved_mask"] = torch.tensor(
            apply_token_wise(atom_array, atom_array.occupancy > 0, np.any)
        ).float()  # MaskResiduesWithSpecificUnresolvedAtoms: If N, Ca, or C were unresolved, the whole residue is masked.
        feats["atom_resolved_mask"] = torch.tensor(atom_array.occupancy > 0).float()

        # Make pad masks
        feats["token_pad_mask"] = torch.ones_like(feats["token_resolved_mask"])
        feats["atom_pad_mask"] = torch.ones_like(feats["atom_resolved_mask"])

        # Get protein backbone and sidechain atom masks
        feats["atom_to_token_map"] = feats["atom_to_token_map"].long()
        atomwise_is_prot = feats["is_protein"].gather(dim=-1, index=feats["atom_to_token_map"])

        atomized = torch.tensor(atom_array.atomize)
        bb_atom_mask = torch.tensor(atom_array.is_backbone_atom)

        feats["prot_bb_atom_mask"] = bb_atom_mask * ~atomized * atomwise_is_prot
        feats["prot_scn_atom_mask"] = ~bb_atom_mask * ~atomized * atomwise_is_prot
        feats["token_to_center_atom"] = torch.tensor(get_af3_token_center_idxs(atom_array))

        return data


class PadSDFeats(Transform):
    """Pad the token and atom features to the maximum number of tokens and atoms."""

    def __init__(self, max_tokens: int | None, max_atoms: int | None):
        self.max_tokens = max_tokens
        self.max_atoms = max_atoms

    @override
    def forward(
        self,
        data: dict[str, Any],
    ) -> dict[str, Any]:
        feats = data["feats"]

        # Pad to max tokens if given
        if self.max_tokens is not None:
            token_pad_len = self.max_tokens - len(feats["token_index"])
            if token_pad_len > 0:
                for k, v in FEAT_TO_TOKEN_DIM.items():
                    if k not in feats:
                        continue
                    for dim_to_pad in v:
                        feats[k] = pad_dim(feats[k], dim_to_pad, token_pad_len)

        # Pad to max atoms if given
        if self.max_atoms is not None:
            atom_pad_len = self.max_atoms - len(feats["atom_resolved_mask"])
            if atom_pad_len > 0:
                for k, v in FEAT_TO_ATOM_DIM.items():
                    if k not in feats:
                        continue
                    for dim_to_pad in v:
                        feats[k] = pad_dim(feats[k], dim_to_pad, atom_pad_len)

        return data


class FlattenSDFeats(Transform):
    """Flatten features into the data dict.

    Flattens "feats" key into the data dict.
    If "encoded" is in the data dict, flatten it into the data dict with prefix 'encoded_'.
    """

    @override
    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        feats = data.pop("feats")
        for k, v in feats.items():
            data[k] = v

        if "encoded" in data:
            encoded = data.pop("encoded")
            for k, v in encoded.items():
                data[f"encoded_{k}"] = v
        return data


class FeaturizeEncodedMasks(Transform):
    """Featurize encoded masks:
    - standard_atom_mask: 1 for atoms that are in the residue type; not ghost atoms
    - atom_or_ghost_mask: 1 for atoms that are present in the PDB file or are ghost atoms
    """

    @override
    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        encoded = data["encoded"]
        # standard_atom_mask: 1 for atoms that are in the residue type; standard atoms.
        standard_atom_mask = const.AF2_STANDARD_ATOM_MASK_WITH_X[encoded["seq"]]

        # atom_or_ghost_mask: 1 for atoms that are present in the PDB file or are ghost atoms.
        atom_or_ghost_mask = encoded["mask"] | (1 - standard_atom_mask)

        data["encoded"]["standard_atom_mask"] = standard_atom_mask.astype(float)
        data["encoded"]["atom_or_ghost_mask"] = atom_or_ghost_mask.astype(float)
        return data


class CenterRandomAugmentation(Transform):
    """Center the atom array. If apply_random_augmentation is True, also randomly rotate and translate.
    If update_atom_array is True, update the atom array in the data dict.
    """

    def __init__(self, apply_random_augmentation: bool, translation_scale: float, update_atom_array: bool = False):
        self.apply_random_augmentation = apply_random_augmentation
        self.translation_scale = translation_scale
        self.update_atom_array = update_atom_array

    @override
    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        coords = data["feats"]["coords"]
        mask = data["feats"]["atom_resolved_mask"].bool()
        centered_coords = coords.clone()
        centered_coords = masked_center(centered_coords, mask)
        if self.apply_random_augmentation:
            centered_coords = random_rigid_augmentation(
                centered_coords[None], batch_size=1, s=self.translation_scale
            ).squeeze(0)
        data["feats"]["coords"] = centered_coords

        if self.update_atom_array:
            data["atom_array"].coord = centered_coords.numpy()

        return data


class FilterToQueryPNUnits(Transform):
    """Filter the atom array to the query PN units."""

    @override
    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        atom_array = data["atom_array"]

        # * From atomworks.ml.datasets.parsers.GenericDFParser: "During VALIDATION, then we do not crop, and query_pn_unit_iids should be None."
        if "query_pn_unit_iids" in data:
            atom_array = filter_to_specified_pn_units(atom_array, data["query_pn_unit_iids"])

        data["atom_array"] = atom_array
        return data


class MaskAtomizedTokens(Transform):
    """Mask atomized tokens from the atom array."""

    @override
    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        atom_array = data["atom_array"]
        atom_array.occupancy[atom_array.atomize] = 0
        data["atom_array"] = atom_array
        return data


class ErrIfAllUnresolved(Transform):
    """Throw an error if all atoms are unresolved."""

    @override
    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        atom_array = data["atom_array"]
        if (atom_array.occupancy == 0).all():
            raise ValueError(f"All atoms are unresolved for {data['example_id']}")
        return data
