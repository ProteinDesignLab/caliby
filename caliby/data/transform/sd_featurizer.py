import atomworks.constants as aw_const
import atomworks.enums as aw_enums
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
)

import caliby.data.const as const
import caliby.data.transform.encoding as encoding
from caliby.data.transform.sd_featurizer_transforms import (
    CenterRandomAugmentation,
    ErrIfAllUnresolved,
    FeaturizeCoordsAndMasks,
    FeaturizeEncodedMasks,
    FilterToQueryPNUnits,
    FlattenSDFeats,
    MaskAtomizedTokens,
    PadSDFeats,
)

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
        MaskAtomizedTokens(),
        RemoveUnsupportedChainTypes(),
        RemoveUnresolvedTokens(),
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
