"""
Modified encoding functions from AtomWorks to encode coordinates while keeping a
mapping from encoding indices to atom array indices.
"""

from typing import Any

import numpy as np
from atomworks.common import KeyToIntMapper, exists
from atomworks.constants import ELEMENT_NAME_TO_ATOMIC_NUMBER
from atomworks.io.utils.ccd import get_std_to_alt_atom_name_map
from atomworks.ml.encoding_definitions import TokenEncoding
from atomworks.ml.transforms._checks import check_atom_array_annotation
from atomworks.ml.transforms.base import Transform
from atomworks.ml.utils.token import get_token_count, token_iter
from biotite.structure import AtomArray


def atom_array_to_encoding_with_mapping(
    atom_array: AtomArray,
    encoding: TokenEncoding,
    default_coord: np.ndarray | float = float("nan"),
    occupancy_threshold: float = 0.0,
    extra_annotations: list[str] = [
        "chain_id",
        "chain_entity",
        "molecule_iid",
        "chain_iid",
        "transformation_id",
    ],
) -> dict:
    """
    Encode an atom array using a specified `TokenEncoding`, keeping a mapping
    from encoding indices to atom array indices.

    Returns:
        dict with keys:
            - `xyz` [n_token, n_atoms_per_token, 3]
            - `mask` [n_token, n_atoms_per_token]
            - `seq` [n_token]
            - `token_is_atom` [n_token]
            - `atom_to_within_token_map` [n_atom]
    """
    # Extract atom array information
    n_token = get_token_count(atom_array)
    n_atom = len(atom_array)

    # Init encoded arrays
    encoded_coord = np.full(
        (n_token, encoding.n_atoms_per_token, 3), fill_value=default_coord, dtype=np.float32
    )
    encoded_mask = np.zeros((n_token, encoding.n_atoms_per_token), dtype=bool)
    encoded_atom_to_within_token_map = np.zeros(n_atom, dtype=int)
    encoded_seq = np.empty(n_token, dtype=int)
    encoded_token_is_atom = np.empty(n_token, dtype=bool)

    # init additional annotation
    extra_annot_counters = {}
    extra_annot_encoded = {}
    for key in extra_annotations:
        if key in atom_array.get_annotation_categories():
            extra_annot_counters[key] = KeyToIntMapper()
            extra_annot_encoded[key] = []

    has_atomize = "atomize" in atom_array.get_annotation_categories()
    atom_idx = 0
    for i, token in enumerate(token_iter(atom_array)):
        if (has_atomize and token.atomize[0]) or len(token) == 1:
            token_name = (
                token.atomic_number[0]
                if "atomic_number" in token.get_annotation_categories()
                else ELEMENT_NAME_TO_ATOMIC_NUMBER[token.element[0].upper()]
            )
            token_is_atom = True
        else:
            token_name = token.res_name[0]
            token_is_atom = False

        if token_name not in encoding.token_to_idx:
            token_name = encoding.resolve_unknown_token_name(token_name, token_is_atom)
            assert token_name in encoding.token_to_idx, f"Unknown token name: {token_name}"

        encoded_seq[i] = encoding.token_to_idx[token_name]
        encoded_token_is_atom[i] = token_is_atom

        for atom in token:
            atom_name = str(token_name) if token_is_atom else atom.atom_name

            if (token_name, atom_name) in encoding.atom_to_idx:
                to_idx = encoding.atom_to_idx[(token_name, atom_name)]
                encoded_coord[i, to_idx, :] = atom.coord
                encoded_mask[i, to_idx] = atom.occupancy > occupancy_threshold
                encoded_atom_to_within_token_map[atom_idx] = to_idx

            elif token_name in encoding.unknown_tokens:
                continue

            elif not token_is_atom:
                alt_to_std = get_std_to_alt_atom_name_map(token_name)
                alt_atom_name = alt_to_std.get(atom_name, None)
                if exists(alt_atom_name) and (token_name, alt_atom_name) in encoding.atom_to_idx:
                    to_idx = encoding.atom_to_idx[(token_name, alt_atom_name)]
                    encoded_coord[i, to_idx, :] = atom.coord
                    encoded_atom_to_within_token_map[atom_idx] = to_idx

            else:
                msg = f"Atom ({token_name}, {atom_name}) not in encoding for token `{token_name}`"
                msg += "\nProblematic atom:\n"
                msg += f"{atom}"
                raise ValueError(msg)

            atom_idx += 1

        for key in extra_annot_counters:
            annot = token.get_annotation(key)[0]
            extra_annot_encoded[key].append(extra_annot_counters[key](annot))

    return {
        "xyz": encoded_coord,
        "mask": encoded_mask,
        "seq": encoded_seq,
        "token_is_atom": encoded_token_is_atom,
        "atom_to_within_token_map": encoded_atom_to_within_token_map,
        **{annot: np.array(extra_annot_encoded[annot], dtype=np.int16) for annot in extra_annot_encoded},
        **{annot + "_to_int": extra_annot_counters[annot].key_to_id for annot in extra_annot_counters},
    }


class EncodeAtomArrayWithMapping(Transform):
    """Encode an atom array to an arbitrary `TokenEncoding`, keeping atom-to-within-token mapping."""

    def __init__(
        self,
        encoding: TokenEncoding,
        default_coord: float | np.ndarray = float("nan"),
        occupancy_threshold: float = 0.0,
        extra_annotations: list[str] = [
            "chain_id",
            "chain_entity",
            "molecule_iid",
            "chain_iid",
            "transformation_id",
        ],
    ):
        if not isinstance(encoding, TokenEncoding):
            raise ValueError(f"Encoding must be a `TokenEncoding`, but got: {type(encoding)}.")
        self.encoding = encoding
        self.default_coord = default_coord
        self.occupancy_threshold = occupancy_threshold
        self.extra_annotations = extra_annotations

    def check_input(self, data: dict[str, Any]) -> None:
        check_atom_array_annotation(data, ["occupancy"])

    def forward(self, data: dict[str, Any]) -> dict[str, Any]:
        atom_array = data["atom_array"]

        encoded = atom_array_to_encoding_with_mapping(
            atom_array,
            encoding=self.encoding,
            default_coord=self.default_coord,
            occupancy_threshold=self.occupancy_threshold,
            extra_annotations=self.extra_annotations,
        )

        data["encoded"] = encoded
        return data
