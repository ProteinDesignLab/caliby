"""Caliby constants."""

from collections.abc import Sequence
from functools import cached_property
from itertools import cycle
from typing import Final

import biotite.structure as struc
import numpy as np
from atomworks.constants import (
    AA_LIKE_CHEM_TYPES,
    DNA_LIKE_CHEM_TYPES,
    GAP,
    RNA_LIKE_CHEM_TYPES,
    STANDARD_AA,
    STANDARD_DNA,
    STANDARD_RNA,
    UNKNOWN_AA,
    UNKNOWN_DNA,
    UNKNOWN_RNA,
)
from atomworks.io.utils import sequence as aw_sequence
from atomworks.ml.encoding_definitions import AF2_ATOM14_ENCODING, AF2_ATOM37_ENCODING

"""Sequence tokens in AF3"""

# fmt: off
AF3_TOKENS = (
    # 20 AA + 1 unknown AA
    *STANDARD_AA, UNKNOWN_AA,
    # 4 RNA + 1 unknown RNA
    *STANDARD_RNA, UNKNOWN_RNA,
    # 4 DNA + 1 unknown DNA
    *STANDARD_DNA, UNKNOWN_DNA,
    # 1 gap
    GAP,
)
# fmt: on


class AF3SequenceEncoding:
    """
    Encodes and decodes sequence tokens for AlphaFold 3.

    This class provides functionality to convert between residue names and their
    corresponding integer encodings as used in AlphaFold 3. It handles standard
    amino acids, RNA, DNA, and unknown residues.

    Methods:
        encode(res_names): Encode residue names to integer indices.
        decode(res_indices): Decode integer indices to residue names.
        tokens: Property that returns the list of AF3 tokens.
        n_tokens: Property that returns the number of AF3 tokens.
    """

    def __init__(self):
        # Load CCD from biotite
        ccd = struc.info.ccd.get_ccd()

        # Get all residue names and their corresponding chemtypes
        self.all_res_names = ccd["chem_comp"]["id"].as_array()
        self.all_res_chemtypes = np.char.upper(ccd["chem_comp"]["type"].as_array())

        # Get boolean arrays for each chemtype

        self.is_rna_like = np.isin(self.all_res_chemtypes, list(RNA_LIKE_CHEM_TYPES))
        self.is_dna_like = np.isin(self.all_res_chemtypes, list(DNA_LIKE_CHEM_TYPES))
        self.is_aa_like = np.isin(self.all_res_chemtypes, list(AA_LIKE_CHEM_TYPES))

        # Build mappings for all CCD residue names to AF3 tokens
        res_name_to_token = dict(zip(self.all_res_names[self.is_rna_like], cycle([UNKNOWN_RNA])))
        res_name_to_token |= dict(zip(self.all_res_names[self.is_dna_like], cycle([UNKNOWN_DNA])))
        res_name_to_token |= dict(zip(AF3_TOKENS, AF3_TOKENS, strict=False))
        self.res_name_to_token = res_name_to_token

        # Build mappings for AF3 tokens to indices
        self.af3_token_to_int = {token: i for i, token in enumerate(AF3_TOKENS)}

    @property
    def tokens(self) -> list[str]:
        return AF3_TOKENS

    def res_name_to_af3_token(self, res_name: str) -> str:
        return np.vectorize(lambda res_name: self.res_name_to_token.get(res_name, UNKNOWN_AA))(res_name)

    @property
    def token_to_idx(self) -> dict[str, int]:
        return self.af3_token_to_int

    @cached_property
    def idx_to_token(self) -> np.ndarray:
        return np.array(AF3_TOKENS)

    @property
    def n_tokens(self) -> int:
        return len(self.tokens)

    @property
    def protein_tokens(self) -> list[str]:
        return [token for token in self.tokens if token in PROT_LETTER_TO_TOKEN.values()]

    @property
    def non_protein_tokens(self) -> list[str]:
        return [token for token in self.tokens if token not in PROT_LETTER_TO_TOKEN.values()]

    def encode(self, res_names: Sequence[str]) -> list[int]:
        return [self.af3_token_to_int.get(x, self.af3_token_to_int[UNKNOWN_AA]) for x in res_names]

    def decode(self, token_idxs: int | Sequence[int]) -> list[str]:
        if isinstance(token_idxs, int):
            token_idxs = [token_idxs]
        return [self.idx_to_token[idx] for idx in token_idxs]

    def encode_aa(self, aa: str) -> int:
        """First converts 1-letter AA name to protein token, then encodes to integer index."""
        return self.af3_token_to_int.get(PROT_LETTER_TO_TOKEN[aa], self.af3_token_to_int[UNKNOWN_AA])

    def encode_aa_seq(self, aa_seq: Sequence[str]) -> list[int]:
        """First converts 1-letter AA names to protein tokens, then encodes to integer indices."""
        return [self.encode_aa(aa) for aa in aa_seq]

    def decode_aa_seq(self, token_idxs: Sequence[int]) -> str:
        """First converts integer indices to tokens, then decodes to a string."""
        return "".join([PROT_TOKEN_TO_LETTER[token] for token in self.decode(token_idxs)])


AF3_ENCODING: Final[AF3SequenceEncoding] = AF3SequenceEncoding()

MAX_NUM_ATOMS: Final[int] = 23
PROT_BB_ATOMS: Final[list[str]] = ["N", "CA", "C", "O"]
PROT_LETTER_TO_TOKEN: Final[dict[str, str]] = {
    **aw_sequence.aa_chem_comp_1to3(),
    "X": "UNK",
}  # include "X" for unknown amino acids
PROT_TOKEN_TO_LETTER: Final[dict[str, str]] = {v: k for k, v in PROT_LETTER_TO_TOKEN.items()}

# Reduced alphabet for Potts model.
POTTS_TOKENS: Final[tuple[str, ...]] = (*STANDARD_AA, UNKNOWN_AA, GAP)
POTTS_AF3_TOKEN_IDXS: Final[np.ndarray] = np.array(AF3_ENCODING.encode(POTTS_TOKENS), dtype=np.int64)
AF3_TO_POTTS_TOKEN_IDX: Final[np.ndarray] = np.full(AF3_ENCODING.n_tokens, -1, dtype=np.int64)
AF3_TO_POTTS_TOKEN_IDX[POTTS_AF3_TOKEN_IDXS] = np.arange(len(POTTS_TOKENS), dtype=np.int64)

# When generating partial diffusion ensembles, other resnames get ignored by Protpardelle-1c.
PROTPARDELLE_SUPPORTED_RESNAMES: Final[list[str]] = [*STANDARD_AA, UNKNOWN_AA, "MSE"]

# AF2 encoding constants
AF2_BB_ATOMS: Final[list[str]] = ["N", "CA", "C", "O"]
AF2_BB_ATOM_ORDER: Final[dict[str, int]] = {atom: i for i, atom in enumerate(AF2_BB_ATOMS)}
AF2_BB_IDXS: Final[list[int]] = [0, 1, 2, 4]
AF2_SCN_IDXS: Final[list[int]] = [i for i in range(37) if i not in AF2_BB_IDXS]
RES_NAME_TO_SYMMETRIC_CHI: Final[dict[str, list[int]]] = {
    "ASP": [1],  # chi1
    "GLU": [2],  # chi2
    "PHE": [1],  # chi1
    "TYR": [1],  # chi1
}

AF2_STANDARD_ATOM_MASK_WITH_X: np.ndarray[bool] = AF2_ATOM37_ENCODING.idx_to_atom != ""
AF2_STANDARD_ATOM_MASK_WITH_X[:, -1] = False  # remove OXT atoms
AF2_STANDARD_ATOM_MASK_WITH_X[AF2_ATOM37_ENCODING.token_to_idx["UNK"]][AF2_BB_IDXS] = (
    True  # keep backbone atoms for unknown amino acids
)


def _make_restype_atom14_to_atom37() -> np.ndarray:
    """
    Map from atom14 to atom37 per residue type.
    For UNK tokens, we include backbone atoms.
    """
    restype_atom14_to_atom37 = []  # mapping (restype, atom14) --> atom37
    atom37_tokens = AF2_ATOM37_ENCODING.tokens
    atom14_tokens = AF2_ATOM14_ENCODING.tokens
    atom37_atoms = AF2_ATOM37_ENCODING.idx_to_atom  # [n_tokens, 37]
    atom14_atoms = AF2_ATOM14_ENCODING.idx_to_atom  # [n_tokens, 14]

    # Ensure tokens are in the same order.
    assert np.array_equal(atom37_tokens, atom14_tokens), "Token orders must match between encodings"

    # Create mapping from atom name to index for atom37.
    atom37_token_to_atom_to_idx = {}
    for token_idx, token in enumerate(atom37_tokens):
        atom_name_to_idx = {}
        for atom_idx, atom_name in enumerate(atom37_atoms[token_idx]):
            atom_name_stripped = atom_name.strip()
            if atom_name_stripped and atom_name_stripped not in atom_name_to_idx:
                atom_name_to_idx[atom_name_stripped] = atom_idx
        atom37_token_to_atom_to_idx[token] = atom_name_to_idx

    # Build mapping.
    for token_idx, token in enumerate(atom14_tokens):
        atom14_mapping = []
        atom14_atom_names = atom14_atoms[token_idx]
        atom37_atom_name_to_idx = atom37_token_to_atom_to_idx.get(token, {})

        for atom14_idx, atom14_atom_name in enumerate(atom14_atom_names):
            atom14_atom_name_stripped = atom14_atom_name.strip()
            if atom14_atom_name_stripped and atom14_atom_name_stripped in atom37_atom_name_to_idx:
                atom37_idx = atom37_atom_name_to_idx[atom14_atom_name_stripped]
            else:
                atom37_idx = 0
            atom14_mapping.append(atom37_idx)

        restype_atom14_to_atom37.append(atom14_mapping)

    restype_atom14_to_atom37 = np.array(restype_atom14_to_atom37, dtype=np.int32)

    # Add backbone atoms for UNK tokens.
    unk_token_idx = AF2_ATOM37_ENCODING.token_to_idx["UNK"]
    gly_token_idx = AF2_ATOM37_ENCODING.token_to_idx["GLY"]
    restype_atom14_to_atom37[unk_token_idx, :] = restype_atom14_to_atom37[gly_token_idx, :]

    return restype_atom14_to_atom37


AF2_RESTYPE_ATOM14_TO_ATOM37: Final[np.ndarray] = _make_restype_atom14_to_atom37()
AF2_RESTYPE_ATOM14_MASK_WITH_X: np.ndarray = AF2_ATOM14_ENCODING.idx_to_atom != ""
AF2_RESTYPE_ATOM14_MASK_WITH_X[AF2_ATOM14_ENCODING.token_to_idx["UNK"]][[0, 1, 2, 3]] = True  # keep bb atoms for UNK
