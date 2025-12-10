"""
Script to generate symmetry_pos strings for homooligomers.
"""

import argparse


def _make_symmetry_pos_string(chains: list[str], start_res: int, end_res: int) -> str:
    """
    Generate a symmetry_pos string for a homooligomer.

    Args:
        chains: List of chain IDs (e.g., ['A', 'B', 'C'])
        start_res: Starting residue number (inclusive)
        end_res: Ending residue number (inclusive)

    Returns:
        String in format "A10,B10,C10|A11,B11,C11|..." for use with parse_symmetry_pos_str
    """
    groups = []
    for res_num in range(start_res, end_res + 1):
        group = ",".join(f"{chain}{res_num}" for chain in chains)
        groups.append(group)
    return "|".join(groups)


def main():
    parser = argparse.ArgumentParser(
        description="Generate symmetry_pos string for homooligomer symmetry constraints"
    )
    parser.add_argument(
        "--chains",
        nargs="+",
        required=True,
        help="Chain IDs to tie together (e.g., A B C)",
    )
    parser.add_argument(
        "--residue-range",
        nargs=2,
        type=int,
        required=True,
        metavar=("START", "END"),
        help="Residue range to tie (inclusive, e.g., 10 20)",
    )

    args = parser.parse_args()

    if len(args.chains) < 2:
        parser.error("At least 2 chains must be provided")

    start_res, end_res = args.residue_range
    symmetry_pos_str = _make_symmetry_pos_string(args.chains, start_res, end_res)
    print(symmetry_pos_str)


if __name__ == "__main__":
    main()
