#!/usr/bin/env python3
"""
native_iptm2.py

Detect optional confidence files and normalize native chain-pair ipTM values.

Supported:
  AF3 / Protenix-2: chain_pair_iptm matrix
  Boltz-2: pair_chains_iptm nested numeric dict
  Chai-1: per_chain_pair_iptm array, including shape (1,N,N)
  OpenFold3: chain_pair_iptm dict with "(A, B)" keys
  Promera: iptm dict with "A_B" keys
  ESMFold2 convenience files: pair_chains_iptm matrix

Output DataFrame columns:
  chain_i, chain_j, native_iptm, directional, source_key
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Optional

import numpy as np
import pandas as pd


PAIR_IPTM_KEYS = (
    "chain_pair_iptm",
    "pair_chains_iptm",
    "per_chain_pair_iptm",
    "iptm",
)

PAE_KEYS = (
    "pae",
    "token_pair_pae",
    "predicted_aligned_error",
)


def file_suffix(path: str) -> str:
    name = os.path.basename(path).lower()
    if name.endswith(".mmcif") or name.endswith(".cif"):
        return "cif"
    if name.endswith(".pdb"):
        return "pdb"
    if name.endswith(".npz"):
        return "npz"
    if name.endswith(".json"):
        return "json"
    return ""


def load_json_or_npz(path: str) -> dict[str, Any]:
    suffix = file_suffix(path)

    if suffix == "json":
        with open(path, "r", encoding="utf-8") as handle:
            obj = json.load(handle)
        # AF2/ColabFold PAE JSON may be a one-element list of dictionaries.
        if isinstance(obj, list) and len(obj) == 1 and isinstance(obj[0], dict):
            obj = obj[0]
        if not isinstance(obj, dict):
            raise ValueError(f"Expected JSON dictionary (or one-element list) in {path}")
        return obj

    if suffix == "npz":
        with np.load(path, allow_pickle=False) as npz:
            return {key: np.asarray(npz[key]) for key in npz.files}

    raise ValueError(f"Expected JSON or NPZ file, got: {path}")


def looks_like_confidence_obj(obj: dict[str, Any]) -> bool:
    for key in PAIR_IPTM_KEYS:
        if key not in obj:
            continue

        value = obj[key]

        # Scalar overall ipTM is not a pairwise matrix.
        if key == "iptm" and not isinstance(value, dict):
            arr = np.asarray(value)
            if arr.ndim == 0 or arr.size == 1:
                continue

        return True

    return False


def looks_like_pae_obj(
    obj: dict[str, Any],
    pae_key: str = "pae",
) -> bool:
    keys = (pae_key,) + tuple(k for k in PAE_KEYS if k != pae_key)

    for key in keys:
        if key not in obj:
            continue
        arr = np.asarray(obj[key])
        if arr.ndim == 2 and arr.shape[0] == arr.shape[1]:
            return True

    return False


def identify_input_files(
    paths: list[str],
    pae_key: str = "pae",
    verbose: bool = True,
) -> tuple[str, str, Optional[str]]:
    """
    Identify:
      required coordinate file
      required PAE file
      optional native-confidence file

    Input order is arbitrary.
    """
    if len(paths) not in {2, 3}:
        raise ValueError(
            f"Expected 2 or 3 input files, got {len(paths)}: {paths}"
        )

    structures = [
        path for path in paths
        if file_suffix(path) in {"pdb", "cif"}
    ]

    if len(structures) != 1:
        raise ValueError(
            "Could not identify exactly one coordinate file "
            "(.pdb/.cif/.mmcif)"
        )

    structure_file = structures[0]
    other_files = [p for p in paths if p != structure_file]

    info = []
    for path in other_files:
        obj = load_json_or_npz(path)
        info.append(
            (
                path,
                obj,
                looks_like_pae_obj(obj, pae_key),
                looks_like_confidence_obj(obj),
            )
        )

    pae_candidates = [
        path for path, _, is_pae, _ in info if is_pae
    ]

    if len(pae_candidates) != 1:
        details = "; ".join(
            f"{path}: PAE={is_pae}, confidence={is_conf}"
            for path, _, is_pae, is_conf in info
        )
        raise ValueError(
            "Could not identify exactly one PAE file. " + details
        )

    pae_file = pae_candidates[0]

    remaining = [p for p in other_files if p != pae_file]
    confidence_file = None

    if remaining:
        confidence_file = remaining[0]
        obj = next(obj for p, obj, _, _ in info if p == confidence_file)

        if not looks_like_confidence_obj(obj):
            raise ValueError(
                f"Third file has no recognized pairwise ipTM field: "
                f"{confidence_file}"
            )

    if verbose:
        print()
        print("Identified input files:")
        print(f"  coordinates: {structure_file}")
        print(f"  PAE:         {pae_file}")
        print(f"  confidence:  {confidence_file or '(none)'}")

    return pae_file, structure_file, confidence_file


def parse_pair_key(key: str) -> Optional[tuple[str, str]]:
    s = str(key).strip()

    # OpenFold3: "(A, B)"
    match = re.fullmatch(
        r"\(\s*([^,]+?)\s*,\s*([^)]+?)\s*\)",
        s,
    )
    if match:
        return match.group(1).strip(), match.group(2).strip()

    # Promera or similar: A_B or A-B
    for sep in ("_", "-"):
        if sep in s:
            left, right = s.split(sep, 1)
            if left and right:
                return left.strip(), right.strip()

    return None


def matrix_to_pair_df(
    matrix: np.ndarray,
    chain_ids: list[str],
    source_key: str,
) -> pd.DataFrame:
    arr = np.asarray(matrix, dtype=float)
    arr = np.squeeze(arr)

    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(
            f"{source_key} must reduce to a square matrix; "
            f"got shape {arr.shape}"
        )

    if arr.shape[0] != len(chain_ids):
        raise ValueError(
            f"{source_key} has {arr.shape[0]} chains, but coordinates "
            f"have {len(chain_ids)} chains: {chain_ids}"
        )

    symmetric = np.allclose(
        arr,
        arr.T,
        rtol=1e-6,
        atol=1e-8,
        equal_nan=True,
    )

    rows = []

    for i, chain_i in enumerate(chain_ids):
        for j, chain_j in enumerate(chain_ids):
            if i == j:
                continue

            # For symmetric matrices, report each pair once.

            value = arr[i, j]
            if np.isnan(value):
                continue

            rows.append(
                {
                    "chain_i": chain_i,
                    "chain_j": chain_j,
                    "native_iptm": float(value),
                    "directional": not symmetric,
                    "source_key": source_key,
                }
            )

    return pd.DataFrame(rows)


def nested_index_dict_to_matrix(
    value: dict[str, Any],
    n_chains: int,
) -> np.ndarray:
    arr = np.full((n_chains, n_chains), np.nan, dtype=float)

    for i_key, row in value.items():
        i = int(i_key)

        if not isinstance(row, dict):
            raise ValueError(
                "Expected nested dictionary for pair_chains_iptm"
            )

        for j_key, score in row.items():
            j = int(j_key)
            arr[i, j] = float(score)

    return arr




def boltz_nested_pair_to_df(
    value: dict[str, Any],
    chain_ids: list[str],
    source_key: str = "pair_chains_iptm",
) -> pd.DataFrame:
    """Convert Boltz pair_chains_iptm to ipSAE direction semantics.

    Boltz stores pair_chains_iptm[scored_chain][aligned_chain].  In the
    ipSAE tables, chain_i is the alignment/reference chain and chain_j is
    the scored chain.  Therefore the two indices are intentionally swapped
    when the DataFrame is constructed here.
    """
    rows = []
    for scored_key, row in value.items():
        scored_idx = int(scored_key)
        if not isinstance(row, dict):
            raise ValueError("Expected nested dictionary for pair_chains_iptm")
        if scored_idx < 0 or scored_idx >= len(chain_ids):
            raise ValueError(f"Boltz chain index out of range: {scored_idx}")

        for aligned_key, score in row.items():
            aligned_idx = int(aligned_key)
            if aligned_idx < 0 or aligned_idx >= len(chain_ids):
                raise ValueError(f"Boltz chain index out of range: {aligned_idx}")
            if scored_idx == aligned_idx:
                continue

            rows.append({
                "chain_i": chain_ids[aligned_idx],
                "chain_j": chain_ids[scored_idx],
                "native_iptm": float(score),
                "directional": True,
                "source_key": source_key,
            })

    return pd.DataFrame(rows)

def extract_native_pair_iptm(
    obj: dict[str, Any],
    chain_ids: list[str],
) -> tuple[Optional[pd.DataFrame], Optional[str]]:
    for key in PAIR_IPTM_KEYS:
        if key not in obj:
            continue

        value = obj[key]

        # Ignore scalar overall ipTM.
        if key == "iptm" and not isinstance(value, dict):
            arr = np.asarray(value)
            if arr.ndim == 0 or arr.size == 1:
                continue

        if isinstance(value, dict):
            # Boltz-2 stores pair_chains_iptm[scored][aligned].  Convert
            # explicitly so chain_i -> chain_j has the same meaning as
            # ipSAE: align on chain_i, score chain_j.
            if (
                key == "pair_chains_iptm"
                and value
                and all(str(k).isdigit() for k in value)
            ):
                return boltz_nested_pair_to_df(
                    value,
                    chain_ids,
                    key,
                ), key

            # Generic nested numeric matrix, if encountered for another key.
            if value and all(str(k).isdigit() for k in value):
                matrix = nested_index_dict_to_matrix(
                    value,
                    len(chain_ids),
                )
                return matrix_to_pair_df(
                    matrix,
                    chain_ids,
                    key,
                ), key

            # OF3 and Promera pair-label dictionaries.
            rows = []

            for pair_label, score in value.items():
                parsed = parse_pair_key(str(pair_label))
                if parsed is None:
                    continue

                chain_i, chain_j = parsed

                score = float(score)

                rows.append(
                    {
                        "chain_i": chain_i,
                        "chain_j": chain_j,
                        "native_iptm": score,
                        "directional": False,
                        "source_key": key,
                    }
                )
                
                rows.append(
                    {
                        "chain_i": chain_j,
                        "chain_j": chain_i,
                        "native_iptm": score,
                        "directional": False,
                        "source_key": key,
                    }
                )

            if rows:
                return pd.DataFrame(rows), key

            continue

        # AF3, Protenix-2, Chai-1, ESMFold2 convenience NPZ.
        arr = np.asarray(value)
        if arr.ndim >= 2:
            return matrix_to_pair_df(
                arr,
                chain_ids,
                key,
            ), key

    return None, None


def read_native_confidences(
    path: Optional[str],
    chain_ids: list[str],
    verbose: bool = False,
) -> tuple[Optional[dict[str, Any]], Optional[pd.DataFrame]]:
    if path is None:
        return None, None

    summary = load_json_or_npz(path)

    pair_df, source_key = extract_native_pair_iptm(
        summary,
        chain_ids,
    )
    # AF2/ColabFold often report only scalar overall ipTM. For a dimer,
    # overall ipTM is a valid pair score, but it is not directional. For
    # complexes with >2 chains it MUST NOT be copied into every pair.
    if (pair_df is None or pair_df.empty) and isinstance(summary, dict) and len(chain_ids) == 2:
        raw = summary.get("iptm")
        try:
            scalar = float(np.asarray(raw).item())
        except (TypeError, ValueError):
            scalar = float("nan")
        if np.isfinite(scalar):
            pair_df = pd.DataFrame([
                dict(chain_i=chain_ids[0],chain_j=chain_ids[1],native_iptm=scalar,directional=False,source_key="iptm"),
                dict(chain_i=chain_ids[1],chain_j=chain_ids[0],native_iptm=scalar,directional=False,source_key="iptm"),
            ])
            source_key = "iptm"

    if verbose:
        print()
        print(f"Native confidences from {path}:")
        if pair_df is None or pair_df.empty:
            print("  No recognized native pairwise ipTM values found.")
        else:
            print(f"  source key: {source_key}")
            print()
            print("Native pairwise ipTM:")
            print(pair_df[["chain_i", "chain_j", "native_iptm"]].to_string(
                index=False, float_format=lambda x: f"{x:.4f}"))

    return summary, pair_df
