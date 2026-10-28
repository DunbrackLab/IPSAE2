"""ipSAE2 scoring. No spatial coordinate work is performed in this module."""
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd

TYPE_ORDER = {"prot": 0, "nuc": 1, "lig": 2, "ion": 3}
OUTPUT_COLUMNS = [
    "chn1", "chn2", "type1", "type2", "mol1", "mol2",
    "iptm", "ipsae", "ipsae12", "ipsae21", "ipsae_source",
    "align_token12", "align_token21", "iPAE12", "iPAE21",
    "n0_12", "n0_21", "d0_12", "d0_21",
    "n_scored_contact12", "n_scored_contact21",
    "pae_cutoff12", "pae_cutoff21", "native_iptm12", "native_iptm21",
]


def molecule_label(tokens, chain, mol_type):
    """Compact modeled-molecule description for human-readable TSVs.

    Polymer lengths are counts of *distinct residues in the coordinate model*,
    not raw PAE tokens (modified residues may each contain many PAE tokens),
    nor the full UniProt/sequence length of a truncated construct.
    """
    chain_tokens = tokens.loc[tokens.chain == chain]
    # For polymers, unit_key identifies a residue even when its atoms are
    # tokenized separately. For ligands, we preserve the chemical component ID.
    if mol_type in ("prot", "nuc"):
        residues = chain_tokens.drop_duplicates("unit_key")
        length = len(residues)
        if mol_type == "prot":
            return f"AA_{length}"
        names = set(residues.resname.astype(str).str.upper())
        dna = bool(names & {"DA", "DC", "DG", "DT", "DI", "DU"})
        rna = bool(names & {"A", "C", "G", "U", "I"})
        # If canonical bases of both types occur, use the compact NUC label. Unknown
        # modified nucleotide residue names inherit the recognized chain type.
        # If none are recognizable, avoid guessing DNA versus RNA.
        kind = "NUC" if dna and rna else "DNA" if dna else "RNA" if rna else "NA"
        return f"{kind}_{length}"
    names = chain_tokens.resname.astype(str).unique().tolist()
    return "+".join(names) if names else "-"



def protein_d0(n: int, minimum: float = 1.0) -> float:
    if n < 1:
        return float(minimum)
    return max(float(minimum), float(1.24 * np.cbrt(float(n) - 15.) - 1.8))


def scored_unit_key(row) -> str:
    """Weight modified polymer residues once even if represented as atoms."""
    if row["type"] in ("prot", "nuc"):
        return f'{row["chain"]}:res:{row["resnum"]}:{row["resname"]}'
    return f'{row["chain"]}:token:{int(row["token_id"])}'


@dataclass
class DirectionResult:
    score: Optional[float]
    ipae: Optional[float]
    n0: Optional[int]
    d0: Optional[float]
    ncontact: Optional[int]
    best_token: Optional[str]


def score_direction(pae, tokens, aligned_idx, scored_idx, scored_type,
                    scored_contact_units, pae_cutoff=10.0, fixed_d0=2.0,
                    protein_min_d0=1.0):
    """Max over valid reference frames, mean over scored units for each frame."""
    possible = [int(i) for i in aligned_idx if bool(tokens.iloc[i]["frame_valid"])]
    if not possible:
        return DirectionResult(None, None, None, None, None, None)
    # Fast path: the common case (standard protein/nucleotide tokens and
    # atom-tokenized ligands). Scores and PAEs are vectorized over all eligible
    # reference frames. No atom distances or full PAE-pair DataFrame are built.
    unit_keys = [str(tokens.iloc[int(j)]["unit_key"]) for j in scored_idx]
    if len(set(unit_keys)) == len(unit_keys):
        sub = np.asarray(pae[np.ix_(possible, scored_idx)], dtype=float)
        if scored_type in ("prot", "nuc"):
            mask = np.isfinite(sub) & (sub <= pae_cutoff)
        else:
            if not np.isfinite(sub).all():
                raise ValueError("Nonfinite ligand/ion PAE: cannot score the whole molecule")
            mask = np.ones_like(sub, dtype=bool)
        counts = mask.sum(axis=1).astype(int)
        if scored_type == "prot":
            d0s = np.maximum(protein_min_d0, 1.24 * np.cbrt(np.maximum(counts, 1) - 15.) - 1.8)
        else:
            d0s = np.full(len(possible), float(fixed_d0), dtype=float)
        tm = 1. / (1. + (sub / d0s[:, None]) ** 2.)
        with np.errstate(invalid="ignore", divide="ignore"):
            summed = np.where(mask, tm, 0.).sum(axis=1)
            means = np.divide(summed, counts, out=np.zeros(len(counts), float), where=counts > 0)
        best_index = int(np.argmax(means))
        good = mask[best_index]
        n0 = int(counts[best_index])
        win = np.asarray(sub[best_index, good], dtype=float)
        ipae = float(np.mean(win)) if n0 else None
        ncontacts = sum(unit_keys[j] in scored_contact_units for j in np.flatnonzero(good))
        tok = tokens.iloc[possible[best_index]]
        label = f'{tok["resname"]}:{tok["resnum"]}:{tok["atom_name"]}'
        return DirectionResult(float(means[best_index]), ipae, n0,
                               float(d0s[best_index]), int(ncontacts), label)
    groups = OrderedDict()
    for j in scored_idx:
        key = str(tokens.iloc[int(j)]["unit_key"])
        groups.setdefault(key, []).append(int(j))
    best = None
    use_cutoff = scored_type in ("prot", "nuc")
    for i in possible:
        all_values = []
        all_scores = []
        selected_keys = []
        # For modified residues: equal-weight residue contributions; average
        # PAE / TM kernel only over the atom tokens passing the cutoff.
        selected = []
        for key, idx in groups.items():
            vals = np.asarray(pae[i, idx], dtype=float)
            if use_cutoff:
                vals = vals[np.isfinite(vals) & (vals <= pae_cutoff)]
            elif not np.isfinite(vals).all():
                raise ValueError("Nonfinite ligand/ion PAE: cannot score the whole molecule")
            if len(vals):
                selected.append(vals)
                selected_keys.append(key)
        n0 = len(selected)
        if scored_type == "prot":
            d0 = protein_d0(n0, protein_min_d0)
        else:
            d0 = float(fixed_d0)
        for vals in selected:
            all_scores.append(float(np.mean(1. / (1. + (vals / d0) ** 2.))))
            all_values.append(float(np.mean(vals)))
        value = float(np.mean(all_scores)) if n0 else 0.0
        ipae = float(np.mean(all_values)) if n0 else None
        contact_count = len(set(selected_keys) & scored_contact_units)
        tok = tokens.iloc[i]
        label = f'{tok["resname"]}:{tok["resnum"]}:{tok["atom_name"]}'
        rec = DirectionResult(value, ipae, n0, d0, contact_count, label)
        if best is None or rec.score > best.score:
            best = rec
    return best


def native_pair(native_df, chain1, chain2):
    """Read predictor's scores; never invent directional scores from symmetric ones."""
    if native_df is None or native_df.empty:
        return None, None, None
    matching = native_df[
        ((native_df.chain_i == chain1) & (native_df.chain_j == chain2)) |
        ((native_df.chain_i == chain2) & (native_df.chain_j == chain1))
    ]
    if matching.empty:
        return None, None, None
    vals = matching.native_iptm.to_numpy(dtype=float)
    val = float(max(vals)) if len(vals) else None
    if not bool(matching.directional.any()):
        return val, None, None
    left = matching[(matching.chain_i == chain1) & (matching.chain_j == chain2)]
    right = matching[(matching.chain_i == chain2) & (matching.chain_j == chain1)]
    v12 = float(left.iloc[0].native_iptm) if len(left) else None
    v21 = float(right.iloc[0].native_iptm) if len(right) else None
    return val, v12, v21


def calculate_scores(pae, tokens, geometries, native_df=None, pae_cutoff=10.0,
                     contact_cutoff=5.0, fixed_d0=2.0, protein_min_d0=1.0,
                     all_pairs=False):
    if len(tokens) != len(pae):
        raise ValueError("PAE/token count mismatch")
    first = tokens.drop_duplicates("chain", keep="first")
    molecule_type = dict(zip(first.chain, first.type))
    molecule_order = list(first.chain)
    molecules = sorted(molecule_order, key=lambda c: TYPE_ORDER[molecule_type[c]])
    token_chains = tokens["chain"].to_numpy()
    token_idxs = {c: np.flatnonzero(token_chains == c) for c in molecules}
    rows = []
    for ai, chain1 in enumerate(molecules):
        for chain2 in molecules[ai + 1:]:
            contact1, contact2 = geometries[chain1], geometries[chain2]
            from .contacts import contact_units
            c1, c2 = contact_units(contact1, contact2, contact_cutoff)
            if not (c1 and c2) and not all_pairs:
                continue
            type1, type2 = molecule_type[chain1], molecule_type[chain2]
            if type1 == "ion" and type2 == "ion":
                continue
            a = score_direction(pae, tokens, token_idxs[chain1], token_idxs[chain2],
                                type2, c2, pae_cutoff, fixed_d0, protein_min_d0)
            b = score_direction(pae, tokens, token_idxs[chain2], token_idxs[chain1],
                                type1, c1, pae_cutoff, fixed_d0, protein_min_d0)
            # Same-type pairs use the maximum; mixed pairs use the higher-ranked
            # molecule as the aligned reference (the sorted 12 direction).
            if type1 == type2:
                if b.score is not None and (a.score is None or b.score > a.score):
                    score, source = b.score, "ipsae21"
                else:
                    score, source = a.score, "ipsae12" if a.score is not None else "NA"
            else:
                score, source = a.score, "ipsae12" if a.score is not None else "NA"
            native, native12, native21 = native_pair(native_df, chain1, chain2)
            cutoff12 = (pae_cutoff if type2 in ("prot", "nuc") else "none") if a.score is not None else "NA"
            cutoff21 = (pae_cutoff if type1 in ("prot", "nuc") else "none") if b.score is not None else "NA"
            rows.append(dict(
                chn1=chain1, chn2=chain2, type1=type1, type2=type2,
                mol1=molecule_label(tokens, chain1, type1),
                mol2=molecule_label(tokens, chain2, type2),
                iptm=native, ipsae=score, native_iptm12=native12,
                native_iptm21=native21, ipsae12=a.score, ipsae21=b.score,
                ipsae_source=source, iPAE12=a.ipae, iPAE21=b.ipae,
                n0_12=a.n0, n0_21=b.n0, d0_12=a.d0, d0_21=b.d0,
                n_scored_contact12=a.ncontact,
                n_scored_contact21=b.ncontact,
                pae_cutoff12=cutoff12, pae_cutoff21=cutoff21,
                align_token12=a.best_token, align_token21=b.best_token,
            ))
    result = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    for column in ("n0_12", "n0_21", "n_scored_contact12", "n_scored_contact21"):
        result[column] = pd.array(result[column], dtype="Int64")
    return result
