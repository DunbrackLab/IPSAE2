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
                    protein_min_d0=1.0, include_chain=False):
    """Max over valid reference frames, mean over scored units for each frame."""
    possible = [int(i) for i in aligned_idx if bool(tokens.iloc[i]["frame_valid"])]
    if not possible:
        return DirectionResult(None, None, None, None, None, None)
    # Fast path: the common case (standard protein/nucleotide tokens and
    # atom-tokenized ligands). Scores and PAEs are vectorized over all eligible
    # reference frames. No atom distances or full PAE-pair DataFrame are built.
    unit_keys = [str(tokens.iloc[int(j)]["unit_key"]) for j in scored_idx]
    # Virtual groups may contain more than one molecular type. A PAE cutoff
    # applies to *all* scored units when any protein/nucleic acid is present,
    # whereas the length used for protein d0 counts only qualifying protein
    # residues (not ligand atoms or nucleotides).
    scored_types = tokens.iloc[np.asarray(scored_idx, dtype=int)]["type"].to_numpy(dtype=str)
    use_cutoff = bool(np.isin(scored_types, ("prot", "nuc")).any())
    has_protein = bool(np.any(scored_types == "prot"))
    if len(set(unit_keys)) == len(unit_keys):
        sub = np.asarray(pae[np.ix_(possible, scored_idx)], dtype=float)
        if use_cutoff:
            mask = np.isfinite(sub) & (sub <= pae_cutoff)
        else:
            if not np.isfinite(sub).all():
                raise ValueError("Nonfinite ligand/ion PAE: cannot score the whole molecule")
            mask = np.ones_like(sub, dtype=bool)
        counts = mask.sum(axis=1).astype(int)
        if has_protein:
            protein_counts = mask[:, scored_types == "prot"].sum(axis=1).astype(int)
            d0s = np.maximum(protein_min_d0, 1.24 * np.cbrt(np.maximum(protein_counts, 1) - 15.) - 1.8)
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
        if include_chain:
            label = f'{tok["chain"]}/{label}'
        return DirectionResult(float(means[best_index]), ipae, n0,
                               float(d0s[best_index]), int(ncontacts), label)
    groups = OrderedDict()
    for j in scored_idx:
        key = str(tokens.iloc[int(j)]["unit_key"])
        groups.setdefault(key, []).append(int(j))
    best = None
    group_types = {key: str(tokens.iloc[idx[0]]["type"]) for key, idx in groups.items()}
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
        if has_protein:
            n_protein = sum(group_types[k] == "prot" for k in selected_keys)
            d0 = protein_d0(n_protein, protein_min_d0)
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
        if include_chain:
            label = f'{tok["chain"]}/{label}'
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


def resolve_groups(tokens, groups=()):
    """Validate disjoint virtual molecules and return entity-name -> member IDs.

    The returned mapping puts requested groups first, followed by all ungrouped
    chains in their coordinate-file order. Group labels never replace or alter
    the underlying chains, residue identifiers, or token order.
    """
    chains = list(dict.fromkeys(tokens.chain.astype(str)))
    known = set(chains)
    used = set()
    entities = OrderedDict()
    for spec in groups or ():
        members = str(spec).split("+")
        if len(members) < 2 or any(not m or m != m.strip() for m in members):
            raise ValueError(f"Invalid --group {spec!r}; use two or more chain IDs joined by '+', e.g. H+L")
        if len(set(members)) != len(members):
            raise ValueError(f"Invalid --group {spec!r}: a chain cannot appear twice in one group")
        unknown = sorted(set(members) - known)
        if unknown:
            raise ValueError(f"Unknown chain(s) {', '.join(unknown)} in --group {spec!r}. "
                             f"Available chain IDs: {', '.join(chains)}. "
                             "mmCIF uses label_asym_id; PDB uses author chain IDs.")
        overlap = set(members) & used
        if overlap:
            raise ValueError(f"Chain(s) {', '.join(sorted(overlap))} occur in multiple --group options")
        entities[str(spec)] = tuple(members)
        used.update(members)
    for chain in chains:
        if chain not in used:
            entities[chain] = (chain,)
    return entities


def entity_contacts(geometries, left, right, cutoff):
    """Contact sets for disjoint virtual entities, preserving original unit keys."""
    from .contacts import contact_units
    a, b = set(), set()
    for chain1 in left:
        for chain2 in right:
            ca, cb = contact_units(geometries[chain1], geometries[chain2], cutoff)
            a.update(ca)
            b.update(cb)
    return a, b


def _group_scores(pae, tokens, geometries, groups, pae_cutoff,
                  contact_cutoff, fixed_d0, protein_min_d0, all_pairs):
    entities = resolve_groups(tokens, groups)
    # Only append pairs involving a virtual entity; original pairs are handled
    # separately, unchanged, in calculate_scores.
    names = list(entities)
    n_groups = len(groups)
    chain_types = dict(zip(tokens.chain.astype(str), tokens.type.astype(str)))
    original = tokens.chain.astype(str).to_numpy()
    idx = {name: np.flatnonzero(np.isin(original, members))
           for name, members in entities.items()}
    labels = {name: '+'.join(chain_types[c] for c in members)
              for name, members in entities.items()}
    descriptions = {name: '+'.join(molecule_label(tokens, c, chain_types[c]) for c in members)
                    for name, members in entities.items()}
    rows = []
    for ai, name1 in enumerate(names):
        for name2 in names[ai+1:]:
            if ai >= n_groups:  # Both are original single-chain entities.
                continue
            m1, m2 = entities[name1], entities[name2]
            c1, c2 = entity_contacts(geometries, m1, m2, contact_cutoff)
            if not (c1 and c2) and not all_pairs:
                continue
            # The no-ion/ion rule for original pairs is unchanged; composite
            # entities are intentionally allowed to include any molecule type.
            a = score_direction(pae, tokens, idx[name1], idx[name2],
                                labels[name2], c2, pae_cutoff, fixed_d0, protein_min_d0,
                                include_chain=len(m1) > 1)
            b = score_direction(pae, tokens, idx[name2], idx[name1],
                                labels[name1], c1, pae_cutoff, fixed_d0, protein_min_d0,
                                include_chain=len(m2) > 1)
            if b.score is not None and (a.score is None or b.score > a.score):
                value, source = b.score, 'ipsae21'
            else:
                value, source = a.score, 'ipsae12' if a.score is not None else 'NA'
            t1 = set(chain_types[c] for c in m1)
            t2 = set(chain_types[c] for c in m2)
            rows.append(dict(
                chn1=name1, chn2=name2, type1=labels[name1], type2=labels[name2],
                mol1=descriptions[name1], mol2=descriptions[name2],
                iptm=None, ipsae=value, native_iptm12=None, native_iptm21=None,
                ipsae12=a.score, ipsae21=b.score, ipsae_source=source,
                iPAE12=a.ipae, iPAE21=b.ipae, n0_12=a.n0, n0_21=b.n0,
                d0_12=a.d0, d0_21=b.d0,
                n_scored_contact12=a.ncontact, n_scored_contact21=b.ncontact,
                pae_cutoff12=(pae_cutoff if t2 & {'prot', 'nuc'} else 'none') if a.score is not None else 'NA',
                pae_cutoff21=(pae_cutoff if t1 & {'prot', 'nuc'} else 'none') if b.score is not None else 'NA',
                align_token12=a.best_token, align_token21=b.best_token,
            ))
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def calculate_scores(pae, tokens, geometries, native_df=None, pae_cutoff=10.0,
                     contact_cutoff=5.0, fixed_d0=2.0, protein_min_d0=1.0,
                     all_pairs=False, groups=()):
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
    if groups:
        grouped = _group_scores(pae, tokens, geometries, groups, pae_cutoff,
                                contact_cutoff, fixed_d0, protein_min_d0, all_pairs)
        rows.extend(grouped.to_dict('records'))
    result = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    for column in ("n0_12", "n0_21", "n_scored_contact12", "n_scored_contact21"):
        result[column] = pd.array(result[column], dtype="Int64")
    return result
