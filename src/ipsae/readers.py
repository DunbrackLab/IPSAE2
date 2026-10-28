"""Read coordinates/PAE, identify molecules and reconstruct predictor token order.

The tokenization policies originate from build_pae_data11.py, with additional
checks against source token-chain metadata and ambiguous policies.
"""
from dataclasses import dataclass
from collections import OrderedDict
import json
from pathlib import Path
import numpy as np
import pandas as pd
import gemmi
from .native_iptm import identify_input_files, read_native_confidences
from .scores import scored_unit_key

PROT = set("ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL SEC PYL".split())
NUC = set("A C G U I DA DC DG DT DI DU".split())
IONS = set("LI NA K RB CS MG CA SR BA MN FE FE2 FE3 CO NI CU CU1 CU2 ZN CD AL GA HG PB F CL BR IOD I".split())
WATER = {"HOH", "WAT", "DOD"}


def read_pae(path, key="pae"):
    suffix = Path(path).suffix.lower()
    metadata = {}
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as f:
            keys = list(f.files)
            match = next((k for k in [key, "token_pair_pae", "predicted_aligned_error"] if k in f), None)
            if match is None:
                raise ValueError(f"No PAE array found in {path}. Keys: {keys}")
            pae = np.asarray(f[match], dtype=float)
            for k in ("frame_mask", "asym_id", "token_asym_id", "plddt", "token_has_frame"):
                if k in f:
                    metadata[k] = np.asarray(f[k])
    elif suffix == ".json":
        with open(path, encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, list):
            if len(obj) != 1 or not isinstance(obj[0], dict):
                raise ValueError("PAE JSON list must contain exactly one data object")
            obj = obj[0]
        match = next((k for k in [key, "token_pair_pae", "predicted_aligned_error"] if k in obj), None)
        if match is None:
            raise ValueError(f"No PAE array found in {path}. Keys: {list(obj)}")
        pae = np.asarray(obj[match], dtype=float)
        for k in ("token_chain_ids", "token_res_ids", "token_has_frame", "token_asym_id", "atom_to_token_idx", "frame_mask", "asym_id", "plddt"):
            if k in obj:
                metadata[k] = np.asarray(obj[k])
    else:
        raise ValueError(f"PAE file must be .json or .npz: {path}")
    if pae.ndim != 2 or pae.shape[0] != pae.shape[1] or pae.size == 0:
        raise ValueError(f"PAE must be a nonempty square matrix; got {pae.shape}")
    return pae, metadata


def _clean(x):
    """Decode mmCIF values, including atom names like "C1'".

    Gemmi's Table rows expose *raw CIF tokens*, so quoted atom names retain
    their enclosing quotes until passed through gemmi.cif.as_string().
    Without this, every fallback nucleotide frame fails the C1'/C3'/C4'
    lookup and nucleotide-aligned ipSAE is incorrectly returned as NA.
    """
    if x is None:
        return ""
    return gemmi.cif.as_string(str(x))


def read_atoms(path):
    if Path(path).suffix.lower() == ".pdb":
        # Preserve PDB chain IDs, atom ordering, and residue numbers.
        rows = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                rec = line[:6].strip()
                if rec == "ENDMDL":
                    break
                if rec not in ("ATOM", "HETATM") or line[16:17].strip() not in ("", "A", "1"):
                    continue
                name = line[17:20].strip()
                if name in WATER:
                    continue
                rows.append(dict(
                    record=rec, chain=line[21:22].strip() or "_",
                    resnum=(line[22:27].strip() or "1"), resname=name,
                    atom_name=line[12:16].strip(),
                    element=(line[76:78].strip() if len(line) >= 78 else ""),
                    x=float(line[30:38]), y=float(line[38:46]), z=float(line[46:54]),
                ))
    else:
        block = gemmi.cif.read_file(str(path)).sole_block()
        required = ["group_PDB", "label_atom_id", "label_comp_id", "label_asym_id",
                    "Cartn_x", "Cartn_y", "Cartn_z"]
        required_tags = ["_atom_site." + t for t in required]
        table = block.find(required_tags)
        if not len(table):
            raise ValueError("mmCIF missing required _atom_site fields")
        # Optional fields are requested separately to handle CIFs lacking them.
        optional = {}
        for name in ("label_seq_id", "auth_seq_id", "auth_asym_id", "type_symbol", "label_alt_id"):
            col = block.find_values("_atom_site." + name)
            if col and len(col) == len(table):
                optional[name] = list(map(str, col))
        rows = []
        for i, r in enumerate(table):
            record, atom, name, chain, x, y, z = map(str, r)
            if name in WATER:
                continue
            alt = _clean(optional.get("label_alt_id", [""] * len(table))[i])
            if alt not in ("", "A", "1"):
                continue
            seq = _clean(optional.get("label_seq_id", [""] * len(table))[i])
            if not seq:
                seq = _clean(optional.get("auth_seq_id", [""] * len(table))[i])
            chain = _clean(chain) or _clean(optional.get("auth_asym_id", [""] * len(table))[i]) or "_"
            rows.append(dict(record=record, chain=chain,
                             resnum=seq or "1", resname=_clean(name),
                             atom_name=_clean(atom),
                             element=_clean(optional.get("type_symbol", [""] * len(table))[i]),
                             x=float(x), y=float(y), z=float(z)))
    if not rows:
        raise ValueError(f"No coordinate atoms in {path}")
    atoms = pd.DataFrame(rows)
    # Infer absent PDB element fields, particularly for hydrogen removal.
    blank = atoms.element.astype(str).str.strip() == ""
    if blank.any():
        def infer_element(row):
            name = str(row.atom_name).upper().lstrip("0123456789")
            if name.startswith(("H", "D")):
                return "H"
            if str(row.resname).upper() in IONS:
                return str(row.resname).upper()
            return name[:1]
        atoms.loc[blank,"element"] = atoms.loc[blank].apply(infer_element,axis=1)
    # Atom-tokenized predictors exclude explicit hydrogens.
    atoms = atoms.loc[~atoms.element.astype(str).str.upper().isin(("H", "D"))].reset_index(drop=True)
    # Stable ordered residue identities; PDB insertion codes are retained.
    atoms["residue_key"] = list(zip(atoms.chain, atoms.resnum, atoms.resname))
    return atoms


def classify(atoms):
    """Classify distinct molecules; prevent mixed polymer/nonpolymer chain ambiguity."""
    atoms = atoms.copy()
    residues = atoms.drop_duplicates("residue_key").copy()
    chain_types = {}
    for chain, group in residues.groupby("chain", sort=False):
        names = group.resname.tolist()
        has_p = any(n in PROT for n in names)
        has_n = any(n in NUC for n in names)
        if has_p and not has_n:
            ctype = "prot"
        elif has_n and not has_p:
            ctype = "nuc"
        elif has_p and has_n:
            raise ValueError(f"Chain {chain} mixes protein and nucleic acid residues")
        else:
            if len(group) > 1:
                raise ValueError(f"Chain {chain} has multiple nonpolymer residues. "
                                 "Split the molecules into unique chain IDs (one per ligand/ion).")
            g = atoms[atoms.chain == chain]
            element = str(g.iloc[0].element).upper()
            ctype = "ion" if len(g) == 1 and (names[0].upper() in IONS or element in IONS) else "lig"
        chain_types[chain] = ctype
    atoms["type"] = atoms.chain.map(chain_types)
    # A nonpolymer inside a polymer chain is ambiguous: rather than quietly
    # folding it into that protein, stop and ask for unambiguous molecule IDs.
    for chain, group in residues.groupby("chain", sort=False):
        ctype = chain_types[chain]
        if ctype == "prot":
            for r in group.itertuples():
                g = atoms[atoms.residue_key == r.residue_key]
                if (r.resname not in PROT and g.record.eq("HETATM").all()
                        and len(g) == 1 and (str(g.iloc[0].element).upper() in IONS)):
                    raise ValueError(f"Ion {r.resname} shares protein chain {chain}; "
                                     "use unique per-molecule chain IDs")
    return atoms


def _token_candidates(atoms, mod_mode, ligand_mode):
    residue_groups = OrderedDict((k, g) for k, g in atoms.groupby("residue_key", sort=False))
    records = []
    atom_to_token = {}
    chain_lig_token = {}
    for (chain, resnum, resname), group in residue_groups.items():
        ctype = str(group.iloc[0].type)
        modified = ctype in ("prot", "nuc") and resname not in (PROT if ctype == "prot" else NUC)
        atom_tokens = (ctype in ("lig", "ion") and ligand_mode == "atom") or (modified and mod_mode == "atom")
        if ctype in ("lig", "ion") and ligand_mode == "chain":
            # Chains are guaranteed to represent individual molecules.
            atom_tokens = False
        if atom_tokens:
            chunks = [[idx] for idx in group.index]
        else:
            chunks = [group.index.tolist()]
        for chunk in chunks:
            sub = atoms.loc[chunk]
            if len(chunk) == 1:
                rep = sub.iloc[0]
            else:
                choices = ["CA", "C", "N"] if ctype == "prot" else (["C3'", "C3*", "P"] if ctype == "nuc" else [])
                rep = next((sub[sub.atom_name == name].iloc[0] for name in choices if not sub[sub.atom_name == name].empty), sub.iloc[0])
            records.append(dict(token_id=len(records)+1, chain=chain,
                                resnum=resnum, resname=resname, type=ctype,
                                tokenization="atom" if len(chunk) == 1 and (atom_tokens or ctype in ("lig", "ion")) else "residue",
                                atom_name=rep.atom_name, element=rep.element,
                                x=rep.x, y=rep.y, z=rep.z))
            for idx in chunk:
                atom_to_token[idx] = len(records)
    tokens = pd.DataFrame(records)
    return tokens, atom_to_token


def _metadata_chains(metadata, n):
    if "token_chain_ids" in metadata:
        v = np.asarray(metadata["token_chain_ids"])
        if v.ndim == 1 and len(v) == n:
            return [str(x) for x in v], "names"
    for key in ("token_asym_id", "asym_id"):
        if key in metadata:
            v = np.asarray(metadata[key])
            if v.ndim == 1 and len(v) == n:
                return [str(x) for x in v], "index"
    return None, None


def infer_tokens(atoms, pae, metadata):
    n = len(pae)
    chain_info, mode = _metadata_chains(metadata, n)
    matches = []
    descriptions = []
    residue_first = atoms.drop_duplicates("residue_key")
    res_sizes = atoms.groupby("residue_key", sort=False).size().to_dict()
    n_canonical, mod_count, mod_atoms = 0, 0, 0
    lig_count, lig_atoms = 0, 0
    for row in residue_first.itertuples():
        size = int(res_sizes[row.residue_key])
        if row.type in ("prot", "nuc"):
            if row.resname not in (PROT if row.type == "prot" else NUC):
                mod_count += 1
                mod_atoms += size
            else:
                n_canonical += 1
        else:
            lig_count += 1
            lig_atoms += size
    built = {}
    for m in ("atom", "residue"):
        for l in ("atom", "residue", "chain"):
            estimated = (n_canonical + (mod_atoms if m=="atom" else mod_count)
                         + (lig_atoms if l=="atom" else lig_count))
            descriptions.append(f"modified={m},ligand={l}: {estimated} tokens")
            if estimated != n:
                continue
            # Policies for absent PTMs or ligands are mathematically identical.
            effective = (m if mod_count else "none", l if lig_count else "none")
            if effective not in built:
                built[effective] = _token_candidates(atoms, m, l)
            t, mapping = built[effective]
            chain_seq = t.chain.astype(str).tolist()
            if chain_info is not None:
                if mode == "names":
                    if chain_seq != chain_info:
                        continue
                else:
                    seen = list(dict.fromkeys(chain_info))
                    chains = list(dict.fromkeys(chain_seq))
                    expected = {key: chains[i] for i, key in enumerate(seen)} if len(seen) == len(chains) else {}
                    if [expected.get(c) for c in chain_info] != chain_seq:
                        continue
            matches.append((m, l, t, mapping))
    if not matches:
        raise ValueError("No token mapping matches PAE dimensions/chain metadata: " + "; ".join(descriptions))
    # Distinct ordered atom/residue tokens with the same total n can happen in
    # mixed PTM+ligand systems. Unless source information disambiguates, FAIL.
    signatures = {}
    for m,l,t,mapping in matches:
        sig = tuple(zip(t.chain, t.resnum, t.resname, t.atom_name))
        signatures.setdefault(sig, []).append((m,l,t,mapping))
    if len(signatures) != 1:
        raise ValueError(f"Ambiguous token order: {len(signatures)} count-compatible "
                         "mappings. Provide source token metadata or a supported predictor format.")
    m,l,t,mapping = matches[0]
    atoms = atoms.copy()
    atoms["token_id"] = atoms.index.map(mapping).astype(int)
    if "atom_to_token_idx" in metadata:
        source_map = np.asarray(metadata["atom_to_token_idx"])
        if source_map.ndim == 1 and len(source_map) == len(atoms):
            try:
                source_map = source_map.astype(int)
                if source_map.min() >= 0 and source_map.max() < len(t):
                    if not np.array_equal(source_map, atoms.token_id.to_numpy(dtype=int)-1):
                        raise ValueError("Predictor atom_to_token_idx metadata disagrees with inferred PAE token ordering")
            except (TypeError, ValueError) as exc:
                if "disagrees" in str(exc):
                    raise
                # Some predictors encode atom mapping differently; chain IDs
                # and token counts remain the primary validation.
    t["unit_key"] = t.apply(scored_unit_key, axis=1)
    atoms["unit_key"] = atoms.token_id.map(dict(zip(t.token_id, t.unit_key)))
    return atoms, t, f"modified={m};ligand={l}"


def _frame_validity(atoms, tokens, metadata):
    """Use predictor mask if present; otherwise check three-atom geometry."""
    for key in ("token_has_frame", "frame_mask"):
        mask = np.asarray(metadata.get(key, []))
        if mask.ndim == 1 and len(mask) == len(tokens):
            valid = mask.astype(bool) & (tokens["type"].to_numpy() != "ion")
            return valid.tolist(), f"source:{key}"
    lookup = {key: g for key,g in atoms.groupby("residue_key", sort=False)}
    valid = []
    for tok in tokens.itertuples(index=False):
        if tok.type == "ion":
            valid.append(False)
            continue
        g = lookup[(tok.chain, tok.resnum, tok.resname)]
        xyz = g[["x", "y", "z"]].to_numpy(dtype=float)
        if tok.tokenization == "residue" and tok.type in ("prot", "nuc"):
            required = ["N", "CA", "C"] if tok.type == "prot" else ["C1'", "C3'", "C4'"]
            atom_positions = {r.atom_name: np.array([r.x, r.y, r.z]) for r in g.itertuples()}
            if not all(n in atom_positions for n in required):
                valid.append(False)
                continue
            p0, p1, p2 = (atom_positions[n] for n in required)
        else:
            origin = np.array([tok.x, tok.y, tok.z])
            others = xyz[np.linalg.norm(xyz - origin, axis=1) > 1e-5]
            if len(others) < 2:
                valid.append(False)
                continue
            idx = np.argsort(np.linalg.norm(others - origin, axis=1))[:2]
            p0, p1, p2 = origin, others[idx[0]], others[idx[1]]
        valid.append(bool(np.linalg.norm(np.cross(p1-p0, p2-p0)) > 1e-4))
    return valid, "inferred:geometry"


@dataclass
class ModelData:
    pae: np.ndarray
    atoms: pd.DataFrame
    tokens: pd.DataFrame
    native: pd.DataFrame | None
    token_policy: str
    frame_policy: str
    files: tuple[str, str, str | None]


def load_model(files, pae_key="pae", verbose=True):
    pae_file, coords_file, conf_file = identify_input_files(list(files), pae_key=pae_key, verbose=verbose)
    pae, metadata = read_pae(pae_file, pae_key)
    atoms = classify(read_atoms(coords_file))
    atoms, tokens, policy = infer_tokens(atoms, pae, metadata)
    mask, frame_policy = _frame_validity(atoms, tokens, metadata)
    tokens["frame_valid"] = mask
    # Reuse native confidences in the PAE file when no distinct third file was passed.
    native_path = conf_file or pae_file
    _, native = read_native_confidences(native_path, list(dict.fromkeys(tokens.chain)))
    return ModelData(pae, atoms, tokens, native, policy, frame_policy,
                     (pae_file, coords_file, conf_file))
