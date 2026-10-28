"""Fast heavy-atom molecular contact detection, independent of the PAE score."""
from dataclasses import dataclass
import numpy as np
from scipy.spatial import cKDTree


@dataclass
class MoleculeGeometry:
    xyz: np.ndarray
    unit_ids: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    tree: cKDTree


def prepare_geometry(atom_df):
    geometries = {}
    for chain, atoms in atom_df.groupby("chain", sort=False):
        elements = atoms["element"].astype(str).str.upper().str.strip()
        heavy = (~elements.isin(("H", "D"))) & np.isfinite(atoms[["x", "y", "z"]]).all(axis=1)
        good = atoms.loc[heavy]
        xyz = good[["x", "y", "z"]].to_numpy(dtype=float)
        if not len(xyz):
            raise ValueError(f"Molecule {chain} has no finite heavy-atom coordinates")
        geometries[str(chain)] = MoleculeGeometry(
            xyz=xyz,
            unit_ids=good["unit_key"].to_numpy(dtype=object),
            lo=xyz.min(axis=0), hi=xyz.max(axis=0),
            tree=cKDTree(xyz),
        )
    return geometries


def contact_units(a: MoleculeGeometry, b: MoleculeGeometry, cutoff: float = 5.0):
    """Return (units in A touching B, units in B touching A), including <= cutoff.

    Fast exact bbox rejection followed by nearest-neighbor query, avoiding a
    Cartesian atom-atom distance array or an explicit contact-pair list.
    """
    if np.any(a.hi + cutoff < b.lo) or np.any(b.hi + cutoff < a.lo):
        return set(), set()
    # nextafter preserves the exact inclusive condition at the boundary.
    r = float(np.nextafter(float(cutoff), np.inf))
    da, _ = b.tree.query(a.xyz, distance_upper_bound=r, workers=1)
    if not np.isfinite(da).any():
        return set(), set()
    db, _ = a.tree.query(b.xyz, distance_upper_bound=r, workers=1)
    return set(a.unit_ids[np.isfinite(da)].tolist()), set(b.unit_ids[np.isfinite(db)].tolist())
