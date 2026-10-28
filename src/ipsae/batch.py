"""Conservative discovery of matching predictor files for batch scoring.

A batch run never combines coordinate and PAE files across directories.
Models with ambiguous matching files are skipped with an explicit diagnostic
instead of silently producing scientifically invalid scores.
"""
import os
from pathlib import Path

COORD_EXTENSIONS = {'.pdb', '.cif', '.mmcif'}
DATA_EXTENSIONS = {'.json', '.npz'}
SKIP_DIRS = {'.git', '.venv', '__pycache__', '.pytest_cache', 'node_modules',
             'build', 'dist', '.mypy_cache', '.tox'}


def model_key(path):
    """Strip file-role prefixes/suffixes while retaining seed/model numbers."""
    s = Path(path).stem.lower()
    prefixes = ('summary_confidences_', 'predicted_aligned_error_',
                'confidences_', 'confidence_', 'full_data_', 'pae_')
    suffixes = ('_summary_confidences', '_predicted_aligned_error',
                '_confidences', '_confidence', '_full_data', '_pae')
    changed = True
    while changed:
        changed = False
        for pre in prefixes:
            if s.startswith(pre):
                s = s[len(pre):]
                changed = True
                break
        for suf in suffixes:
            if s.endswith(suf):
                s = s[:-len(suf)]
                changed = True
                break
    # Common AF3 outputs: id_model.cif + id_confidences.json
    if s.endswith('_model'):
        s = s[:-6]
    return s


def _file_role(path):
    """Classify common predictor filenames, inspect contents only if unknown."""
    name = path.stem.lower()
    if 'summary_confidences' in name or name.startswith(('confidence_', 'iptm_')) or name.endswith(('_confidence', '_iptm')):
        return 'confidence'
    if ('pae' in name or 'full_data' in name or 'confidences' in name
            or 'predicted_aligned_error' in name):
        return 'pae'
    # Ambiguous filenames: inspect JSON/NPZ keys. Parse only when needed.
    from .native_iptm import load_json_or_npz, looks_like_pae_obj, looks_like_confidence_obj
    try:
        contents = load_json_or_npz(str(path))
    except (OSError, ValueError):
        return 'unknown'
    if looks_like_pae_obj(contents):
        return 'pae'
    if looks_like_confidence_obj(contents) or 'iptm' in contents:
        return 'confidence'
    return 'unknown'


def _select(coord, candidates, n_coords):
    """Match by model key; use per-directory singleton fallback only."""
    exact = [p for p in candidates if model_key(p) == model_key(coord)]
    if len(exact) == 1:
        return exact[0], None
    if len(exact) > 1:
        return None, f'ambiguous matches for {coord.name}: {", ".join(p.name for p in exact)}'
    if n_coords == 1 and len(candidates) == 1:
        return candidates[0], None
    return None, f'no unique match for {coord.name}'


def discover_models(root, recursive=True):
    """Return (coordinate, PAE, optional confidence) tuples and warnings."""
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError(f'Not a directory: {root}')
    jobs, warnings = [], []
    for folder, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS and not d.startswith('.')) if recursive else []
        directory = Path(folder)
        coords = sorted((directory / name for name in files
                         if Path(name).suffix.lower() in COORD_EXTENSIONS and not name.startswith('.')))
        if not coords:
            continue
        data_files = sorted(directory / name for name in files
                            if Path(name).suffix.lower() in DATA_EXTENSIONS and not name.startswith('.'))
        roles = {p: _file_role(p) for p in data_files}
        pae_candidates = [p for p in data_files if roles[p] == 'pae']
        confidence_candidates = [p for p in data_files if roles[p] == 'confidence']
        for coord in coords:
            pae, problem = _select(coord, pae_candidates, len(coords))
            if problem:
                warnings.append(f'{coord.relative_to(root)}: {problem}')
                continue
            # If an exact confidence cannot be distinguished, score without it
            # rather than attach values for the wrong model.
            conf, problem = _select(coord, confidence_candidates, len(coords))
            if problem and 'ambiguous' in problem:
                warnings.append(f'{coord.relative_to(root)}: {problem}; omitting native ipTM')
            if problem:
                conf = None
            jobs.append((coord, pae, conf))
    return jobs, warnings
