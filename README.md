# ipSAE2

**Directional PAE-based confidence scoring for biomolecular complexes**  
**Alpha release · v0.1.10**

ipSAE2 computes directional interface predicted aligned error (ipSAE) scores
from predicted coordinates and a PAE matrix. It supports protein, DNA, RNA,
ligand and ion interactions. It can also import native pairwise ipTM confidence
scores when the predictor provides them.

The preprint is now available: https://www.biorxiv.org/content/10.64898/2026.11.02.763449v1

This is a **new program and command-line interface**. The original
[ipSAE implementation](https://github.com/DunbrackLab/IPSAE) remains available
separately. The new release is suitable for testing and research evaluation;
it is not yet validated across every prediction format.

## Install (Linux or macOS)

Python **3.10 or newer** is required. You do **not** need AlphaFold, Boltz,
PyTorch, a GPU or Conda. Use a separate virtual environment to avoid changing
Gemmi or other libraries needed by your prediction programs.

```bash
git clone https://github.com/DunbrackLab/IPSAE2.git
cd IPSAE2
python3 -m venv .venv
source .venv/bin/activate
python -m pip install .
ipsae --version
```

Installed dependencies: NumPy, pandas, SciPy and Gemmi. For optional PAE
plots, install `python -m pip install '.[plot]'` from the repository.

## To activate ipSAE in a new terminal session, 

if path_to_IPSAE2 is the IPSAE2 directory above

```
source path_to_IPSAE2/.venv/bin/activate
```

## Run one model

Provide coordinates (`.pdb`, `.pdb.gz`, `.cif` or `.mmcif`) and PAE (`.json`, `.json.gz` or `.npz`),
with an optional native-confidence (`.json` or `.npz`) file **in any order**:

```bash
ipsae model.cif full_data.json summary_confidences.json
ipsae model.cif full_data.json
ipsae pae_model_0.npz model_0.cif confidence_model_0.json
# ColabFold 1.5: one JSON holds both PAE and native scalar ipTM
ipsae RAF1_KSR1_rank001_iptm0.50_ptm0.45_nrx_5_004.pdb.gz RAF1_KSR1_rank001_iptm0.50_ptm0.45_nrx_5_004.json
# Standard ColabFold 1.5: PDB.gz and scores JSON.gz, including job prefixes
ipsae job_unrelaxed_rank_001_alphafold2_multimer_v3_model_1_seed_000.pdb.gz job_scores_rank_001_alphafold2_multimer_v3_model_1_seed_000.json.gz
```

Options may appear **before, after, or between** input filenames:

```bash
ipsae model.cif full_data.json --output-file result.tsv
ipsae --output-file result.tsv model.cif full_data.json
ipsae model.cif --output-file result.tsv full_data.json
```

The default result is `model_ipsae.tsv` in the coordinate-file directory
(the `.pdb.gz` extension is removed as a unit). Missing native ipTM data is
reported as `NA`; it is not an error. A scalar overall ipTM is assigned to an
individual molecular pair only when the structure has exactly two chains.

## Score multiple chains as a single virtual molecule (`--group`)

Append grouped scores *in addition* to the normal chain-by-chain rows. Group
members are joined by `+`; repeat the flag to declare disjoint groups:

```bash
# Antibody heavy and light chains against antigen A in a PDB file:
ipsae predicted.pdb pae.json --group H+L

# The same chains in mmCIF using label_asym_id A (antigen), B (heavy), C (light):
ipsae predicted.cif pae.json --group B+C

# Two groups, plus any ungrouped components in the structure:
ipsae model.cif pae.json --group A+B --group C+D

# Protein–double-stranded DNA: A = protein; B and C = the two DNA strands:
ipsae predicted_protein_dna.cif pae.json --group B+C
```

For PDB, chain identifiers come from the one-character PDB chain field
(typically the author IDs). For mmCIF, ipSAE2 reads `_atom_site.label_asym_id`,
not `_atom_site.auth_asym_id`. For example, PDB entry 2J88 uses author IDs
`A/H/L` for antigen/heavy/light but label IDs `A/B/C`. The flags must refer
to the identifiers in the *coordinate file being scored*. A prediction's chain
IDs may differ from those in a reference crystal structure.

`--group H+L` adds the `H+L` versus other ungrouped molecules, but leaves the
original `H`/`A` and `L`/`A` scores untouched. With multiple groups, only
virtual-group-versus-virtual-group and virtual-group-versus-ungrouped rows are
added (not groups versus their own members). Groups cannot overlap. Groups
can include more than two molecules or mixed molecular types (e.g., protein
plus ligand or ligand plus ion). Existing ≤5 Å contact filtering and
`--all-pairs` apply to group pairs as well.

A virtual group collects the original PAE tokens; it does **not** combine or
average the separate pairwise scores or define a new reference frame. `ipsae12`
aligns on `chn1`, scores `chn2`; `ipsae21` aligns on `chn2`, scores `chn1`.
For grouped rows, `ipsae` is the **maximum of the two valid directions**,
regardless of group composition; ordinary pairwise scoring retains its original
same-type/mixed-type convention.

When a scored group contains a protein, all group units with PAE ≤10 Å
(default) may contribute to the score, but **only qualifying protein residues**
set the protein length used in $d_0$. With nucleic acids but no protein, the
PAE cutoff applies and $d_0$ is fixed at 2 Å. With only ligands and ions, no
PAE cutoff applies and $d_0$ is fixed at 2 Å. Modified polymer residues count
once even if tokenized by atoms. For mixed groups `n0_12/21` reports *all*
selected scoring units; protein-only units are used to calculate `d0_12/21`.

New group rows have `NA` in native ipTM columns (the predictors do not supply
native confidence scores for these virtual pairs). Optional `--write-by-token`,
`--write-interface`, and `--write-pymol` output supports group rows; PyMOL
selections refer to the individual member chains, not a literal `H+L` chain.
The generated PyMOL script sets `cif_use_auth, off` for mmCIF coordinates so
its chain and residue selections match `label_asym_id`/`label_seq_id`.
Alignment tokens for a virtual group are prefixed with their original chain
identifier (for example, `H/VAL:147:CA`) to distinguish residues from
different members; ordinary pairwise alignment-token labels do not change.
This feature requires a **predicted PAE matrix**; an experimental PDB structure
alone does not contain PAE and cannot be scored.

## Run a directory of models

```bash
# Inspect auto-discovered coordinate/PAE/confidence matches first:
ipsae predictions/ --dry-run

# Process recursively, write individual TSVs beside each model,
# and create one combined file (with a single header):
ipsae predictions/ --output-file all_ipsae.tsv

# Only examine the given directory, not its subdirectories:
ipsae predictions/ --no-recursive --output-file all_ipsae.tsv
```

The combined TSV starts with a `model` column giving the relative path to each
input structure. Ambiguous filename matches are skipped with a warning rather
than incorrectly pairing a PAE matrix with a structure. For ColabFold
results containing both `relaxed_` and `unrelaxed_` models for a given rank,
model and seed, the unrelaxed structure is used; if only the relaxed model
is present, it is used instead. ColabFold `scores_*.json.gz` files are read
directly, without decompression.

## Try the bundled example

```bash
ipsae tests/data/model.pdb tests/data/pae.json tests/data/summary.json \
  --output example_ipsae.tsv
cat example_ipsae.tsv
```

The synthetic example produces **three molecular-pair rows**. For the
full automated tests:

```bash
python -m pip install pytest
python -m pytest -q
```

## How scoring works

For a pair of molecules, `ipsae12` aligns on `chn1` and scores `chn2`;
`ipsae21` aligns on `chn2` and scores `chn1`. The final `ipsae` is:

- For **same-type pairs** (protein–protein, nucleic acid–nucleic acid,
  ligand–ligand), the larger directional value.
- For **different-type pairs**, the direction aligning on the more complex
  molecule, ordered **protein > nucleic acid > ligand > ion**.

Scored protein and nucleic-acid units use a PAE cutoff of **≤10 Å** by default;
ligands and ions have **no PAE cutoff**. Protein $d_0$ follows the
length-dependent TM-style formula with a 1 Å minimum, and nucleic acids,
ligands and ions use fixed **$d_0=2$ Å**. By default, only molecule pairs
having a heavy-atom contact **≤5 Å** are reported, but the contact filter does
**not** limit which PAE-selected units contribute to ipSAE. Use `--all-pairs`
if needed. Ion–ion scores are undefined and not reported.

The TSV puts the most useful information first:

```text
chn1 chn2 type1 type2 mol1 mol2 iptm ipsae ipsae12 ipsae21
ipsae_source align_token12 align_token21 iPAE12 iPAE21 ...
```

Molecule labels report the modeled polymer length, for example `AA_403`,
`RNA_6`, `DNA_10`, or `NUC_109` (one chain containing both DNA and RNA).
Ligands and ions retain labels such as `ATP` and `MG`.

See **[full scoring conventions, all columns, options and format details](docs/REFERENCE.md)**.
Run `ipsae --help` for the complete list of options, including optional PAE
plots and PyMOL selections.

## Validation and limitations

- **AlphaFold3 and Boltz-2:** directional scores have been numerically
  compared against the previous implementation for AURKA–TPX2–ATP–Mg
  examples; all six contacting pairs agreed to six decimal places in each
  case. Batch processing has been exercised on 1,600 seeded AURKA models per
  predictor.
- **Nucleic acids:** initial AF3 DNA and Boltz-2 RNA tests have been performed.
  Wider DNA/RNA benchmark regression is still needed.
- **ColabFold 1.5:** PDB or compressed `.pdb.gz` coordinates and the matching
  single JSON with `pae` and scalar `iptm` have been tested against a real
  RAF1–KSR1 heterodimer (648 and 923 residues). For two-chain complexes,
  scalar `iptm` populates the pair's `iptm` column; directional native ipTM
  columns are `NA`.
- **Other predictors:** readers are included for additional formats, but
  remain *experimental* pending validation on real output files.
- **Platforms:** installed and run on Linux and macOS; Windows has not been
  tested. The automated test suite uses synthetic examples.
- Models need PAE matrices with a correctly inferred or specified token order;
  ambiguous mappings are rejected. Different predictors may report native
  ipTM differently, and some directions can be undefined (`NA`).

This is an **alpha research release**. Please report reproducible problems
with predictor, version, command, file types and a minimal test case in
GitHub Issues. Do not assume that two different scoring conventions are
numerically interchangeable.

## Citation and license

This package extends the original ipSAE work:
[**Dunbrack laboratory — original ipSAE preprint**](https://doi.org/10.1101/2025.02.10.637595).
A separate ipSAE2 manuscript is in preparation. See [CITATION.cff](CITATION.cff)
for how to cite this software release in the meantime.

Licensed under the **MIT License**; see [LICENSE](LICENSE).
