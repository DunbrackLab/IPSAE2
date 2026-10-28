# ipSAE2 — directional PAE-based interaction confidence

**Alpha release 0.1.4** for the ipSAE2 preprint. Built from the
Dunbrack laboratory's `build_pae_data11.py`, `ipsae_scores7.py`, and
`native_iptm2.py` prototypes. MIT license.

ipSAE2 calculates two asymmetric PAE-based scores for a molecular pair and
chooses a final `ipsae` based on the molecule types. It does **not** calculate
pDockQ, pDockQ2, LIS, or c-ipSAE. The ordinary scoring pass does not compute
an all-against-all atom distance matrix. It uses atom contacts only to decide
which pairs to report and to count contacting scored units.

## Installation

Python **3.10+** on Linux or macOS. A CPU is sufficient (no CUDA, PyTorch, or
AF3/Boltz installation required).

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install .
```

Alternatively, in an existing Conda environment:

```bash
conda create -n ipsae2 python=3.11 pip
conda activate ipsae2
python -m pip install .
```

Once installed, `python -m ipsae` is also available as an alternative to the `ipsae` entry point.

The mandatory dependencies are NumPy, pandas, SciPy, and Gemmi. Matplotlib is
optional, installed with `pip install '.[plot]'` when PAE PNG images are needed.

Windows has not yet been tested.

## Basic usage

```bash
ipsae model.cif full_data.json
ipsae full_data.json summary_confidences.json model.cif
ipsae pae_model_0.npz model_0.cif confidence_model_0.json

# Equivalently, after installing the package:
python -m ipsae model.cif full_data.json summary_confidences.json
```

Specify exactly two files (coordinates + PAE) or three files (coordinates +
PAE + native confidences), **in any order**. The coordinate file may be PDB,
CIF or mmCIF; PAE and confidences may be JSON or NPZ. A separate confidences
file is optional; the PAE file is also searched for native ipTM when it
contains compatible confidence keys. Missing native values are written `NA`.
An explicitly supplied malformed file is an error.

For a model named `model.cif`, the default output is `model_ipsae.tsv`, in the
same directory. Use `--output scores.tsv` to choose another path.

### Directory / batch mode

Give a **directory** rather than individual input files. The directory is
searched **recursively by default**, including arbitrarily nested model folders.
Each successfully matched model writes `<coordinates>_ipsae.tsv` beside its
original coordinates, and the combined TSV has just **one header** plus a
leading `model` column with the coordinate filename *relative to the input root*.
The combined file goes to the explicit path supplied by `--output-file`
(alias `-output_file`, or `--output`), or to `ipsae_all.tsv` in the input root.

```bash
# Preview pairings without calculating scores (highly recommended first):
ipsae /path/to/aurka_models/ --dry-run

# Recurse through all subdirectories and combine every successful model:
ipsae /path/to/af3_output/ \
  --output-file dna_af3_ipsae.tsv

# Alternative spelling requested for command-line compatibility:
ipsae output_folder/ -output_file output.tsv --pae-cutoff 10

# Limit search to the specified directory:
ipsae output_folder/ --no-recursive --output-file output.tsv
```

Matching is restricted to the **same folder** as each coordinate file. Common
predictor filenames such as `model.cif`/`confidences.json`/
`summary_confidences.json`, `name_model_4.cif`/
`name_model_4_confidences.json`/`name_model_4_summary_confidences.json`,
and `model_0.cif`/`pae_model_0.npz`/`confidence_model_0.json`
are recognized. If the available files cannot be matched unambiguously,
that model is skipped with an explicit warning rather than being assigned a
possibly incorrect PAE matrix. A missing native confidence file is not an
error: its scores are `NA`. Other failed models are reported and processing
continues; a nonzero exit status indicates one or more processing errors.

Single-model commands remain unchanged.

```bash
ipsae model.cif pae.json --pae-cutoff 10 --contact-cutoff 5
ipsae model.cif pae.json --all-pairs --output everything.tsv
ipsae model.cif pae.json --write-by-token --write-interface --write-pymol
ipsae model.cif pae.json --write-pae-png --write-pae-df
ipsae --help
```

## Pair inclusion and ordering

By default, only unordered molecule pairs containing at least one **heavy
atom–heavy atom distance ≤ 5.0 Å** are reported. The distance threshold is
adjustable with `--contact-cutoff`. Overlapping bounding boxes are tested
with SciPy's `cKDTree`; no full interatomic distance matrix is constructed.
Use `--all-pairs` to report noncontacting pairs too. Ion–ion scores are not
reported since neither ion has a valid reference frame.

Rows are ordered by molecule type **prot, nuc, lig, ion**, then by the first
appearance of each chain in the coordinate file. An unordered pair appears
only once. Within each pair, `12` means **align on chn1, score chn2**.

## Scoring conventions

| Scored type | Scored units | PAE filtering | d0 (Å) |
| --- | --- | --- | --- |
| `prot` | amino-acid residues | PAE ≤ 10 Å (configurable) | `max(1, 1.24*cbrt(n0-15)-1.8)` |
| `nuc` | nucleotides | PAE ≤ 10 Å (configurable) | 2.0 |
| `lig` | all ligand heavy-atom tokens | none | 2.0 |
| `ion` | ion tokens | none | 2.0 |

Use `--pae-cutoff` for proteins and nucleic acids. `--fixed-d0` changes the
constant for nuc/lig/ion; `--protein-min-d0` changes the protein floor. These
options exist for research but deviate from the preprint defaults.

The **final** `ipsae` is the maximum of the directional scores for molecule
pairs of the same type (`prot/prot`, `nuc/nuc`, `lig/lig`). For different types,
the final score is the direction aligned on the more complex molecule, using
the hierarchy `prot > nuc > lig > ion`. The canonical row order means this is
normally `ipsae12`. A missing/invalid alignment frame produces `NA`, not zero;
a valid frame with no PAE-selected scored units produces score zero.

Atom-tokenized modified protein/nucleic-acid residues have one *scoring unit*
per residue: their contributing atom-token scores are first averaged, then
that residue receives the same weight as other residues. `n0` counts scoring
units, not raw atom tokens.

## TSV columns

The columns have deliberately been ordered for viewing with `less -S`: molecule
identities and comparable native/ipSAE values come first, followed by the
alignment-token labels, then diagnostics. `iptm` and `native_iptm12/21` are
printed with **three significant digits**, while `iPAE` and `d0` are printed
with **three decimal places**. The ipSAE scores retain six decimal places.

```text
chn1 chn2 type1 type2 mol1 mol2 iptm ipsae ipsae12 ipsae21
ipsae_source align_token12 align_token21 iPAE12 iPAE21
n0_12 n0_21 d0_12 d0_21 n_scored_contact12 n_scored_contact21
pae_cutoff12 pae_cutoff21 native_iptm12 native_iptm21
```

- `mol1` and `mol2`: compact molecule labels. For example, `AA_403` for a 403-residue protein, `DNA_10` for a
  10-nucleotide DNA strand, `RNA_109` for a 109-nucleotide RNA, and `ATP` or
  `MG` for individual ligands/ions. Polymer lengths count **modeled residues**
  (unique scoring units), not PAE tokens or the full sequence length. A
  modified residue therefore counts once. Mixed canonical DNA and RNA in one
  chain is marked `NUC_N` (rather than choosing the majority type); if the subtype cannot be inferred, `NA_N`. These labels are important because
  the `align_token` columns identify *alignment frames*, **not** ligand/ion
  identities. An ion has no valid alignment frame, so that direction is `NA`.

- `iptm`: predictor-reported pair confidence when available. For
  Boltz-2, the maximum of the two directional native pair scores; for AF3,
  the symmetric pair score. For AF2/ColabFold dimers, a scalar overall ipTM
  is usable as a pair score, but never duplicated into directional columns.
- `native_iptm12/21`: predictor-reported asymmetric pair scores **only** if
  the predictor reports them. Boltz-2's pair matrix indexing is transposed
  into the common `aligned → scored` convention.
- `ipsae12/21`: directional PAE-based scores; `ipsae`: selected final score;
  `ipsae_source`: `ipsae12` or `ipsae21`.
- `iPAE12/21`: average PAE of scored units, **for the maximizing alignment
  frame of that direction**, using the same scored-unit weighting as ipSAE.
  It is not a mean of the entire interchain PAE matrix.
- `n0_12/21`, `d0_12/21`: scored-unit count and d0 used **for the maximizing
  frame**. These differ between directions and can vary across frames.
- `n_scored_contact12/21`: number of scored units contributing to that
  maximizing score which have at least one heavy atom ≤ contact cutoff from
  *any* atom in the partner molecule. This does **not** restrict the ipSAE
  score; ligand scoring still includes every ligand token.
- `pae_cutoff12/21`: scored-side criterion (number or `none`).
- `align_token12/21`: maximizing alignment residue/atom label.

Note: `NA` means an undefined or unavailable quantity. A zero score means the
specified direction is defined, but no scored unit satisfies the PAE cutoff.

## Optional diagnostics

Nothing below is computed or written by default:

| Option | Output |
| --- | --- |
| `--write-token-df` | token table with mapping and frame validity |
| `--write-atom-df` | atom table |
| `--write-by-token` | score for each eligible alignment token |
| `--write-pae-df` | large directional interchain token–token PAE table |
| `--write-pae-png` | PAE PNG (requires matplotlib) |
| `--write-interface` | units that are in contact and satisfy the scored-side PAE rule |
| `--write-pymol` | PyMOL file of residue selections `color_A_B` |

For ligand/ion scored units, the scored-side PAE cutoff is `none`; they are
included in diagnostics if they are in contact. The optional interface
selection is never used in the ipSAE calculation.

## Prediction formats and validation

The core reads of predictor-derived JSON/NPZ follow the existing multi-program
prototype: AlphaFold2/ColabFold, AlphaFold3, Boltz-1/2, Chai-1,
OpenFold3, Protenix, ESMFold2, and Promera. **Support is conditional on having
PAE and coordinate files with a compatible token order.** Native ipTM
availability and representation differ among programs.

Tokenization is inferred by testing modified-residue (atom or residue) and
ligand (atom, residue, or chain) policies, checking PAE dimensions and, where
provided, source token-chain labels or numeric chain assignments. When source
atom-to-token indices have compatible dimensions, they are also checked.
If materially distinct tokenizations remain equally possible, ipSAE2 fails
instead of silently guessing. It does not infer a wholly arbitrary token
permutation without metadata.

Every ligand/ion should have a unique chain (`label_asym_id` in CIF). PDB files
that reuse a protein chain for a ligand or combine multiple nonpolymer
entities under one chain may need conversion or unique molecule IDs. The
program stops rather than silently merging an ion into a protein. Source
frame-validity masks are honored when available; otherwise, three-atom
frame validity is estimated geometrically.

### Validation status for this release

Automated tests use small synthetic PDB/mmCIF and JSON/NPZ matrices, including
AF3-like symmetric confidence matrices, Boltz-2 nested directional matrices,
AF2 scalar ipTM, ions, ligand atoms, and modified-residue grouping. The new scoring path has also been compared against the previous code on
an AF3 and a Boltz-2 AURKA–TPX2–ATP–2Mg example, where all six reported
pairs agreed to six decimal places. Initial AF3 DNA and Boltz-2 RNA runs have been performed; full
numerical regression across DNA/RNA benchmarks and other predictor formats
remains in progress. Treat support beyond tested AF3 and Boltz-2 as experimental. The v0.1.1 fix for quoted mmCIF nucleic-acid atom names is
retained in this version.

Run the tests with:

```bash
python -m pip install '.[test]'
pytest -q
```

The `tests/data` directory includes synthetic inputs; its scores are
illustrative and do not represent a biological example.

## Version 0.1.1 fixes

- Decode mmCIF quoted atom IDs via `gemmi.cif.as_string` (e.g. `"C1'"` → `C1'`).
  This restores nucleotide reference-frame detection and the associated DNA/RNA
  `ipsae12`/`ipsae21` scores; previous release 0.1.0 incorrectly produced `NA`
  whenever alignment was on nucleic acid in AF3-style mmCIF files.
- Label nucleotide reference tokens using `C3'` rather than `C1'` to follow
  the nucleotide token convention. (This changes labels, not the PAE score.)
- For a direction lacking a valid alignment frame, report `NA` for its PAE
  cutoff as well as its directional score.


## Version 0.1.3 changes

- `mol1` and `mol2` now label polymers with their **modeled residue count**:
  `AA_122`, `DNA_10`, or `RNA_109`. Ligands and ions retain their names (`ATP`,
  `MG`, etc.). This is a display-only change; scores, PAE filters, and chain
  ordering are unchanged.
- Labels count each atom-tokenized modified residue once rather than counting
  each of its PAE tokens.


## Version 0.1.4 changes

- A **single polymer chain containing both RNA and DNA** is now labeled `NUC_N`
  (for example `NUC_109`), rather than `DNA/RNA_N`. An all-RNA chain is
  still `RNA_N` and an all-DNA chain is still `DNA_N`. Separate RNA and DNA
  chains retain their individual types. Chains consisting only of unrecognized
  modified nucleotide IDs retain the conservative `NA_N` label.
- This is a display-only modification; directional scores, tokenization,
  native ipTM, contact filtering, and other columns are unchanged.

### Quick macOS / Linux installation and smoke test

Use Python **3.10 or later**. Inside the unzipped `ipsae2` source directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install .
ipsae --version
ipsae tests/data/model.pdb tests/data/pae.json tests/data/summary.json --output mac_test.tsv
cat mac_test.tsv
```

The version should be `ipSAE2 0.1.4`; the sample input should produce a
three-row TSV. You can optionally run all bundled tests using
`python -m pip install pytest` and, from the **parent directory**
containing `ipsae2/`, `python -m pytest -q ipsae2/tests`. Running tests from
inside the source directory should also work after installation.

Linux and macOS installation have been tested, including a successful macOS
run on a Boltz-2 protein–RNA example. CPU architecture was not documented in
that test; Windows remains untested. Install in a
**dedicated** virtual environment to avoid disrupting predictor dependencies.
