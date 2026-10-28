# Changelog

## v0.1.5-alpha — 2026-10-28

- Accept command-line options before, after, or between positional input files.
- Added CLI argument-order tests; no changes to scoring or output columns.


## v0.1.4-alpha — 2026-10-28

First public **alpha** source snapshot. The Python package retains version
`0.1.4` and is compatible with the v0.1.4 wheel previously tested on Linux
and macOS.

- Directional ipSAE2 scores and molecule-type selection for proteins,
  nucleic acids, ligands and ions.
- Heavy-atom ≤5 Å pair inclusion and contact-unit diagnostics.
- Native AF3 and Boltz-2 ipTM import when provided.
- Recursive directory processing, per-model TSV output and combined TSV.
- `AA_N`, `DNA_N`, `RNA_N`, `NUC_N` labels based on modeled polymer lengths.
- macOS/Linux installation and synthetic test suite.

Other predictor formats are experimental. No stability guarantee for the
command-line interface or TSV schema until a non-alpha release.
