# Changelog

## v0.1.9-alpha — 2026-10-30

- Support standard AlphaFold3 Server output filenames, matching model_N.cif with full_data_N.json and summary_confidences_N.json.
- Exclude AF3 Server template structures from recursive directory processing.
- Add an AF3 Server regression test covering five models and template exclusion; all 35 tests pass.

## v0.1.8-alpha — 2026-10-30

- Fix ColabFold 1.5 file matching when filenames omit rank numbers, using model and seed identifiers.
- Clarify dry-run messages when native ipTM is included in the PAE JSON rather than a separate confidence file.
- Add a regression test covering 100 ColabFold model/seed combinations; all 34 tests pass.

## v0.1.7-alpha — 2026-10-30

- Support standard ColabFold 1.5 `scores_*.json.gz` files for both PAE and native scalar ipTM.
- Match `JOB_scores_rank_NNN_...json.gz` with `JOB_unrelaxed_rank_NNN_...pdb.gz` (or relaxed PDBs), retaining full rank/model/seed identity.
- Avoid scoring a ColabFold prediction twice when both relaxed and unrelaxed coordinate files are present; prefer unrelaxed, with relaxed-only fallback.
- Retain existing AF3/Boltz-2 score algorithms and other batch matching rules.


## v0.1.6-alpha — 2026-10-30

- Tested ColabFold 1.5 two-file PDB/JSON inputs, including `.pdb.gz` models.
- Import ColabFold's scalar native ipTM from the PAE JSON for a two-chain model.
- Avoid reloading giant combined PAE/ipTM JSON files for native confidences.
- Match ColabFold's `scores_` JSONs against `unrelaxed_` / `relaxed_` PDBs in directory mode.
- Retain AF3/Boltz-2 numerical scoring and the v0.1.5 CLI unchanged.

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
