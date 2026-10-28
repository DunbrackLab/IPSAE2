# Publishing ipSAE2 alpha on GitHub

Keep the existing `DunbrackLab/IPSAE` repository unchanged. Publish this
package in a **new** public repository named `DunbrackLab/IPSAE2`.

## Create the GitHub repository

On GitHub, create a new public repository under the DunbrackLab organization
with the name `IPSAE2`. **Do not** generate a GitHub README, license or
.gitignore during creation: they are already included here.

## Push the source (from this unzipped GitHub-ready directory)

```bash
git init -b main
git add .
git commit -m "Initial ipSAE2 alpha release"
git remote add origin https://github.com/DunbrackLab/IPSAE2.git
git push -u origin main
```

You need permission to create and push repositories in the DunbrackLab
organization, and GitHub authentication (e.g. SSH or GitHub CLI).

Once the push completes, check that the README and GitHub Actions tests
appear and that Linux/macOS CI jobs pass.

## Create a GitHub release

From the GitHub repository page, open Releases > Draft a new release.
Choose **Create new tag**: `v0.1.4-alpha`, targeting `main`.
Set title `ipSAE2 v0.1.4 — alpha` and mark **Pre-release**.
Copy the release text below or use CHANGELOG.md.

Optional: attach a Python wheel. It is *not* required for installation;
users can run `pip install .` after cloning the repository.

## Suggested announcement

ipSAE2 is an alpha version of a directional PAE-based interface confidence
scorer for biomolecular complexes. It reports per-pair scores for protein,
DNA/RNA, ligand and ion interactions, can import available native pairwise
ipTM confidence, and processes folders recursively. Tested on Linux and
macOS with representative AF3 and Boltz-2 outputs. Other prediction formats
remain experimental. Installation: clone the repository and run
`python -m pip install .` in a separate Python environment.

## Final review before making public

- Confirm the README, author and MIT license are correct.
- Verify no private model data, credentials or institutional-only paths
  have been committed.
- Run `python -m pytest -q` and ensure GitHub Actions passes.
- Label the release **Pre-release** and leave the original IPSAE repo intact.
