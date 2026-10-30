"""Command line entry point."""
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from . import __version__
from .readers import load_model
from .scores import calculate_scores
from .contacts import prepare_geometry, contact_units
from .scores import OUTPUT_COLUMNS
from .batch import discover_models


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Directional ipSAE2 from coordinates, PAE, and optional native confidence output")
    ap.add_argument("files", nargs="+", help="Two or three predictor files in any order, OR a directory containing models (recursive by default)")
    ap.add_argument("--version", action="version", version=f"ipSAE2 {__version__}")
    ap.add_argument("--pae-key", default="pae")
    ap.add_argument("--pae-cutoff", type=float, default=10.0)
    ap.add_argument("--contact-cutoff", type=float, default=5.0)
    ap.add_argument("--fixed-d0", type=float, default=2.0, help="d0 for scored nucleic acids, ligands, and ions (default 2 Å)")
    ap.add_argument("--protein-min-d0", type=float, default=1.0, help="Minimum d0 for scored proteins (default 1 Å)")
    ap.add_argument("--all-pairs", action="store_true", help="Include all molecular pairs, even if not in contact")
    ap.add_argument("--output-file", "-output_file", "--output", dest="output", help="Single-model TSV path, or combined TSV path for directory mode")
    ap.add_argument("--no-recursive", action="store_true", help="When given a directory, scan only that directory")
    ap.add_argument("--dry-run", action="store_true", help="List model/PAE/confidence matches in directory mode without scoring")
    ap.add_argument("--write-token-df", action="store_true")
    ap.add_argument("--write-atom-df", action="store_true")
    ap.add_argument("--write-pae-df", action="store_true", help="Large complete interchain directional token-pair PAE table")
    ap.add_argument("--write-by-token", action="store_true", help="Detailed scores for every eligible alignment token")
    ap.add_argument("--write-pae-png", action="store_true")
    ap.add_argument("--write-interface", action="store_true", help="Write contacting-residue selections")
    ap.add_argument("--write-pymol", action="store_true", help="Write PyMOL selections for contacting residues")
    # Accept flags before, after, or between the positional input files.
    a = ap.parse_intermixed_args(argv)
    if not (len(a.files) == 1 and Path(a.files[0]).is_dir()) and len(a.files) not in (2,3):
        ap.error("Supply a directory or exactly 2 or 3 input files")
    if a.dry_run and not (len(a.files) == 1 and Path(a.files[0]).is_dir()):
        ap.error("--dry-run requires a directory")
    if a.pae_cutoff <= 0 or a.contact_cutoff <= 0 or a.fixed_d0 <= 0 or a.protein_min_d0 <= 0:
        ap.error("Cutoffs and d0 values must be positive")
    return a


def _write_optional(data, options, stem, geom, summary):
    if options.write_token_df:
        data.tokens.to_csv(f"{stem}_token_df.tsv", sep="\t", index=False)
    if options.write_atom_df:
        data.atoms.to_csv(f"{stem}_atom_df.tsv", sep="\t", index=False)
    if options.write_pae_df:
        # Avoid a whole-system Cartesian DataFrame unless explicitly requested.
        with open(f"{stem}_pae_df.tsv", "w") as out:
            out.write("align_token\tscored_token\tchain1\tchain2\tpae\n")
            chains = data.tokens.chain.to_numpy()
            for i in range(len(chains)):
                for j in range(len(chains)):
                    if chains[i] != chains[j]:
                        out.write(f"{i+1}\t{j+1}\t{chains[i]}\t{chains[j]}\t{data.pae[i,j]:.6f}\n")
    if options.write_by_token:
        from .scores import score_direction
        rows = []
        for pair in summary.itertuples():
            for aligned, scored, direction in ((pair.chn1,pair.chn2,"12"),(pair.chn2,pair.chn1,"21")):
                source_idxs = np.flatnonzero(data.tokens.chain.to_numpy() == aligned)
                target_idxs = np.flatnonzero(data.tokens.chain.to_numpy() == scored)
                target_type = str(data.tokens.iloc[target_idxs[0]].type)
                _, in_contact = contact_units(geom[aligned],geom[scored],options.contact_cutoff)
                for ti in source_idxs:
                    if not bool(data.tokens.iloc[ti]["frame_valid"]):
                        continue
                    result = score_direction(data.pae,data.tokens,[ti],target_idxs,target_type,in_contact,
                                             options.pae_cutoff,options.fixed_d0,options.protein_min_d0)
                    rows.append(dict(chn1=pair.chn1,chn2=pair.chn2,direction=direction,
                                     align_token=result.best_token,ipsae=result.score,iPAE=result.ipae,
                                     n0=result.n0,d0=result.d0,n_scored_contact=result.ncontact))
        pd.DataFrame(rows,columns=["chn1","chn2","direction","align_token","ipsae","iPAE","n0","d0","n_scored_contact"]).to_csv(
            f"{stem}_ipsae_by_token.tsv",sep="\t",index=False,na_rep="NA",float_format="%.6f")
    if options.write_pae_png:
        try:
            import matplotlib.pyplot as plt
        except ImportError as e:
            raise RuntimeError("PAE images require pip install 'ipsae2[plot]'") from e
        fig, ax = plt.subplots(figsize=(6,6))
        im=ax.imshow(data.pae, origin="upper", cmap="viridis", vmin=0, vmax=32)
        ax.set_xlabel("Scored token")
        ax.set_ylabel("Aligned token")
        fig.colorbar(im, ax=ax, label="PAE (Å)")
        fig.savefig(f"{stem}_pae.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
    if options.write_interface or options.write_pymol:
        # Optional intersection of spatial contact with the scored-side PAE
        # criterion. This is purely diagnostic; it never feeds back into ipSAE.
        rows = []
        selections = []
        chain_arr = data.tokens.chain.to_numpy()
        for pair in summary.itertuples(index=False):
            c1, c2 = pair.chn1, pair.chn2
            touched1, touched2 = contact_units(geom[c1], geom[c2], options.contact_cutoff)
            for chain, other, touched in ((c1,c2,touched1),(c2,c1,touched2)):
                indices = np.flatnonzero(chain_arr == chain)
                source = np.flatnonzero((chain_arr == other) & data.tokens.frame_valid.to_numpy(dtype=bool))
                t = str(data.tokens.iloc[indices[0]].type)
                if t in ("prot", "nuc"):
                    if len(source):
                        low = data.pae[np.ix_(source, indices)] <= options.pae_cutoff
                        good = set(data.tokens.iloc[indices[np.any(low,axis=0)]].unit_key)
                    else:
                        good = set()
                else:
                    # Ligands and ions have no scored-side PAE cutoff.
                    good = set(data.tokens.iloc[indices].unit_key)
                selected = touched & good
                for unit in sorted(selected):
                    rows.append(dict(chain=chain, partner=other, unit=unit))
                # PyMOL selections operate at residue level. Atom-specific
                # ligand tokens are grouped into their parent ligand residue.
                residues = data.atoms[(data.atoms.chain == chain) &
                                      (data.atoms.unit_key.isin(selected))]
                resnums = list(dict.fromkeys(residues.resnum.astype(str)))
                if resnums:
                    selections.append((chain,other,"+".join(resnums)))
        if options.write_interface:
            pd.DataFrame(rows, columns=["chain", "partner", "unit"]).to_csv(
                f"{stem}_interface.tsv", sep="\t", index=False)
        if options.write_pymol:
            with open(f"{stem}_interface.pml", "w") as f:
                f.write("# Contact AND PAE-qualified scored units (no effect on ipSAE).\n")
                f.write(f'load "{data.files[1]}"\n')
                for chain,other,resnums in selections:
                    f.write(f"select color_{chain}_{other}, chain {chain} and resi {resnums}\n")


def _formatted_scores(scores):
    """Retain 6-place ipSAE; compact native ipTM; 3-place PAE and d0."""
    out = scores.copy()
    styles = {'iptm': '#.3g', 'native_iptm12': '#.3g', 'native_iptm21': '#.3g',
              'ipsae': '.6f', 'ipsae12': '.6f', 'ipsae21': '.6f',
              'iPAE12': '.3f', 'iPAE21': '.3f', 'd0_12': '.3f', 'd0_21': '.3f'}
    for col, spec in styles.items():
        if col in out:
            out[col] = out[col].apply(
                lambda value: format(float(value), spec) if pd.notna(value) else 'NA')
    return out


def _score_one(files, args, verbose=True):
    data = load_model(files, pae_key=args.pae_key, verbose=verbose)
    geom = prepare_geometry(data.atoms)
    scores = calculate_scores(data.pae, data.tokens, geom, data.native,
                              pae_cutoff=args.pae_cutoff,
                              contact_cutoff=args.contact_cutoff,
                              fixed_d0=args.fixed_d0,
                              protein_min_d0=args.protein_min_d0,
                              all_pairs=args.all_pairs)
    coords = Path(data.files[1])
    stem = coords.with_suffix('')
    if coords.name.lower().endswith('.pdb.gz'):
        stem = stem.with_suffix('')
    if verbose:
        print(f"ipSAE2 {__version__}: {len(data.tokens)} tokens, {len(geom)} molecules, "
              f"{len(scores)} reported pairs")
        print(f"Tokenization: {data.token_policy}; frames: {data.frame_policy}")
    _write_optional(data, args, stem, geom, scores)
    return _formatted_scores(scores), stem


def _batch_mode(args):
    root = Path(args.files[0]).resolve()
    jobs, warnings = discover_models(root, recursive=not args.no_recursive)
    for note in warnings:
        print(f'ipSAE2 warning: {note}', file=sys.stderr)
    if not jobs:
        print(f'ipSAE2 error: no unambiguously matched models found under {root}', file=sys.stderr)
        return 1
    print(f'Found {len(jobs)} model(s) under {root}')
    if args.dry_run:
        for coord, pae, conf in jobs:
            print(f'{coord.relative_to(root)}  |  {pae.name}  |  {conf.name if conf else "(no separate confidence file)"}')
        return 0
    combined_path = Path(args.output) if args.output else root / 'ipsae_all.tsv'
    if not combined_path.is_absolute():
        combined_path = Path.cwd() / combined_path
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    succeeded, failed, total = 0, 0, 0
    # Stream each completed model into the combined file: no need to hold
    # a large set of model tables in memory, and the header appears once.
    with combined_path.open('w', encoding='utf-8', newline='') as out:
        out.write('model\t' + '\t'.join(OUTPUT_COLUMNS) + '\n')
        for i, (coord, pae, conf) in enumerate(jobs, 1):
            files = [str(coord), str(pae)] + ([str(conf)] if conf else [])
            try:
                table, stem = _score_one(files, args, verbose=False)
                per_model = Path(str(stem) + '_ipsae.tsv')
                table.to_csv(per_model, sep='\t', index=False, na_rep='NA')
                relative = str(coord.relative_to(root))
                table.insert(0, 'model', relative)
                table.to_csv(out, sep='\t', header=False, index=False, na_rep='NA')
                succeeded += 1
                total += len(table)
                print(f'[{i}/{len(jobs)}] {relative}: {len(table)} pairs')
            except (ValueError, RuntimeError, KeyError, OSError, IndexError) as exc:
                failed += 1
                print(f'ipSAE2 ERROR {coord.relative_to(root)}: {exc}', file=sys.stderr)
    print(f'Completed: {succeeded} model(s), {total} pairs, {failed} failure(s); combined: {combined_path}')
    return 1 if failed else 0


def main(argv=None):
    args = parse_args(argv)
    if len(args.files) == 1 and Path(args.files[0]).is_dir():
        return _batch_mode(args)
    try:
        table, stem = _score_one(args.files, args)
        output = Path(args.output) if args.output else Path(str(stem) + '_ipsae.tsv')
        output.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(output, sep='\t', index=False, na_rep='NA')
        print(f'Wrote {output}')
        return 0
    except (ValueError, RuntimeError, KeyError, OSError, IndexError) as exc:
        print(f'ipSAE2 error: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
