"""Virtual-molecule groups: pairwise compatibility, directionality and identifiers."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ipsae.cli import main, parse_args
from ipsae.contacts import prepare_geometry
from ipsae.readers import read_atoms, load_model
from ipsae.scores import calculate_scores, protein_d0, resolve_groups, score_direction

DATA = Path(__file__).parent / 'data'


def test_grouped_pairs_preserve_existing_rows_and_native(tmp_path):
    d = load_model([str(DATA / 'model.pdb'), str(DATA / 'pae.json'),
                    str(DATA / 'summary.json')], verbose=False)
    geo = prepare_geometry(d.atoms)
    original = calculate_scores(d.pae, d.tokens, geo, d.native)
    with_groups = calculate_scores(d.pae, d.tokens, geo, d.native,
                                   groups=['A+B', 'C+D'])
    pd.testing.assert_frame_equal(original, with_groups.iloc[:len(original)].reset_index(drop=True))
    assert len(with_groups) == len(original) + 1
    row = with_groups.iloc[-1]
    assert (row.chn1, row.chn2) == ('A+B', 'C+D')
    assert (row.type1, row.type2) == ('prot+prot', 'lig+ion')
    assert (row.mol1, row.mol2) == ('AA_1+AA_1', 'ATP+MG')
    assert row.ipsae == pytest.approx(max(row.ipsae12, row.ipsae21))
    assert pd.isna(row.iptm) and pd.isna(row.native_iptm12) and pd.isna(row.native_iptm21)
    assert row.n0_12 == 4 and row.d0_12 == 2.0
    assert row.n_scored_contact12 == 4
    assert row.pae_cutoff12 == 'none' and row.pae_cutoff21 == 10


def test_single_group_compared_with_remaining_entities_and_all_pairs():
    d = load_model([str(DATA / 'model.pdb'), str(DATA / 'pae.json')], verbose=False)
    geo = prepare_geometry(d.atoms)
    subset = calculate_scores(d.pae, d.tokens, geo, groups=['A+B'])
    assert [(r.chn1, r.chn2) for r in subset.itertuples()][-2:] == [
        ('A+B', 'C'), ('A+B', 'D')]
    all_rows = calculate_scores(d.pae, d.tokens, geo, groups=['A+B'], all_pairs=True)
    assert len(all_rows) == 8  # original six + grouped vs C,D
    grouped = calculate_scores(d.pae, d.tokens, geo, groups=['A+B', 'C+D'], all_pairs=True)
    assert len(grouped) == 7  # original six + grouped/grouped only


def test_mixed_group_cutoff_d0_uses_protein_only():
    # One alignment residue, 30 protein residues, and 60 ligand atoms.
    # Protein d0 must depend on 30, not the 90 PAE-qualified scored units.
    rows = [dict(chain='A', type='prot', resnum='1', resname='ALA',
                 atom_name='CA', token_id=1, frame_valid=True, unit_key='A:res:1:ALA')]
    for i in range(30):
        rows.append(dict(chain='B', type='prot', resnum=str(i+1), resname='ALA',
                         atom_name='CA', token_id=len(rows)+1, frame_valid=True,
                         unit_key=f'B:res:{i+1}:ALA'))
    for i in range(60):
        rows.append(dict(chain='C', type='lig', resnum='1', resname='BIG',
                         atom_name=f'C{i}', token_id=len(rows)+1,
                         frame_valid=False, unit_key=f'C:token:{len(rows)+1}'))
    tokens = pd.DataFrame(rows)
    pae = np.full((len(tokens), len(tokens)), 2., dtype=float)
    pae[0, -15:] = 20.  # When a protein is present, ligand atoms also obey cutoff.
    result = score_direction(pae, tokens, [0], list(range(1, len(tokens))),
                             'prot+lig', set(), pae_cutoff=10)
    assert result.n0 == 75  # 30 protein + 45 ligand units
    assert result.d0 == pytest.approx(protein_d0(30))
    assert result.score == pytest.approx(1/(1+(2/protein_d0(30))**2))
    assert result.ipae == pytest.approx(2.)
    # Without a protein or nucleotide, no PAE cutoff is applied.
    ligand_only = score_direction(pae, tokens, [0], list(range(31, len(tokens))),
                                  'lig', set(), pae_cutoff=10)
    assert ligand_only.n0 == 60 and ligand_only.d0 == 2.0


def test_mixed_modified_protein_residue_count_once_for_d0():
    toks = [dict(chain='A', type='prot', resnum='1', resname='ALA', atom_name='CA',
                 token_id=1, frame_valid=True, unit_key='A:res:1:ALA')]
    # One modified residue with three PAE tokens, plus 29 standard residues.
    for atom in ('P', 'O1P', 'O2P'):
        toks.append(dict(chain='B', type='prot', resnum='1', resname='TPO',
                         atom_name=atom, token_id=len(toks)+1, frame_valid=True,
                         unit_key='B:res:1:TPO'))
    for i in range(2, 31):
        toks.append(dict(chain='B', type='prot', resnum=str(i), resname='ALA',
                         atom_name='CA', token_id=len(toks)+1, frame_valid=True,
                         unit_key=f'B:res:{i}:ALA'))
    for i in range(40):
        toks.append(dict(chain='C', type='lig', resnum='1', resname='BIG',
                         atom_name=f'C{i}', token_id=len(toks)+1, frame_valid=False,
                         unit_key=f'C:token:{len(toks)+1}'))
    t = pd.DataFrame(toks)
    pae = np.full((len(t), len(t)), 3.0)
    score = score_direction(pae, t, [0], list(range(1, len(t))),
                            'prot+lig', set())
    assert score.n0 == 70  # 30 protein residues + 40 atom-tokenized ligand units
    assert score.d0 == pytest.approx(protein_d0(30))


@pytest.mark.parametrize('groups, message', [
    (['A'], 'Invalid --group'),
    (['A+A'], 'cannot appear twice'),
    (['A+X'], 'Unknown chain'),
    (['A+B', 'B+C'], 'multiple --group'),
    (['A+'], 'Invalid --group'),
])
def test_group_validation(groups, message):
    d = load_model([str(DATA / 'model.pdb'), str(DATA / 'pae.json')], verbose=False)
    with pytest.raises(ValueError, match=message):
        resolve_groups(d.tokens, groups)


def test_cli_grouped_diagnostics_and_baseline_identical(tmp_path):
    path = tmp_path/'model.pdb'
    path.write_bytes((DATA/'model.pdb').read_bytes())
    plain = tmp_path/'normal.tsv'
    grouped = tmp_path/'grouped.tsv'
    assert main([str(path), str(DATA/'pae.json'), '--output', str(plain)]) == 0
    assert main([str(path), str(DATA/'pae.json'), '--group', 'A+B',
                 '--group', 'C+D', '--output', str(grouped),
                 '--write-by-token', '--write-interface', '--write-pymol']) == 0
    before = pd.read_csv(plain, sep='\t')
    after = pd.read_csv(grouped, sep='\t')
    pd.testing.assert_frame_equal(before, after.iloc[:len(before)].reset_index(drop=True))
    assert after[['chn1','chn2']].iloc[-1].tolist() == ['A+B','C+D']
    per_token = pd.read_csv(tmp_path/'model_ipsae_by_token.tsv', sep='\t')
    assert 'A+B' in set(per_token.chn1) and 'C+D' in set(per_token.chn2)
    interface = pd.read_csv(tmp_path/'model_interface.tsv', sep='\t')
    assert 'A+B' in set(interface.chain)
    pymol = (tmp_path/'model_interface.pml').read_text()
    assert 'select color_A_B_C_D_A, chain A and resi' in pymol
    assert 'chain A+B' not in pymol
    assert 'select color_C_D_A_B_C, chain C' in pymol


def test_auth_vs_label_ids_for_2j88_style_cif_and_pdb(tmp_path):
    """2J88 nomenclature: A/H/L in PDB, A/B/C in mmCIF label_asym_id."""
    rows = []
    lines = []
    for i, (label, auth) in enumerate([('A','A'), ('B','H'), ('C','L')]):
        for j, atom in enumerate(('N','CA','C')):
            element = 'N' if atom == 'N' else 'C'
            x, y, z = 2*i + j*0.5, j % 2, 0
            rows.append(f'ATOM {atom} ALA {label} 1 {element} {x} {y} {z} 1 {auth}')
            lines.append(f'ATOM   {i*3+j+1:4d} {atom:4s} ALA {auth}   1    {x:8.3f}{y:8.3f}{z:8.3f}{1.0:6.2f}{90.:6.2f}          {element:>2s}')
    cif = tmp_path/'test.cif'
    cif.write_text('''data_test
loop_
_atom_site.group_PDB
_atom_site.label_atom_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_seq_id
_atom_site.type_symbol
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.auth_seq_id
_atom_site.auth_asym_id
''' + '\n'.join(rows) + '\n')
    pdb = tmp_path/'test.pdb'; pdb.write_text('\n'.join(lines)+'\n')
    assert read_atoms(cif).chain.unique().tolist() == ['A','B','C']
    assert read_atoms(pdb).chain.unique().tolist() == ['A','H','L']
    pae = tmp_path/'pae.json'; pae.write_text(json.dumps({'pae': [[0,2,3],[4,0,5],[6,7,0]]}))
    assert main([str(cif), str(pae), '--group','B+C', '--all-pairs',
                 '--write-pymol', '--output',str(tmp_path/'cif.tsv')]) == 0
    pymol = (tmp_path/'test_interface.pml').read_text()
    assert 'set cif_use_auth, off\nload ' in pymol
    assert 'select color_B_C_A_B, chain B and resi' in pymol
    assert main([str(pdb), str(pae), '--group','H+L', '--all-pairs',
                 '--output',str(tmp_path/'pdb.tsv')]) == 0
    c = pd.read_csv(tmp_path/'cif.tsv', sep='\t')
    p = pd.read_csv(tmp_path/'pdb.tsv', sep='\t')
    assert (c.chn1.iloc[-1],c.chn2.iloc[-1]) == ('B+C','A')
    assert (p.chn1.iloc[-1],p.chn2.iloc[-1]) == ('H+L','A')
    assert c.ipsae.iloc[-1] == p.ipsae.iloc[-1]
    assert c.ipsae12.iloc[-1] == p.ipsae12.iloc[-1]
    assert c.ipsae21.iloc[-1] == p.ipsae21.iloc[-1]


def test_protein_dna_duplex_group_from_predicted_cif(tmp_path):
    """Protein plus two DNA strands: B+C uses nucleic-acid cutoff and fixed d0.

    This is a synthetic predictor-like three-chain mmCIF + directional PAE,
    not a crystallographic PDB without confidence data. Distinct cross-chain
    errors test the combined PAE matrix directly rather than an average of
    separate chain scores.
    """
    num_protein = 35
    num_dna_per_strand = 6
    cif_rows = []
    serial = 0
    def atom(chain, seq, res, atom_name, element, xyz):
        nonlocal serial
        serial += 1
        x, y, z = xyz
        cif_rows.append(f"ATOM {serial} {element} {atom_name} . {res} {chain} "
                        f"{seq} {x:.3f} {y:.3f} {z:.3f} {seq} {chain}")

    # Each protein residue and each nucleotide has a valid three-atom frame.
    for i in range(1, num_protein + 1):
        x = i * 0.09
        atom('A', i, 'ALA', 'N', 'N', (x, 0, 0))
        atom('A', i, 'ALA', 'CA', 'C', (x + .5, 0, 0))
        atom('A', i, 'ALA', 'C', 'C', (x + .5, .7, 0))
    for chain, y in (('B', 2.0), ('C', 3.0)):
        for i in range(1, num_dna_per_strand + 1):
            x = i * 0.20
            res = ('DA', 'DT', 'DC', 'DG', 'DA', 'DT')[i - 1]
            atom(chain, i, res, "C1'", 'C', (x, y, 0))
            atom(chain, i, res, "C3'", 'C', (x + .6, y, 0))
            atom(chain, i, res, "C4'", 'C', (x + .6, y + .7, 0))
    model = tmp_path/'protein_dna_duplex.cif'
    model.write_text('''data_test
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_seq_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.auth_seq_id
_atom_site.auth_asym_id
''' + '\n'.join(cif_rows) + '\n')
    n = num_protein + 2*num_dna_per_strand
    pae = np.full((n, n), 1.0)
    protein = slice(0, num_protein)
    dna_b = slice(num_protein, num_protein + num_dna_per_strand)
    dna_c = slice(num_protein + num_dna_per_strand, n)
    # PAE direction: reference/alignment frame = first index.
    pae[protein, dna_b] = 2.0
    pae[protein, dna_c] = 6.0
    pae[protein, n-1] = 12.0  # Last DNA token is filtered, but only in this direction.
    pae[dna_b, protein] = 4.0
    pae[dna_c, protein] = 8.0
    pae_path = tmp_path/'duplex_pae.json'
    pae_path.write_text(json.dumps({'pae': pae.tolist(),
                                    'token_chain_ids': ['A']*num_protein +
                                                       ['B']*num_dna_per_strand +
                                                       ['C']*num_dna_per_strand}))

    data = load_model([str(model), str(pae_path)], verbose=False)
    assert data.tokens.chain.value_counts().to_dict() == {'A': 35, 'B': 6, 'C': 6}
    assert data.tokens.frame_valid.all()
    geom = prepare_geometry(data.atoms)
    base = calculate_scores(data.pae, data.tokens, geom)
    grouped = calculate_scores(data.pae, data.tokens, geom, groups=['B+C'])
    pd.testing.assert_frame_equal(base, grouped.iloc[:len(base)].reset_index(drop=True))
    assert len(grouped) == len(base) + 1
    row = grouped.iloc[-1]
    assert (row.chn1, row.chn2) == ('B+C', 'A')
    assert row.align_token12.startswith(('B/', 'C/'))
    assert not '/' in row.align_token21
    assert (row.type1, row.type2) == ('nuc+nuc', 'prot')
    assert (row.mol1, row.mol2) == ('DNA_6+DNA_6', 'AA_35')
    assert (row.pae_cutoff12, row.pae_cutoff21) == (10.0, 10.0)
    assert (row.n0_12, row.n0_21) == (num_protein, 11)
    assert row.d0_12 == pytest.approx(protein_d0(num_protein))
    assert row.d0_21 == 2.0
    assert row.ipsae12 == pytest.approx(1/(1+(4/protein_d0(num_protein))**2))
    assert row.ipsae21 == pytest.approx((6*(1/(1+(2/2)**2)) +
                                         5*(1/(1+(6/2)**2)))/11)
    assert row.iPAE21 == pytest.approx((6*2 + 5*6)/11)
    assert row.ipsae == pytest.approx(max(row.ipsae12, row.ipsae21))

    # End-to-end CLI output with both DNA strands as a virtual molecule.
    output = tmp_path/'grouped_dna.tsv'
    assert main([str(model), str(pae_path), '--group', 'B+C', '--output', str(output)]) == 0
    saved = pd.read_csv(output, sep='\t')
    assert saved.chn1.iloc[-1] == 'B+C'
    assert saved.d0_21.iloc[-1] == 2.0
    assert saved.n0_21.iloc[-1] == 11

    # Mixing protein A with DNA B on the scored side: only protein residue
    # tokens control d0, but both protein and DNA units enter the score.
    protein_dna = calculate_scores(data.pae, data.tokens, geom, groups=['A+B'])
    combined = protein_dna.iloc[-1]
    assert (combined.chn1, combined.chn2) == ('A+B', 'C')
    assert combined.n0_21 == num_protein + num_dna_per_strand
    assert combined.d0_21 == pytest.approx(protein_d0(num_protein))
