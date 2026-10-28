"""Synthetic regression tests for directional normalization and predictor formats."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest

from ipsae.contacts import MoleculeGeometry, contact_units
from ipsae.readers import load_model, infer_tokens, read_atoms, classify
from ipsae.scores import protein_d0, score_direction, calculate_scores, molecule_label
from ipsae.cli import main

DATA = Path(__file__).parent / "data"


def test_protein_d0():
    assert protein_d0(1) == 1
    assert protein_d0(26) == 1
    assert protein_d0(300) > 4
    assert protein_d0(15, 2.0) == 2


def test_5a_inclusive_contact():
    from scipy.spatial import cKDTree
    def geom(x, key):
        xyz = np.array([[x,0.,0.]])
        return MoleculeGeometry(xyz, np.array([key],object), xyz[0], xyz[0], cKDTree(xyz))
    assert contact_units(geom(0,"a"),geom(5,"b"),5.) == ({"a"},{"b"})
    assert contact_units(geom(0,"a"),geom(5.01,"b"),5.) == (set(),set())


def test_known_pdb_json_native_symmetric():
    d = load_model([str(DATA/'summary.json'), str(DATA/'model.pdb'), str(DATA/'pae.json')])
    from ipsae.contacts import prepare_geometry
    frame = calculate_scores(d.pae, d.tokens, prepare_geometry(d.atoms), d.native)
    assert [(r.chn1,r.chn2) for r in frame.itertuples()] == [('A','C'),('A','D'),('C','D')]
    al = frame[(frame.chn1 == 'A') & (frame.chn2 == 'C')].iloc[0]
    assert al.type1 == 'prot' and al.type2 == 'lig'
    assert al.ipsae == pytest.approx(np.mean([.8,.5,.3076923076923]))
    assert al.ipsae_source == 'ipsae12'
    assert al.n0_12 == 3 and al.n_scored_contact12 == 3
    assert al.iPAE12 == 2.0 and al.d0_12 == 2.0
    assert al.iptm == .8
    assert pd.isna(al.native_iptm12) and pd.isna(al.native_iptm21)
    ai = frame[(frame.chn1 == 'A') & (frame.chn2 == 'D')].iloc[0]
    assert ai.ipsae == .8
    assert pd.isna(ai.ipsae21)
    assert pd.isna(ai.n0_21)


def test_no_confidence_and_all_pairs(tmp_path):
    out=tmp_path/'scores.tsv'
    assert main([str(DATA/'pae.json'), str(DATA/'model.pdb'), '--output', str(out)]) == 0
    a=pd.read_csv(out,sep='\t')
    assert len(a)==3
    assert a.iptm.isna().all()
    out2=tmp_path/'all.tsv'
    assert main([str(DATA/'pae.json'), str(DATA/'model.pdb'), '--all-pairs','--output',str(out2)]) == 0
    b=pd.read_csv(out2,sep='\t')
    assert len(b)==6  # four molecules, all six unordered pairs


def test_native_boltz_reverse_index(tmp_path):
    f=tmp_path/'conf.json'
    json.dump({'pair_chains_iptm':{'0':{'2':0.2},'2':{'0':0.9}}},open(f,'w'))
    d=load_model([str(DATA/'model.pdb'),str(DATA/'pae.json'),str(f)])
    from ipsae.contacts import prepare_geometry
    tab=calculate_scores(d.pae,d.tokens,prepare_geometry(d.atoms),d.native)
    row=tab[(tab.chn1=='A')&(tab.chn2=='C')].iloc[0]
    assert row.iptm==.9
    assert row.native_iptm12==.9  # [scored C][aligned A]
    assert row.native_iptm21==.2  # [scored A][aligned C]


def test_pae_npz_pdb(tmp_path):
    f=tmp_path/'matrix.npz'
    np.savez(f, pae=json.load(open(DATA/'pae.json'))['pae'])
    d=load_model([str(f),str(DATA/'model.pdb')])
    assert d.pae.shape==(6,6)
    assert len(d.tokens)==6


def test_cif_fallback_and_metadata(tmp_path):
    d=load_model([str(DATA/'model.cif'),str(DATA/'pae.json')])
    assert d.tokens.chain.tolist()==['A','B','C','C','C','D']
    bad=json.load(open(DATA/'pae.json'))
    bad['token_chain_ids'][2]='B'
    f=tmp_path/'bad.json';f.write_text(json.dumps(bad))
    with pytest.raises(ValueError,match='token mapping'):
        load_model([str(DATA/'model.pdb'),str(f)])


def test_nucleic_acid_cutoff_and_protein_d0():
    t=pd.DataFrame([
        dict(chain='A',type='prot',resnum='1',resname='ALA',atom_name='CA',token_id=1,frame_valid=True,unit_key='A:res:1:ALA'),
        dict(chain='B',type='nuc',resnum='1',resname='DA',atom_name="C1'",token_id=2,frame_valid=True,unit_key='B:res:1:DA'),
    ])
    pae=np.array([[0,9],[1,0]],float)
    ab=score_direction(pae,t,[0],[1],'nuc',set(),pae_cutoff=10)
    ba=score_direction(pae,t,[1],[0],'prot',set(),pae_cutoff=10)
    assert ab.d0==2 and ba.d0==1
    assert ab.score==pytest.approx(1/(1+(9/2)**2))
    assert ba.score==.5
    ab_low=score_direction(pae,t,[0],[1],'nuc',set(),pae_cutoff=8)
    assert ab_low.score==0 and ab_low.ipae is None and ab_low.n0==0


def test_modified_residue_equal_weight():
    # Two atom tokens from one modified protein residue count as one unit.
    t=pd.DataFrame([
        dict(chain='A',type='prot',resnum='1',resname='ALA',atom_name='CA',token_id=1,frame_valid=True,unit_key='A:res:1:ALA'),
        dict(chain='B',type='prot',resnum='7',resname='TPO',atom_name='P',token_id=2,frame_valid=True,unit_key='B:res:7:TPO'),
        dict(chain='B',type='prot',resnum='7',resname='TPO',atom_name='O1P',token_id=3,frame_valid=True,unit_key='B:res:7:TPO'),
    ])
    p=np.array([[0,1,3],[1,0,0],[3,0,0]],float)
    v=score_direction(p,t,[0],[1,2],'prot',{'B:res:7:TPO'})
    assert v.n0==1 and v.ncontact==1
    assert v.ipae==2.0 and v.d0==1.0
    assert v.score==pytest.approx((.5+.1)/2)


def test_by_token_diagnostic(tmp_path):
    import shutil
    pdb=tmp_path/'new.pdb';shutil.copy(DATA/'model.pdb',pdb)
    assert main([str(pdb),str(DATA/'pae.json'),'--write-by-token','--write-interface','--write-pymol']) == 0
    assert (tmp_path/'new_ipsae_by_token.tsv').exists()
    assert (tmp_path/'new_interface.tsv').exists()
    assert (tmp_path/'new_interface.pml').exists()


def test_af2_scalar_native_dimer(tmp_path):
    from ipsae.native_iptm import read_native_confidences
    p=tmp_path/'af2_scores.json'
    p.write_text(json.dumps({'iptm':0.732}))
    _,df=read_native_confidences(str(p),['A','B'])
    assert len(df)==2 and df.native_iptm.iloc[0]==.732
    _,nonpair=read_native_confidences(str(p),['A','B','C'])
    assert nonpair is None or nonpair.empty


def test_af2_pae_list_json(tmp_path):
    original=json.load(open(DATA/'pae.json'))['pae']
    f=tmp_path/'af2_pae.json'
    f.write_text(json.dumps([{'predicted_aligned_error':original}]))
    d=load_model([str(DATA/'model.pdb'),str(f)])
    assert d.pae.shape==(6,6)


def test_quoted_cif_dna_frames_and_nucleic_ion_scores(tmp_path):
    """Regression: AF3 quotes ribose atom names; nucleotide frames must survive CIF parsing."""
    cif = tmp_path / 'dna_ion.cif'
    rows = []
    for chain, offset in [('C', 0.), ('D', 2.)]:
        # Non-collinear C1'/C3'/C4' triplet, as in AF3 nucleotide frames.
        for atom, (x, y, z) in {
            'P': (0.,0.,0.), "C1'": (0.,1.,0.),
            "C3'": (1.,0.,0.), "C4'": (0.,0.,1.),
        }.items():
            quoted = f'"{atom}"' if "'" in atom else atom
            rows.append(f'ATOM {quoted} DA {chain} 1 C {x + offset} {y} {z}')
    rows.append('HETATM MG MG O 1 MG 1. 1. 1.')
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
''' + '\n'.join(rows) + '\n')
    pae = tmp_path / 'pae.json'
    pae.write_text(json.dumps({'pae': [[0,1,2],[3,0,4],[9,9,0]],
                              'token_chain_ids': ['C','D','O']}))
    d = load_model([str(cif), str(pae)])
    assert d.atoms.loc[d.atoms.atom_name.str.contains('C1', regex=False), 'atom_name'].tolist() == ["C1'", "C1'"]
    assert d.tokens.frame_valid.tolist() == [True, True, False]
    assert d.tokens.atom_name.tolist() == ["C3'", "C3'", 'MG']
    assert molecule_label(d.tokens, 'C', 'nuc') == 'DNA_1'
    assert molecule_label(d.tokens, 'D', 'nuc') == 'DNA_1'
    from ipsae.contacts import prepare_geometry
    results = calculate_scores(d.pae, d.tokens, prepare_geometry(d.atoms))
    co = results[(results.chn1=='C') & (results.chn2=='O')].iloc[0]
    assert co.ipsae12 == pytest.approx(0.5)
    assert co.ipsae == pytest.approx(0.5)
    assert co.iPAE12 == pytest.approx(2.0)
    assert co.n0_12 == 1
    assert pd.isna(co.ipsae21) and co.pae_cutoff21 == 'NA'
    cd = results[(results.chn1=='C') & (results.chn2=='D')].iloc[0]
    assert cd.ipsae12 == pytest.approx(0.8)
    assert cd.ipsae21 == pytest.approx(1 / (1 + (3 / 2)**2))
    assert cd.ipsae == pytest.approx(0.8)
    do = results[(results.chn1=='D') & (results.chn2=='O')].iloc[0]
    assert do.ipsae12 == pytest.approx(0.2)


def test_compact_summary_column_order_and_precision(tmp_path):
    path = tmp_path/'score.tsv'
    assert main([str(DATA/'model.pdb'), str(DATA/'pae.json'), str(DATA/'summary.json'),
                 '--output', str(path)]) == 0
    line = path.read_text().splitlines()
    names = line[0].split('\t')
    assert names[:13] == ['chn1','chn2','type1','type2','mol1','mol2',
                          'iptm','ipsae','ipsae12','ipsae21','ipsae_source',
                          'align_token12','align_token21']
    assert names[-2:] == ['native_iptm12','native_iptm21']
    row = dict(zip(names, line[1].split('\t')))
    assert row['iptm'] == '0.800'
    assert row['ipsae'].count('.') == 1 and len(row['ipsae'].split('.')[1]) == 6
    assert row['iPAE12'] == '2.000' and row['d0_12'] == '2.000'
    assert row['mol1'] == 'AA_1' and row['mol2'] == 'ATP'


def test_recursive_batch_and_single_header(tmp_path):
    import shutil
    root = tmp_path/'models'
    for sub in ['a','nested/b']:
        folder = root/sub
        folder.mkdir(parents=True)
        shutil.copy(DATA/'model.pdb', folder/'sample_model.pdb')
        shutil.copy(DATA/'pae.json', folder/'sample_confidences.json')
        shutil.copy(DATA/'summary.json', folder/'sample_summary_confidences.json')
    combined = tmp_path/'combined.tsv'
    assert main([str(root), '-output_file', str(combined)]) == 0
    assert (root/'a/sample_model_ipsae.tsv').exists()
    assert (root/'nested/b/sample_model_ipsae.tsv').exists()
    a = pd.read_csv(combined, sep='\t')
    assert len(a) == 6
    assert a.columns[0] == 'model'
    assert set(a['model']) == {'a/sample_model.pdb', 'nested/b/sample_model.pdb'}
    assert set(a.iptm) == {0.8, 0.7, 0.9}
    assert combined.read_text().count('model\tchn1\t') == 1


def test_batch_matching_multiple_models_one_directory(tmp_path):
    import shutil
    root = tmp_path/'models'; root.mkdir()
    for prefix in ('sample0','sample1'):
        shutil.copy(DATA/'model.pdb', root/(prefix+'_model.pdb'))
        shutil.copy(DATA/'pae.json', root/(prefix+'_confidences.json'))
        shutil.copy(DATA/'summary.json', root/(prefix+'_summary_confidences.json'))
    from ipsae.batch import discover_models
    jobs, warnings = discover_models(root)
    assert len(jobs) == 2 and warnings == []
    assert all('summary_confidences' in job[2].name for job in jobs)
    assert main([str(root), '--dry-run']) == 0
    assert not (root/'ipsae_all.tsv').exists()
    assert main([str(root), '--no-recursive']) == 0
    assert len(pd.read_csv(root/'ipsae_all.tsv', sep='\t')) == 6


def test_skip_ambiguous_model_matches(tmp_path):
    import shutil
    root=tmp_path/'models'; root.mkdir()
    shutil.copy(DATA/'model.pdb', root/'sample_model.pdb')
    shutil.copy(DATA/'pae.json', root/'sample_pae.json')
    shutil.copy(DATA/'pae.json', root/'pae_sample.json')
    from ipsae.batch import discover_models
    jobs, warnings=discover_models(root)
    assert jobs == []
    assert any('ambiguous' in x for x in warnings)


def test_boltz_style_directory_matching(tmp_path):
    import shutil
    root = tmp_path / 'boltz_outputs'
    root.mkdir()
    for ix in (0, 1):
        stem = f'complex_model_{ix}'
        shutil.copy(DATA/'model.pdb', root/f'{stem}.pdb')
        np.savez(root/f'pae_{stem}.npz', pae=np.asarray(json.loads((DATA/'pae.json').read_text())['pae']))
        (root/f'confidence_{stem}.json').write_text((DATA/'summary.json').read_text())
    from ipsae.batch import discover_models
    jobs, warnings = discover_models(root)
    assert len(jobs) == 2 and warnings == []
    assert all(job[1].name.startswith('pae_') and job[2].name.startswith('confidence_') for job in jobs)
    assert main([str(root), '--output-file', str(tmp_path/'combined.tsv')]) == 0
    assert len(pd.read_csv(tmp_path/'combined.tsv', sep='\t')) == 6


def test_molecule_labels_count_residues_not_atomic_tokens():
    """A single PTM residue has multiple tokens but contributes length one."""
    t = pd.DataFrame([
        ('A', 'prot', '1', 'ALA', 'A:res:1:ALA'),
        ('A', 'prot', '2', 'TPO', 'A:res:2:TPO'),
        ('A', 'prot', '2', 'TPO', 'A:res:2:TPO'),
        ('A', 'prot', '2', 'TPO', 'A:res:2:TPO'),
        ('A', 'prot', '3', 'GLY', 'A:res:3:GLY'),
        ('C', 'nuc', '1', 'DA', 'C:res:1:DA'),
        ('C', 'nuc', '2', '5MC', 'C:res:2:5MC'),
        ('C', 'nuc', '2', '5MC', 'C:res:2:5MC'),
        ('C', 'nuc', '3', 'DT', 'C:res:3:DT'),
        ('D', 'nuc', '1', 'A', 'D:res:1:A'),
        ('D', 'nuc', '2', 'U', 'D:res:2:U'),
        ('E', 'nuc', '1', 'DA', 'E:res:1:DA'),
        ('E', 'nuc', '2', 'U', 'E:res:2:U'),
        ('L', 'lig', '1', 'ATP', 'L:token:14'),
        ('L', 'lig', '1', 'ATP', 'L:token:15'),
        ('M', 'ion', '1', 'MG', 'M:token:16'),
    ], columns=['chain', 'type', 'resnum', 'resname', 'unit_key'])
    assert molecule_label(t, 'A', 'prot') == 'AA_3'
    assert molecule_label(t, 'C', 'nuc') == 'DNA_3'
    assert molecule_label(t, 'D', 'nuc') == 'RNA_2'
    assert molecule_label(t, 'E', 'nuc') == 'NUC_2'
    assert molecule_label(t, 'L', 'lig') == 'ATP'
    assert molecule_label(t, 'M', 'ion') == 'MG'


def test_molecule_labels_in_batch_output(tmp_path):
    from shutil import copyfile
    root = tmp_path / 'models'; root.mkdir()
    copyfile(DATA / 'model.pdb', root / 'one_model.pdb')
    copyfile(DATA / 'pae.json', root / 'one_confidences.json')
    assert main([str(root), '--output-file', str(tmp_path / 'combined.tsv')]) == 0
    lines = (tmp_path / 'combined.tsv').read_text().splitlines()
    data = pd.read_csv(tmp_path / 'combined.tsv', sep='\t')
    assert len(lines) == len(data) + 1
    assert data.loc[0, 'mol1'] == 'AA_1'
    assert data.loc[0, 'mol2'] == 'ATP'
    assert 'AA_1' in set(data.mol1)


def test_mixed_rna_dna_uses_nuc_not_majority_type():
    # Nine DNA residues and one RNA residue in a single polymer.
    # A modified residue with multiple atom tokens still counts once.
    rows = [("C", "nuc", str(i), "DA", f"C:res:{i}:DA") for i in range(1,10)]
    rows.extend([
        ("C", "nuc", "10", "U", "C:res:10:U"),
        ("C", "nuc", "10", "U", "C:res:10:U"),
        ("D", "nuc", "1", "5MC", "D:res:1:5MC"),
    ])
    t = pd.DataFrame(rows, columns=["chain","type","resnum","resname","unit_key"])
    assert molecule_label(t, "C", "nuc") == "NUC_10"
    assert molecule_label(t, "D", "nuc") == "NA_1"  # Unknown subtype, not demonstrably mixed.
