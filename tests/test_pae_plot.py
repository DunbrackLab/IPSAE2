"""Regression tests for optional PAE display transformations."""

import numpy as np
import pandas as pd
import pytest

from ipsae.pae_plot import prepare_pae_plot, save_pae_png


def sample():
    # A=protein (2 tokens), B=ligand (3 atoms), C=ion (1 token),
    # D=DNA (2 tokens). All PAE entries unique to test directionality.
    tokens = pd.DataFrame({
        "chain": ["A", "A", "B", "B", "B", "C", "D", "D"],
        "type": ["prot"] * 2 + ["lig"] * 3 + ["ion"] + ["nuc"] * 2,
    })
    pae = np.arange(64, dtype=float).reshape(8, 8)
    return pae, tokens


def test_default_omits_ion_but_not_ligand():
    pae, tokens = sample()
    result = prepare_pae_plot(pae, tokens)
    assert result.ions_hidden
    assert result.values.shape == (7, 7)
    assert result.blocks == [
        ("A", "prot", 0, 2), ("B", "lig", 2, 5), ("D", "nuc", 5, 7)
    ]
    assert result.values[2, 5] == pae[2, 6]
    assert result.values[5, 2] == pae[6, 2]
    assert result.values[2, 5] != result.values[5, 2]


def test_ligand_scale_only_stretches_ligand_tokens():
    pae, tokens = sample()
    result = prepare_pae_plot(pae, tokens, ligand_scale=4)
    assert result.values.shape == (16, 16)  # 2 + 3*4 + 2
    assert result.blocks == [
        ("A", "prot", 0, 2), ("B", "lig", 2, 14), ("D", "nuc", 14, 16)
    ]
    # These fourfold blocks reproduce their source value, no averaging.
    np.testing.assert_array_equal(result.values[2:6, 2:6],
                                  np.full((4, 4), pae[2, 2]))
    np.testing.assert_array_equal(result.values[2:6, 14:16],
                                  np.tile(pae[2, 6:8], (4, 1)))


def test_include_ions_retains_ion_boundary_at_one_pixel():
    pae, tokens = sample()
    result = prepare_pae_plot(pae, tokens, ligand_scale=4, include_ions=True)
    assert not result.ions_hidden
    assert result.values.shape == (17, 17)
    assert result.blocks[2] == ("C", "ion", 14, 15)
    assert result.values[14, 0] == pae[5, 0]


def test_ion_only_model_falls_back_to_original_tokens():
    pae = np.array([[0., 1.], [2., 0.]])
    tokens = pd.DataFrame({"chain": ["X", "Y"], "type": ["ion", "ion"]})
    result = prepare_pae_plot(pae, tokens)
    np.testing.assert_array_equal(result.values, pae)
    assert not result.ions_hidden


def test_invalid_scale_and_dimensions():
    pae, tokens = sample()
    for bad in (0, 11, 1.5, True):
        with pytest.raises(ValueError):
            prepare_pae_plot(pae, tokens, ligand_scale=bad)
    with pytest.raises(ValueError):
        prepare_pae_plot(pae[:-1], tokens)


def test_save_png(tmp_path):
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    pae, tokens = sample()
    path = tmp_path / "example_pae.png"
    assert save_pae_png(pae, tokens, path, ligand_scale=4) == path
    assert path.is_file() and path.stat().st_size > 1000
