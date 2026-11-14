"""Optional PAE heatmap plotting, independent of the ipSAE scoring pipeline.

PAE token rows and columns have the same source order.  The plot can hide
single-ion molecules and widen ligand *pixels* without changing the underlying
PAE values used for scoring or exporting tables.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class PlotData:
    """PAE raster and molecule blocks, in displayed pixel coordinates."""

    values: np.ndarray
    blocks: list[tuple[str, str, int, int]]  # chain, type, start, stop (exclusive)
    ions_hidden: bool


def prepare_pae_plot(pae, tokens, ligand_scale=1, include_ions=False):
    """Select token rows/columns and expand ligand cells for a heatmap.

    Each ligand atom token is duplicated ``ligand_scale`` times in *both*
    directions; proteins, nucleic acids, and ions (if included) retain scale
    1.  This is nearest-neighbor visualization only, never averaging PAE or
    modifying scores.  Chain boundaries are based on consecutive runs in the
    token order rather than assuming chain names sort alphabetically.
    """
    if isinstance(ligand_scale, bool) or not isinstance(ligand_scale, (int, np.integer)) or not 1 <= ligand_scale <= 10:
        raise ValueError("ligand_scale must be an integer between 1 and 10")

    pae = np.asarray(pae)
    chains = tokens["chain"].astype(str).to_numpy()
    kinds = tokens["type"].astype(str).to_numpy()
    n = len(chains)
    if pae.ndim != 2 or pae.shape != (n, n):
        raise ValueError(f"PAE matrix shape {pae.shape} does not match {n} tokens")
    if not n:
        raise ValueError("Cannot plot an empty PAE matrix")

    keep = np.ones(n, dtype=bool) if include_ions else kinds != "ion"
    # A pathological ion-only model should remain plottable, not crash batch
    # processing.  There are no protein/ligand tokens to highlight in it.
    if not np.any(keep):
        keep[:] = True
    ions_hidden = bool(np.any(~keep))

    indices = np.flatnonzero(keep)
    chains = chains[indices]
    kinds = kinds[indices]
    selected = pae[np.ix_(indices, indices)]
    widths = np.where(kinds == "lig", ligand_scale, 1).astype(int)

    # np.repeat performs a visual nearest-neighbor stretch.  It preserves
    # directional PAE: rows are aligned tokens, columns are scored tokens.
    total = int(widths.sum())
    if total * total > 60_000_000:
        raise ValueError(
            f"Expanded PAE image would have {total} x {total} pixels; "
            "reduce --pae-ligand-scale"
        )
    displayed = np.repeat(np.repeat(selected, widths, axis=0), widths, axis=1)

    starts = np.r_[0, np.flatnonzero(chains[1:] != chains[:-1]) + 1]
    stops = np.r_[starts[1:], len(chains)]
    offsets = np.r_[0, np.cumsum(widths)]
    blocks = [(str(chains[start]), str(kinds[start]),
               int(offsets[start]), int(offsets[stop]))
              for start, stop in zip(starts, stops)]
    return PlotData(displayed, blocks, ions_hidden)


def save_pae_png(pae, tokens, filename, ligand_scale=1, include_ions=False):
    """Write a directional PAE PNG with molecule boundaries and safe labels."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("PAE images require pip install 'ipsae2[plot]'") from exc

    plot = prepare_pae_plot(pae, tokens, ligand_scale=ligand_scale,
                            include_ions=include_ions)
    values = plot.values
    n = values.shape[0]
    fig, ax = plt.subplots(figsize=(7, 7))
    try:
        im = ax.imshow(values, origin="upper", interpolation="nearest",
                       cmap="viridis", vmin=0, vmax=32, aspect="equal")
        for _, _, _, stop in plot.blocks[:-1]:
            at = stop - 0.5  # cell boundaries, not centers
            ax.axvline(at, color="black", linewidth=0.8)
            ax.axhline(at, color="black", linewidth=0.8)

        # A 7-inch plot is typically ~1000 pixels wide at 200 dpi, but after
        # margins/tick labels its image area is smaller.  Requiring >=2% of
        # the axis per label avoids collisions in long complexes.  The block
        # lines remain visible even if a chain ID is too small to label.
        min_label_width = max(2, int(np.ceil(0.02 * n)))
        labeled = [(chain, (start + stop - 1) / 2)
                   for chain, kind, start, stop in plot.blocks
                   if stop - start >= min_label_width]
        positions = [pos for _, pos in labeled]
        labels = [chain for chain, _ in labeled]
        ax.set_xticks(positions, labels=labels, rotation=90, fontsize=8)
        ax.set_yticks(positions, labels=labels, fontsize=8)
        ax.tick_params(axis="both", which="both", length=0, pad=4)
        ax.xaxis.tick_top()
        ax.set_xlabel("Scored token")
        ax.xaxis.set_label_position("bottom")
        ax.set_ylabel("Aligned token")

        caption = []
        if ligand_scale != 1:
            caption.append(f"ligand width x{ligand_scale}")
        if plot.ions_hidden:
            caption.append("ions omitted")
        if caption:
            ax.set_title("PAE (" + ", ".join(caption) + ")", fontsize=10, pad=25)
        fig.colorbar(im, ax=ax, label="PAE (Å)", fraction=0.046, pad=0.04)
        filename = Path(filename)
        fig.savefig(filename, dpi=200, bbox_inches="tight")
    finally:
        plt.close(fig)
    return filename
