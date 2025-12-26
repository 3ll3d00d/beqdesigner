from __future__ import annotations

import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from enum import Enum, auto
from itertools import groupby
from typing import Iterable

import math
import matplotlib.pyplot as plt
import numpy as np

from beq_loader import BEQFilter, load


# ============================================================
# Utility functions
# ============================================================

def weighted_rms(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    """Perceptually weighted RMS distance."""
    d = x - y
    return np.sqrt(np.sum(w * d * d) / np.sum(w))


# ============================================================
# Rejection diagnostics
# ============================================================

class RejectionReason(Enum):
    RMS_EXCEEDED = auto()
    MAX_EXCEEDED = auto()
    BOTH_EXCEEDED = auto()


@dataclass(frozen=True)
class BEQRejection:
    index: int
    rms_value: float
    max_value: float
    reason: RejectionReason


# ============================================================
# Composite-level data
# ============================================================

@dataclass
class BEQComposite:
    id: int
    median_shape: np.ndarray
    assignments: list[int]

    # Diagnostics
    deltas: list[float]  # RMS deviation per assignment
    max_deltas: list[float]  # Max per-frequency deviation per assignment
    worst_case_curve: np.ndarray
    worst_case_dev: float

    # Fan envelopes: n -> (min, max)
    fan_envelopes: dict[int, tuple[np.ndarray, np.ndarray]]


# ============================================================
# Pipeline result wrapper
# ============================================================

@dataclass
class BEQCompositePipelineResult:
    composites: list[BEQComposite]

    # Global / pipeline config
    freqs: np.ndarray
    band_limits: tuple[float, float]
    fan_counts: list[int]
    rms_limit: float
    max_limit: float

    # Catalogue accounting
    catalogue_size: int
    mapped_count: int
    unmapped_count: int

    # Rejection diagnostics
    rejections: dict[int, BEQRejection]

    # --------------------------------------------------------
    # Diagnostic summary table
    # --------------------------------------------------------
    def diagnostic_table(self) -> str:
        counts = Counter(r.reason for r in self.rejections.values())

        lines = []
        lines.append("BEQ COMPOSITE PIPELINE DIAGNOSTICS")
        lines.append("-" * 44)
        lines.append(f"Catalogue size        : {self.catalogue_size}")
        lines.append(f"Mapped entries        : {self.mapped_count}")
        lines.append(f"Unmapped entries      : {self.unmapped_count}")
        lines.append("")
        lines.append("Rejection breakdown:")
        for reason in RejectionReason:
            lines.append(
                f"  {reason.name:<15}: {counts.get(reason, 0)}"
            )
        lines.append("")
        lines.append("Assignment constraints:")
        lines.append(f"  RMS limit           : {self.rms_limit:.2f} dB")
        lines.append(f"  Max deviation limit : {self.max_limit:.2f} dB")
        lines.append(f"  Band                : {self.band_limits[0]}–{self.band_limits[1]} Hz")

        return "\n".join(lines)


# ============================================================
# Main pipeline
# ============================================================

def build_beq_composites_pipeline(
        responses_db: np.ndarray,
        freqs: np.ndarray,
        weights: np.ndarray,
        band: tuple[float, float],
        k: int,
        rms_limit: float = 5.0,
        max_limit: float = 5.0,
        fan_counts: Iterable[int] = (5, 10, 20),
        rng: np.random.Generator | None = None,
) -> BEQCompositePipelineResult:
    """
    Build BEQ composites from magnitude responses using bounded assignment.
    """

    rng = rng or np.random.default_rng()

    responses_db = np.asarray(responses_db, dtype=float)
    freqs = np.asarray(freqs, dtype=float)
    weights = np.asarray(weights, dtype=float)

    N, n_freqs = responses_db.shape

    # --------------------------------------------------------
    # Band selection
    # --------------------------------------------------------
    f_lo, f_hi = band
    band_mask = (freqs >= f_lo) & (freqs <= f_hi)

    freqs_band = freqs[band_mask]
    weights_band = weights[band_mask]

    # Mean-remove (shape extraction)
    shapes = responses_db[:, band_mask]
    shapes = shapes - shapes.mean(axis=1, keepdims=True)

    # --------------------------------------------------------
    # Initial clustering (simple k-means-like seeding)
    # --------------------------------------------------------
    centers = shapes[rng.choice(N, size=k, replace=False)].copy()

    for _ in range(10):
        dists = np.array([
            [weighted_rms(s, c, weights_band) for c in centers]
            for s in shapes
        ])
        labels = dists.argmin(axis=1)
        for j in range(k):
            if np.any(labels == j):
                centers[j] = np.median(shapes[labels == j], axis=0)

    # --------------------------------------------------------
    # Bounded assignment + rejection capture
    # --------------------------------------------------------
    assignments: list[list[int]] = [[] for _ in range(k)]
    rejections: dict[int, BEQRejection] = {}

    for i in range(N):
        j = labels[i]
        shape = shapes[i]
        center = centers[j]

        rms = weighted_rms(shape, center, weights_band)
        max_dev = np.max(np.abs(shape - center))

        rms_ok = rms <= rms_limit
        max_ok = max_dev <= max_limit

        if rms_ok and max_ok:
            assignments[j].append(i)
        else:
            if not rms_ok and not max_ok:
                reason = RejectionReason.BOTH_EXCEEDED
            elif not rms_ok:
                reason = RejectionReason.RMS_EXCEEDED
            else:
                reason = RejectionReason.MAX_EXCEEDED

            rejections[i] = BEQRejection(
                index=i,
                rms_value=rms,
                max_value=max_dev,
                reason=reason,
            )

    # --------------------------------------------------------
    # Build composites
    # --------------------------------------------------------
    fan_counts = sorted(set(int(n) for n in fan_counts))
    composites: list[BEQComposite] = []

    for cid, idxs in enumerate(assignments):
        if not idxs:
            composites.append(
                BEQComposite(
                    id=cid,
                    median_shape=np.zeros_like(freqs_band),
                    assignments=[],
                    deltas=[],
                    max_deltas=[],
                    worst_case_curve=np.zeros_like(freqs_band),
                    worst_case_dev=0.0,
                    fan_envelopes={},
                )
            )
            continue

        curves = shapes[idxs]
        median = np.median(curves, axis=0)

        deltas = [weighted_rms(shapes[i], median, weights_band) for i in idxs]
        max_deltas = [np.max(np.abs(shapes[i] - median)) for i in idxs]

        worst_i = idxs[int(np.argmax(deltas))]
        worst_curve = shapes[worst_i]
        worst_dev = max(deltas)

        # Fan envelopes based on curve-wise distance ordering
        ordered = sorted(idxs, key=lambda i: weighted_rms(shapes[i], median, weights_band))

        fan_env = {}
        for n in fan_counts:
            if n <= len(ordered):
                sel = shapes[ordered[:n]]
                fan_env[n] = (sel.min(axis=0), sel.max(axis=0))

        composites.append(
            BEQComposite(
                id=cid,
                median_shape=median,
                assignments=idxs,
                deltas=deltas,
                max_deltas=max_deltas,
                worst_case_curve=worst_curve,
                worst_case_dev=worst_dev,
                fan_envelopes=fan_env,
            )
        )

    mapped_count = sum(len(c.assignments) for c in composites)

    # --------------------------------------------------------
    # Return result
    # --------------------------------------------------------
    return BEQCompositePipelineResult(
        composites=composites,
        freqs=freqs_band,
        band_limits=band,
        fan_counts=fan_counts,
        rms_limit=rms_limit,
        max_limit=max_limit,
        catalogue_size=N,
        mapped_count=mapped_count,
        unmapped_count=len(rejections),
        rejections=rejections,
    )


# ============================================================
# 3. Plotting function
# ============================================================

def plot_beq_composites(
        result: BEQCompositePipelineResult,
        ncols: int = 3,
        figsize=(16, 10),
        bins: int = 10
):
    """
    Plot BEQ composites with fan envelopes, worst-case curves,
    and histograms of RMS and max deviations per assigned curve.
    """

    composites = result.composites
    k = len(composites)
    band_freqs = result.freqs[(result.freqs >= result.band_limits[0]) & (result.freqs <= result.band_limits[1])]
    total = result.catalogue_size

    nrows = math.ceil(k / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=True, sharey=False)
    axes = np.atleast_1d(axes).ravel()

    # Fan style
    fan_counts_sorted = sorted(result.fan_counts)
    n_fan = len(fan_counts_sorted)
    alpha_max, alpha_min = 0.35, 0.15
    alphas = [alpha_max] if n_fan == 1 else np.linspace(alpha_max, alpha_min, n_fan)
    fan_styles = {c: {"alpha": a, "label": f"Fan ({c})"} for c, a in zip(fan_counts_sorted, alphas)}

    composite_kw = dict(color="black", lw=2.0, label="Composite")
    worst_kw = dict(color="crimson", lw=1.5, ls="--", label="Worst-case curve")

    legend_handles, legend_labels = [], []

    for i, ax in enumerate(axes):
        if i >= k:
            ax.axis("off")
            continue

        comp: BEQComposite = composites[i]

        # Plot fan envelopes
        for n in result.fan_counts:
            if n in comp.fan_envelopes:
                lo, hi = comp.fan_envelopes[n]
                poly = ax.fill_between(band_freqs, lo, hi, **fan_styles[n])
                if i == 0:
                    legend_handles.append(poly)
                    legend_labels.append(f"Fan ({n})")

        # Plot median shape
        comp_line, = ax.plot(band_freqs, comp.median_shape, **composite_kw)
        if i == 0:
            legend_handles.append(comp_line)
            legend_labels.append("Composite")

        # Plot worst-case curve
        if comp.assignments:
            worst_line, = ax.plot(band_freqs, comp.worst_case_curve, **worst_kw)
            if i == 0:
                legend_handles.append(worst_line)
                legend_labels.append("Worst-case curve")

        # --------------------------------------------------------
        # Subplot title
        # --------------------------------------------------------
        n_assigned = len(comp.assignments)
        pct = n_assigned / total * 100
        ax.set_title(
            f"C{comp.id:02d} | n={n_assigned} ({pct:.1f}%)\n"
            f"worst RMS={comp.worst_case_dev:.2f} dB, "
            f"band {result.band_limits[0]}–{result.band_limits[1]} Hz",
            fontsize=9
        )
        ax.grid(True, alpha=0.3)

        # --------------------------------------------------------
        # Plot RMS and max deviation histograms in inset axes
        # --------------------------------------------------------
        if comp.deltas is not None and comp.max_deltas is not None:
            # [x0, y0, width, height] in axes fraction coordinates = top right
            inset_ax = ax.inset_axes([0.65, 0.55, 0.32, 0.35])
            inset_ax.hist(comp.deltas, bins=bins, alpha=0.6, color='blue', label='RMS', range=(0.0, result.rms_limit), )
            inset_ax.hist(comp.max_deltas, bins=bins, alpha=0.6, color='orange', label='Max')
            # Show x and y axis scales
            inset_ax.tick_params(axis='both', which='major', labelsize=6)
            inset_ax.set_xlabel("dB", fontsize=6)
            inset_ax.set_ylabel("Count", fontsize=6)
            inset_ax.legend(fontsize=6)
            inset_ax.set_xlim(0.0, max(result.rms_limit, result.max_limit))

    fig.supxlabel("Frequency (Hz)")
    fig.supylabel("Magnitude (dB, mean-removed)")

    # --------------------------------------------------------
    # Shared legend
    # --------------------------------------------------------
    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        ncol=len(legend_labels),
        frameon=False,
        bbox_to_anchor=(0.5, 0.98),
    )
    fig.subplots_adjust(top=0.88, hspace=0.35, wspace=0.25)
    plt.show()


import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.stats import gaussian_kde

def plot_rms_vs_max_scatter_with_rejected_density(result: BEQCompositePipelineResult):
    """
    RMS vs max-deviation scatter:
      - Mapped points: black dots
      - Mapped KDE contours: solid black
      - Rejected points: color-coded 'x'
      - Rejected density: semi-transparent heatmap
      - Constraint lines
      - Axes expanded to show all data
    """

    # --------------------------------------------------------
    # Gather mapped data
    # --------------------------------------------------------
    mapped_rms = []
    mapped_max = []
    for comp in result.composites:
        mapped_rms.extend(comp.deltas)
        mapped_max.extend(comp.max_deltas)
    mapped_rms = np.asarray(mapped_rms)
    mapped_max = np.asarray(mapped_max)

    # --------------------------------------------------------
    # Gather rejected data
    # --------------------------------------------------------
    rejected_by_reason = defaultdict(lambda: ([], []))
    rejected_rms = []
    rejected_max = []
    for r in result.rejections.values():
        rejected_by_reason[r.reason][0].append(r.rms_value)
        rejected_by_reason[r.reason][1].append(r.max_value)
        rejected_rms.append(r.rms_value)
        rejected_max.append(r.max_value)
    rejected_rms = np.array(rejected_rms)
    rejected_max = np.array(rejected_max)

    # --------------------------------------------------------
    # Compute axis limits
    # --------------------------------------------------------
    all_rms = np.concatenate([mapped_rms, rejected_rms])
    all_max = np.concatenate([mapped_max, rejected_max])
    x_min, x_max = 0, all_rms.max() * 1.05
    y_min, y_max = 0, all_max.max() * 1.05

    fig, ax = plt.subplots(figsize=(8, 8))

    # --------------------------------------------------------
    # Rejected density heatmap
    # --------------------------------------------------------
    if len(rejected_rms) > 10:
        xy_rejected = np.vstack([rejected_rms, rejected_max])
        kde_rejected = gaussian_kde(xy_rejected)
        xi, yi = np.mgrid[x_min:x_max:300j, y_min:y_max:300j]
        zi_rejected = kde_rejected(np.vstack([xi.ravel(), yi.ravel()])).reshape(xi.shape)
        # Overlay as semi-transparent heatmap
        ax.imshow(
            zi_rejected.T,
            origin='lower',
            extent=[x_min, x_max, y_min, y_max],
            cmap='Reds',
            alpha=0.3,
            aspect='auto',
            zorder=1
        )

    # --------------------------------------------------------
    # KDE contours - mapped population
    # --------------------------------------------------------
    if len(mapped_rms) > 10:
        xy_mapped = np.vstack([mapped_rms, mapped_max])
        kde_mapped = gaussian_kde(xy_mapped)
        xi, yi = np.mgrid[x_min:x_max:300j, y_min:y_max:300j]
        zi_mapped = kde_mapped(np.vstack([xi.ravel(), yi.ravel()])).reshape(xi.shape)
        levels_mapped = np.quantile(zi_mapped.ravel(), [0.5, 0.7, 0.85, 0.93, 0.97, 0.99])
        ax.contour(
            xi, yi, zi_mapped,
            levels=levels_mapped,
            colors='black',
            linewidths=1.2,
            alpha=0.9,
            zorder=2
        )

    # --------------------------------------------------------
    # Scatter - mapped points
    # --------------------------------------------------------
    ax.scatter(
        mapped_rms,
        mapped_max,
        s=15,
        alpha=0.5,
        color='black',
        label='Mapped',
        zorder=3
    )

    # --------------------------------------------------------
    # Scatter - rejected points by reason
    # --------------------------------------------------------
    styles = {
        RejectionReason.RMS_EXCEEDED: dict(color='tab:blue', marker='x', label='Rejected (RMS)'),
        RejectionReason.MAX_EXCEEDED: dict(color='tab:orange', marker='x', label='Rejected (Max)'),
        RejectionReason.BOTH_EXCEEDED: dict(color='tab:red', marker='x', label='Rejected (Both)'),
    }

    for reason, (xr, yr) in rejected_by_reason.items():
        ax.scatter(
            xr, yr,
            s=25,
            alpha=0.85,
            **styles[reason],
            zorder=4
        )

    # --------------------------------------------------------
    # Constraint boundaries
    # --------------------------------------------------------
    ax.axvline(result.rms_limit, color='blue', ls='--', lw=1.2, label='RMS limit', zorder=5)
    ax.axhline(result.max_limit, color='orange', ls='--', lw=1.2, label='Max limit', zorder=5)

    # --------------------------------------------------------
    # Formatting
    # --------------------------------------------------------
    ax.set_xlabel('RMS deviation (dB)')
    ax.set_ylabel('Max per-frequency deviation (dB)')
    ax.set_title('BEQ Assignment Diagnostics: RMS vs Max Deviation')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    ax.legend(frameon=False)
    plt.tight_layout()
    plt.show()


def print_assignments(result: BEQCompositePipelineResult):
    with open('beq_composites.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['composite_id', 'digest', 'content type', 'author', 'title', 'year', 'beqc url'])
        for comp in result.composites:
            for idx in comp.assignments:
                underlying: BEQFilter = filters[idx]
                writer.writerow([
                    comp.id,
                    underlying.entry.digest,
                    underlying.entry.content_type,
                    underlying.entry.author,
                    underlying.entry.formatted_title,
                    underlying.entry.year,
                    underlying.entry.beqc_url
                ])


if __name__ == '__main__':
    catalogue: list[BEQFilter] = load()

    by_author = groupby(catalogue, lambda x: x.entry.author)
    for author, filters_iter in by_author:
        filters = list(filters_iter)
        freqs = filters[0].mag_freqs
        if author != 'aron7awol':
            continue
        responses_db = np.array([f.mag_db - f.mag_db[-1] for f in filters])
        weights = np.ones_like(freqs)

        for i in range(6, 7, 1):
            result: BEQCompositePipelineResult = build_beq_composites_pipeline(
                responses_db=responses_db,
                freqs=freqs,
                weights=weights,
                band=(5, 50),
                k=i,
                rms_limit=3.0,
                max_limit=8.0,
                fan_counts=[5, 10, 20, 50, 100],
            )

            print(result.diagnostic_table())
            plot_beq_composites(result)
            print_assignments(result)
            plot_rms_vs_max_scatter_with_rejected_density(result)