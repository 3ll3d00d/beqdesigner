from __future__ import annotations

import csv
from dataclasses import dataclass
from itertools import groupby

import numpy as np
import matplotlib.pyplot as plt
import math
from typing import Optional, Tuple

from beq_loader import BEQFilter, load

# ============================================================
# 1. Dataclasses
# ============================================================

from enum import Enum, auto


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


@dataclass
class BEQComposite:
    id: int
    median_shape: np.ndarray
    assignments: list[int]
    worst_case_curve: np.ndarray
    worst_case_dev: float
    fan_envelopes: dict[int, tuple[np.ndarray, np.ndarray]]
    deltas: list[float] | None = None  # RMS deviation per assigned curve
    max_deltas: list[float] | None = None  # Max per-frequency deviation per assigned curve


@dataclass
class BEQCompositePipelineResult:
    freqs: np.ndarray
    band_limits: tuple[float, float]
    shape_rms_limit: float
    max_dev_limit: float
    weights: np.ndarray
    fan_counts: list[int]

    catalogue_size: int
    mapped_count: int
    unmapped_count: int

    rms_limit: float
    max_limit: float

    composites: list[BEQComposite]
    rejections: dict[int, BEQRejection]
    distances: list[np.ndarray] | None = None

    def summary(self) -> str:
        pct_mapped = self.mapped_count / self.catalogue_size * 100
        pct_unmapped = self.unmapped_count / self.catalogue_size * 100
        return (
            f"Catalogue: {self.catalogue_size}, "
            f"Mapped: {self.mapped_count} ({pct_mapped:.1f}%), "
            f"Unmapped: {self.unmapped_count} ({pct_unmapped:.1f}%), "
            f"k={len(self.composites)}, "
            f"Band={self.band_limits[0]}–{self.band_limits[1]} Hz, "
            f"RMS≤{self.shape_rms_limit}, |Δ|≤{self.max_dev_limit}"
        )


# ============================================================
# 2. Pipeline function
# ============================================================

def build_beq_composites_pipeline(
        responses_db: np.ndarray,
        freqs: np.ndarray,
        weights: np.ndarray,
        band: tuple[float, float] = (5.0, 50.0),
        k: int = 8,
        rms_limit: float = 5.0,
        max_limit: float = 5.0,
        fan_counts: list[int] = (5, 10, 20, 50),
        max_iters: int = 20,
        random_state: int = 0,
        diagnostics: bool = True,
) -> BEQCompositePipelineResult:
    rng = np.random.default_rng(random_state)

    band_lo, band_hi = band
    mask = (freqs >= band_lo) & (freqs <= band_hi)
    responses_band = responses_db[:, mask]
    weights_band = weights[mask]
    band_freqs = freqs[mask]

    N_curves, N_freqs = responses_band.shape
    print(f"[BEQ] Catalogue size: {N_curves} responses")
    print(f"[BEQ] Band: {band_lo}-{band_hi} Hz, bins={N_freqs}")
    print(f"[BEQ] Clustering into k={k} composites with RMS≤{rms_limit}, maxΔ≤{max_limit}")

    # --------------------------------------------------------
    # Shape / strength separation
    # --------------------------------------------------------
    strengths = responses_band.mean(axis=1)
    shapes = responses_band - strengths[:, None]

    # --------------------------------------------------------
    # Weighted RMS & max deviation helpers
    # --------------------------------------------------------
    def weighted_rms(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> float:
        diff = a - b
        return np.sqrt(np.sum(w * diff ** 2) / np.sum(w))

    def max_abs_dev(a: np.ndarray, b: np.ndarray) -> float:
        return np.max(np.abs(a - b))

    def within_bounds(curve: np.ndarray, comp: np.ndarray) -> tuple[bool, float, float]:
        rms = weighted_rms(curve, comp, weights_band)
        max_dev = max_abs_dev(curve, comp)
        return (rms <= rms_limit and max_dev <= max_limit), rms, max_dev

    # --------------------------------------------------------
    # Initialize composites
    # --------------------------------------------------------
    init_idx = rng.choice(N_curves, size=k, replace=False)
    composites = [shapes[i].copy() for i in init_idx]

    # --------------------------------------------------------
    # Iterative assignment and update
    # --------------------------------------------------------
    for it in range(max_iters):
        assignments = [[] for _ in range(k)]
        distances = np.full(N_curves, np.inf)
        max_devs = np.zeros(N_curves)
        unmapped = []

        for i, curve in enumerate(shapes):
            best_k = None
            best_rms = np.inf
            best_max = 0.0
            for j, comp in enumerate(composites):
                ok, rms, max_dev = within_bounds(curve, comp)
                if ok and rms < best_rms:
                    best_rms = rms
                    best_k = j
                    best_max = max_dev
            if best_k is None:
                unmapped.append(i)
            else:
                assignments[best_k].append(i)
                distances[i] = best_rms
                max_devs[i] = best_max

        mapped = N_curves - len(unmapped)
        print(f"[BEQ] Iter {it + 1:02d}: mapped={mapped} ({mapped / N_curves * 100:.1f}%), unmapped={len(unmapped)}")

        # update composites
        new_composites = []
        for idxs, old in zip(assignments, composites):
            if idxs:
                new_composites.append(np.median(shapes[idxs], axis=0))
            else:
                new_composites.append(old)
        delta_max = max(np.max(np.abs(n - o)) for n, o in zip(new_composites, composites))
        composites = new_composites
        if delta_max < 1e-3:
            print(f"[BEQ] Converged after {it + 1} iterations (Δmax={delta_max:.4f} dB)")
            break

    # --------------------------------------------------------
    # Build per-composite objects
    # --------------------------------------------------------
    composite_objects = []

    for cid, idxs in enumerate(assignments):
        if idxs:
            comp_shape = np.median(shapes[idxs], axis=0)
            worst_idx = max(idxs, key=lambda j: distances[j])
            worst_curve = shapes[worst_idx]
            worst_dev = distances[worst_idx]

            # Compute fan envelopes
            comp_fans = {}
            dists = [(i, weighted_rms(shapes[i], comp_shape, weights_band)) for i in idxs]
            dists.sort(key=lambda x: x[1])
            for n in fan_counts:
                if n > len(dists):
                    continue
                sel = [i for i, _ in dists[:n]]
                curves = shapes[sel]
                comp_fans[n] = (curves.min(axis=0), curves.max(axis=0))

            # Compute per-assignment RMS and max deviations
            deltas = [weighted_rms(shapes[i], comp_shape, weights_band) for i in idxs]
            max_deltas = [np.max(np.abs(shapes[i] - comp_shape)) for i in idxs]

        else:
            comp_shape = np.zeros(N_freqs)
            worst_curve = np.zeros(N_freqs)
            worst_dev = 0.0
            comp_fans = {}
            deltas = None
            max_deltas = None

        composite_objects.append(
            BEQComposite(
                id=cid,
                median_shape=comp_shape,
                assignments=idxs,
                worst_case_curve=worst_curve,
                worst_case_dev=worst_dev,
                fan_envelopes=comp_fans,
                deltas=deltas,
                max_deltas=max_deltas
            )
        )

    # --------------------------------------------------------
    # Build pipeline result
    # --------------------------------------------------------
    result = BEQCompositePipelineResult(
        freqs=freqs,
        band_limits=(band_lo, band_hi),
        shape_rms_limit=rms_limit,
        max_dev_limit=max_limit,
        weights=weights,
        fan_counts=fan_counts,
        catalogue_size=N_curves,
        mapped_count=mapped,
        unmapped_count=len(unmapped),
        composites=composite_objects,
        distances=[distances] if diagnostics else None,
        rms_limit=rms_limit,
        max_limit=max_limit,
    )

    return result


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
                max_limit=10.0,
                fan_counts=[5, 10, 20, 50, 100],
                diagnostics=True
            )

            print(result.summary())
            plot_beq_composites(result)
            print_assignments(result)
