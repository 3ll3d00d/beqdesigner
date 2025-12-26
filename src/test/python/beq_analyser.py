from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import List, Tuple, Dict

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.stats import gaussian_kde


# ------------------------------
# Rejection reasons
# ------------------------------
class RejectionReason(IntEnum):
    RMS_EXCEEDED = 1
    MAX_EXCEEDED = 2
    BOTH_EXCEEDED = 3
    COSINE_TOO_LOW = 4
    DERIVATIVE_TOO_HIGH = 5


# ------------------------------
# Data classes
# ------------------------------
@dataclass
class BEQComposite:
    shape: np.ndarray
    assigned_indices: list[int] = field(default_factory=list)
    deltas: list[float] = field(default_factory=list)
    max_deltas: list[float] = field(default_factory=list)
    derivative_deltas: list[float] = field(default_factory=list)
    cosine_similarities: list[float] = field(default_factory=list)
    worst_rms_index: int | None = None
    worst_max_index: int | None = None
    fan_envelopes: list[np.ndarray] = field(default_factory=list)


@dataclass
class AssignmentRecord:
    entry_index: int
    assigned_composite: int | None = None
    rejected: bool = False
    rejection_reason: RejectionReason | None = None
    rms_value: float | None = None
    max_value: float | None = None
    derivative_value: float | None = None
    cosine_value: float | None = None


@dataclass
class BEQCompositePipelineResult:
    composites: list[BEQComposite]
    assignment_table: list[AssignmentRecord]
    rms_limit: float
    max_limit: float
    cosine_limit: float
    derivative_limit: float


# ------------------------------
# Helper functions
# ------------------------------
def rms(a: np.ndarray, weights: np.ndarray | None = None) -> float:
    if weights is not None:
        return float(np.sqrt(np.mean((a * weights) ** 2)))
    return float(np.sqrt(np.mean(a ** 2)))


def derivative_rms(a: np.ndarray) -> float:
    return rms(np.diff(a))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / norm) if norm != 0 else 1.0


# ------------------------------
# Pure NumPy K-medoids
# ------------------------------
def k_medoids(X: np.ndarray, n_clusters: int, max_iter: int = 100, random_state: int = 0) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    n_samples = X.shape[0]
    medoid_indices = rng.choice(n_samples, size=n_clusters, replace=False)

    for _ in range(max_iter):
        dists = np.linalg.norm(X[:, None, :] - X[medoid_indices][None, :, :], axis=2)
        labels = np.argmin(dists, axis=1)

        new_medoids = np.copy(medoid_indices)
        for i in range(n_clusters):
            cluster_points = X[labels == i]
            if len(cluster_points) == 0:
                continue
            total_dists = np.sum(np.linalg.norm(cluster_points[:, None, :] - cluster_points[None, :, :], axis=2),
                                 axis=1)
            min_idx = np.argmin(total_dists)
            new_medoids[i] = np.where((X == cluster_points[min_idx]).all(axis=1))[0][0]

        if np.all(new_medoids == medoid_indices):
            break
        medoid_indices = new_medoids

    return medoid_indices


# ------------------------------
# Assignment function
# ------------------------------
def assign_to_composites_with_record(
        entry: np.ndarray,
        composites: list[BEQComposite],
        rms_limit: float,
        max_limit: float,
        cosine_limit: float,
        derivative_limit: float,
        entry_idx: int,
        weights: np.ndarray | None = None
) -> AssignmentRecord:
    best_comp: BEQComposite | None = None
    best_comp_idx: int | None = None

    best_rms: float = np.inf
    best_max: float = np.inf
    best_deriv: float = np.inf
    best_cos: float = -1.0

    rejection: RejectionReason | None = None
    rms_val = max_val = deriv_val = cos_val = None

    for comp_idx, comp in enumerate(composites):
        delta = entry - comp.shape
        entry_rms = rms(delta, weights)
        entry_max = float(np.max(np.abs(delta)))
        entry_deriv = derivative_rms(delta)
        entry_cos = cosine_similarity(entry, comp.shape)

        reject: RejectionReason | None = None
        if entry_rms > rms_limit and entry_max > max_limit:
            reject = RejectionReason.BOTH_EXCEEDED
        elif entry_rms > rms_limit:
            reject = RejectionReason.RMS_EXCEEDED
        elif entry_max > max_limit:
            reject = RejectionReason.MAX_EXCEEDED
        elif entry_cos < cosine_limit:
            reject = RejectionReason.COSINE_TOO_LOW
        elif entry_deriv > derivative_limit:
            reject = RejectionReason.DERIVATIVE_TOO_HIGH

        # Track closest composite regardless
        if entry_rms < best_rms:
            best_rms = entry_rms
            best_max = entry_max
            best_deriv = entry_deriv
            best_cos = entry_cos
            best_comp = comp
            best_comp_idx = comp_idx
            rejection = reject
            rms_val, max_val, deriv_val, cos_val = entry_rms, entry_max, entry_deriv, entry_cos

    # --- ASSIGN OR REJECT (but ALWAYS RECORD composite) ---
    if rejection is None and best_comp is not None:
        comp = best_comp
        comp.assigned_indices.append(entry_idx)
        comp.deltas.append(best_rms)
        comp.max_deltas.append(best_max)
        comp.derivative_deltas.append(best_deriv)
        comp.cosine_similarities.append(best_cos)

        if comp.worst_rms_index is None or best_rms > comp.deltas[comp.worst_rms_index]:
            comp.worst_rms_index = len(comp.deltas) - 1
        if comp.worst_max_index is None or best_max > comp.max_deltas[comp.worst_max_index]:
            comp.worst_max_index = len(comp.max_deltas) - 1

        return AssignmentRecord(
            entry_index=entry_idx,
            assigned_composite=best_comp_idx,
            rejected=False,
            rms_value=best_rms,
            max_value=best_max,
            derivative_value=best_deriv,
            cosine_value=best_cos
        )

    # --- REJECTED, BUT COMPOSITE ATTRIBUTED ---
    return AssignmentRecord(
        entry_index=entry_idx,
        assigned_composite=best_comp_idx,  # <<< FIX
        rejected=True,
        rejection_reason=rejection,
        rms_value=rms_val,
        max_value=max_val,
        derivative_value=deriv_val,
        cosine_value=cos_val
    )


# ------------------------------
# Update composites
# ------------------------------
def update_composite_shapes(catalogue: np.ndarray, composites: list[BEQComposite]) -> None:
    for comp in composites:
        if comp.assigned_indices:
            assigned = catalogue[comp.assigned_indices, :]
            comp.shape = np.median(assigned, axis=0)


# ------------------------------
# Compute non-overlapping fan curves
# ------------------------------
def compute_fan_curves(catalogue: np.ndarray, composites: list[BEQComposite],
                       fan_counts: Tuple[int, ...] = (5,)) -> None:
    """
    For each composite, compute non-overlapping fan levels of assigned curves.
    Each fan level contains curves not in previous levels.
    """
    for comp in composites:
        comp.fan_envelopes = []
        if comp.assigned_indices:
            assigned = catalogue[comp.assigned_indices, :]
            # Rank by RMS distance to the composite
            rms_dists = np.array([rms(c - comp.shape) for c in assigned])
            sorted_idx = np.argsort(rms_dists)

            previous = 0
            for n in fan_counts:
                n_curves = min(n, len(sorted_idx))
                if n_curves > previous:
                    comp.fan_envelopes.append(assigned[sorted_idx[previous:n_curves], :])
                    previous = n_curves
                else:
                    # If not enough new curves, append empty array
                    comp.fan_envelopes.append(np.empty((0, assigned.shape[1])))
        else:
            for _ in fan_counts:
                comp.fan_envelopes.append(np.array([comp.shape]))


# ------------------------------
# Full pipeline
# ------------------------------
def build_beq_composites_pipeline(
        responses_db: np.ndarray,
        freqs: np.ndarray,
        weights: np.ndarray | None = None,
        band: Tuple[float, float] = (5, 50),
        k: int = 5,
        rms_limit: float = 5.0,
        max_limit: float = 5.0,
        cosine_limit: float = 0.95,
        derivative_limit: float = 2.0,
        fan_counts: Tuple[int, ...] = (5,),
        n_prototypes: int = 50
) -> BEQCompositePipelineResult:
    band_mask: np.ndarray = (freqs >= band[0]) & (freqs <= band[1])
    catalogue: np.ndarray = responses_db[:, band_mask]
    band_weights: np.ndarray | None = weights[band_mask] if weights is not None else None

    # Step 1: k-medoids prototypes
    if catalogue.shape[0] <= n_prototypes:
        prototypes: np.ndarray = catalogue.copy()
    else:
        medoid_indices: np.ndarray = k_medoids(catalogue, n_prototypes, max_iter=100, random_state=0)
        prototypes = catalogue[medoid_indices]

    # Step 2: Ward clustering
    linkage_matrix: np.ndarray = linkage(prototypes, method='ward')
    labels: np.ndarray = fcluster(linkage_matrix, t=k, criterion='maxclust')

    # Step 3: median per cluster → composites
    composite_shapes: list[np.ndarray] = []
    for i in range(1, k + 1):
        cluster_curves: np.ndarray = prototypes[labels == i]
        median_shape: np.ndarray = np.median(cluster_curves, axis=0)
        composite_shapes.append(median_shape)
    composites: list[BEQComposite] = [BEQComposite(shape=c) for c in composite_shapes]

    # Step 4: assign all entries
    assignment_table: list[AssignmentRecord] = []
    for i, entry in enumerate(catalogue):
        record: AssignmentRecord = assign_to_composites_with_record(entry, composites,
                                                                    rms_limit, max_limit,
                                                                    cosine_limit, derivative_limit,
                                                                    i, weights=band_weights)
        assignment_table.append(record)

    # Step 5: recompute median composite shapes
    update_composite_shapes(catalogue, composites)

    # Step 6: compute fans
    compute_fan_curves(catalogue, composites, fan_counts)

    return BEQCompositePipelineResult(
        composites=composites,
        assignment_table=assignment_table,
        rms_limit=rms_limit,
        max_limit=max_limit,
        cosine_limit=cosine_limit,
        derivative_limit=derivative_limit
    )


# ------------------------------
# Assignment summary
# ------------------------------
def summarize_assignments(result: BEQCompositePipelineResult) -> None:
    print(f"Total catalogue entries: {len(result.assignment_table)}")

    print("\nComposite assignment counts:")
    for i, comp in enumerate(result.composites):
        print(f"  Composite {i + 1}: {len(comp.assigned_indices)} assigned")

    print(f"\nTotal assigned: {sum(len(c.assigned_indices) for c in result.composites)}")
    print(f"Total rejected: {len([i for i in result.assignment_table if i.rejected])}")

    print("\nRejection breakdown:")
    reason_counts: Dict[RejectionReason, int] = {x: len(list(y)) for x, y in groupby(
        sorted([a for a in result.assignment_table if a.rejection_reason is not None],
               key=lambda b: b.rejection_reason), lambda x: x.rejection_reason)}
    for reason, count in dict(sorted(reason_counts.items(), key=lambda item: item[1])).items():
        print(f"  {reason.name}: {count}")


import numpy as np
from matplotlib.axes import Axes
from typing import List


def plot_assigned_fan_curves(catalogue: np.ndarray,
                             result: BEQCompositePipelineResult,
                             freqs: np.ndarray) -> None:
    """Plot assigned fan curves and composite shapes."""
    n_comps = len(result.composites)
    ncols = min(3, n_comps)
    nrows = (n_comps + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3 * nrows),
                             sharex=True, sharey=True)
    axes = np.array(axes).flatten() if n_comps > 1 else np.array([axes])

    for i, comp in enumerate(result.composites):
        ax: Axes = axes[i]

        # Fan curves
        for lvl, fan_curves in enumerate(comp.fan_envelopes):
            if fan_curves.size == 0:
                continue
            alpha = 0.2 + 0.6 * lvl / max(1, len(comp.fan_envelopes) - 1)
            for curve in fan_curves:
                ax.plot(freqs, curve, color='lightblue', lw=1, alpha=alpha, zorder=1)

        # Composite
        ax.plot(freqs, comp.shape, color='black', lw=2, label='Composite', zorder=3)

        ax.set_title(f"Composite {i + 1} ({len(comp.assigned_indices)} assigned)")
        ax.grid(True, alpha=0.3)
        if i % ncols == 0:
            ax.set_ylabel('Magnitude (dB)')
        if i >= ncols * (nrows - 1):
            ax.set_xlabel('Frequency (Hz)')

        # Inset histogram for RMS of assigned curves
        inset = ax.inset_axes([0.65, 0.65, 0.32, 0.32])
        assigned_rms = np.array([rms(catalogue[idx] - comp.shape) for idx in comp.assigned_indices])
        if len(assigned_rms) > 0:
            inset.hist(assigned_rms, bins=15, color='lightblue', alpha=0.7)
        inset.set_title('Assigned RMS', fontsize=8)
        inset.tick_params(axis='both', labelsize=6)

    for j in range(i + 1, nrows * ncols):
        fig.delaxes(axes[j])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', frameon=True)
    fig.suptitle("Assigned Fan Curves", fontsize=14)
    fig.tight_layout(rect=[0, 0, 0.95, 0.95])
    plt.show()


def plot_rejected_by_reason(catalogue: np.ndarray,
                            result: BEQCompositePipelineResult,
                            freqs: np.ndarray) -> None:
    """Plot rejected curves per rejection reason with metric-specific histograms."""
    n_comps = len(result.composites)
    ncols = min(3, n_comps)
    nrows = (n_comps + ncols - 1) // ncols
    reasons = list(RejectionReason)

    for reason in reasons:
        if not any(r for r in result.assignment_table if r.rejected and r.rejection_reason == reason):
            continue

        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3 * nrows),
                                 sharex=True, sharey=True)
        axes = np.array(axes).flatten() if n_comps > 1 else np.array([axes])

        for i, comp in enumerate(result.composites):
            ax: Axes = axes[i]

            # Rejected entries of this reason for this composite
            rejected_entries = [r for r in result.assignment_table
                                if r.rejected and r.assigned_composite == i and r.rejection_reason == reason]
            if not rejected_entries:
                continue

            rejected_indices = [r.entry_index for r in rejected_entries]
            curves = catalogue[rejected_indices]

            # Fan-style plotting
            rms_vals = np.array([rms(c - comp.shape) for c in curves])
            sort_idx = np.argsort(rms_vals)
            for j, idx_r in enumerate(sort_idx):
                alpha = 0.2 + 0.3 * j / max(1, len(sort_idx) - 1)
                ax.plot(freqs, curves[idx_r], color='lightcoral', lw=1, alpha=alpha, zorder=1)

            # Composite overlay
            ax.plot(freqs, comp.shape, color='black', lw=2, zorder=2)

            # Inset histogram for metric that triggered rejection
            inset = ax.inset_axes([0.65, 0.65, 0.32, 0.32])
            if reason == RejectionReason.RMS_EXCEEDED:
                vals = np.array([r.rms_value for r in rejected_entries])
                inset.hist(vals, bins=15, color='lightblue', alpha=0.7)
                inset.set_title('RMS', fontsize=8)
            elif reason == RejectionReason.MAX_EXCEEDED:
                vals = np.array([r.max_value for r in rejected_entries])
                inset.hist(vals, bins=15, color='salmon', alpha=0.7)
                inset.set_title('Max', fontsize=8)
            elif reason == RejectionReason.BOTH_EXCEEDED:
                rms_vals = np.array([r.rms_value for r in rejected_entries])
                max_vals = np.array([r.max_value for r in rejected_entries])
                inset.hist(rms_vals, bins=15, color='lightblue', alpha=0.7, label='RMS')
                inset.hist(max_vals, bins=15, color='salmon', alpha=0.7, label='Max')
                inset.set_title('RMS/Max', fontsize=8)
                inset.legend(fontsize=6)
            elif reason == RejectionReason.COSINE_TOO_LOW:
                vals = np.array([1 - r.cosine_value for r in rejected_entries])
                inset.hist(vals, bins=15, color='violet', alpha=0.7)
                inset.set_title('1 - Cosine', fontsize=8)
            elif reason == RejectionReason.DERIVATIVE_TOO_HIGH:
                vals = np.array([r.derivative_value for r in rejected_entries])
                inset.hist(vals, bins=15, color='orange', alpha=0.7)
                inset.set_title('Derivative', fontsize=8)

            inset.tick_params(axis='both', labelsize=6)
            ax.set_title(f"Composite {i + 1} ({len(rejected_entries)} rejected)")
            ax.grid(True, alpha=0.3)
            if i % ncols == 0:
                ax.set_ylabel('Magnitude (dB)')
            if i >= ncols * (nrows - 1):
                ax.set_xlabel('Frequency (Hz)')

        for j in range(i + 1, nrows * ncols):
            fig.delaxes(axes[j])

        fig.suptitle(f"Rejected Curves by Composite — Reason: {reason.name}", fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.95])

    plt.show()


def plot_all_beq_curves(catalogue: np.ndarray,
                        result: BEQCompositePipelineResult,
                        freqs: np.ndarray) -> None:
    """Convenience function to plot assigned and rejected curves."""
    plot_assigned_fan_curves(catalogue, result, freqs)
    plot_rejected_by_reason(catalogue, result, freqs)


# ------------------------------
# RMS vs Max scatter with density
# ------------------------------
def plot_rms_max_scatter(result: BEQCompositePipelineResult) -> None:
    all_rms: np.ndarray = np.concatenate([np.array(c.deltas) for c in result.composites])
    all_max: np.ndarray = np.concatenate([np.array(c.max_deltas) for c in result.composites])

    xy: np.ndarray = np.vstack([all_rms, all_max])
    kde: np.ndarray = gaussian_kde(xy)(xy)

    plt.figure(figsize=(6, 5))
    plt.scatter(all_rms, all_max, c=kde, s=20, cmap='viridis')
    plt.xlabel('RMS deviation (dB)')
    plt.ylabel('Max deviation (dB)')
    plt.title('RMS vs Max-deviation scatter with density')
    plt.grid(True, alpha=0.3)
    plt.show()


# ------------------------------
# Histograms from assignment table
# ------------------------------
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def plot_histograms_from_table(assignment_table: list[AssignmentRecord]) -> None:
    rms_vals: list[float] = [r.rms_value for r in assignment_table if r.rms_value is not None]
    max_vals: list[float] = [r.max_value for r in assignment_table if r.max_value is not None]
    cosine_vals: list[float] = [r.cosine_value for r in assignment_table if r.cosine_value is not None]

    fig: Figure
    axes: np.ndarray
    fig, axes = plt.subplots(1, 3, figsize=(12, 5))

    # Helper to add percentile lines and annotate them slightly offset
    def add_percentile_lines(ax, data):
        ylim = ax.get_ylim()
        for p in [50, 90, 95]:
            val = np.percentile(data, p)
            ax.axvline(val, color='lightgrey', linestyle='dotted', linewidth=1)
            # Small horizontal offset to avoid overlapping the line
            ax.text(val + 0.01 * (ax.get_xlim()[1] - ax.get_xlim()[0]), ylim[1] * 0.95,
                    f'{p}%', rotation=90, verticalalignment='top',
                    color='grey', fontsize=8)

    # RMS histogram
    axes[0].hist(rms_vals, bins=30, color='skyblue', edgecolor='black')
    axes[0].set_xlabel('RMS deviation (dB)')
    axes[0].set_ylabel('Count')
    axes[0].set_title('RMS deviation')
    add_percentile_lines(axes[0], rms_vals)

    # Max deviation histogram
    axes[1].hist(max_vals, bins=30, color='salmon', edgecolor='black')
    axes[1].set_xlabel('Max deviation (dB)')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Max Deviation')
    add_percentile_lines(axes[1], max_vals)

    # Cosine similarity histogram
    axes[2].hist(cosine_vals, bins=100, color='palegreen', edgecolor='black')
    axes[2].set_xlabel('Cosine')
    axes[2].set_ylabel('Count')
    axes[2].set_title('Cosine Similarity')
    add_percentile_lines(axes[2], cosine_vals)

    plt.tight_layout()
    plt.show()


def print_assignments(result: BEQCompositePipelineResult):
    with open('beq_composites.csv', 'w', newline='') as f:
        import csv
        writer = csv.writer(f)
        writer.writerow(['composite_id', 'digest', 'content type', 'author', 'title', 'year', 'beqc url'])
        for comp_idx, comp in enumerate(result.composites):
            for idx in comp.assigned_indices:
                from beq_loader import BEQFilter
                underlying: BEQFilter = filters[idx]
                writer.writerow([
                    comp_idx,
                    underlying.entry.digest,
                    underlying.entry.content_type,
                    underlying.entry.author,
                    underlying.entry.formatted_title,
                    underlying.entry.year,
                    underlying.entry.beqc_url,

                ])


if __name__ == '__main__':
    min_freq = 5
    max_freq = 50
    fan_counts = (5, 10, 20, 50, 100)

    from beq_loader import BEQFilter, load
    from itertools import groupby

    catalogue: list[BEQFilter] = load()

    by_author = groupby(catalogue, lambda x: x.entry.author)
    for author, filters_iter in by_author:
        filters = list(filters_iter)
        freqs = filters[0].mag_freqs
        if author != 'aron7awol':
            continue
        responses_db = np.array([f.mag_db - f.mag_db[-1] for f in filters])
        weights = np.ones_like(freqs)

        for i in range(7, 8, 1):
            result = build_beq_composites_pipeline(
                responses_db=responses_db,
                freqs=freqs,
                weights=weights,
                band=(min_freq, max_freq),
                k=i,
                rms_limit=10.0,
                max_limit=10.0,
                cosine_limit=0.95,
                derivative_limit=2.0,
                fan_counts=fan_counts,
                n_prototypes=100
            )

            # Diagnostics
            summarize_assignments(result)
            plot_histograms_from_table(result.assignment_table)
            plot_all_beq_curves(responses_db[:, (freqs >= min_freq) & (freqs <= max_freq)],
                                result, freqs[(freqs >= min_freq) & (freqs <= max_freq)])
            plot_rms_max_scatter(result)
            print_assignments(result)
