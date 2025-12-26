import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from enum import Enum
from scipy.spatial.distance import cdist
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.stats import gaussian_kde
from typing import List, Tuple, Dict


# ------------------------------
# Rejection reasons
# ------------------------------
class RejectionReason(Enum):
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
    assigned_indices: List[int] = field(default_factory=list)
    deltas: List[float] = field(default_factory=list)
    max_deltas: List[float] = field(default_factory=list)
    derivative_deltas: List[float] = field(default_factory=list)
    cosine_similarities: List[float] = field(default_factory=list)
    worst_rms_index: int | None = None
    worst_max_index: int | None = None
    fan_envelopes: List[np.ndarray] = field(default_factory=list)


@dataclass
class RejectionRecord:
    reason: RejectionReason
    rms_value: float
    max_value: float
    derivative_value: float
    cosine_value: float


@dataclass
class BEQCompositePipelineResult:
    composites: List[BEQComposite]
    rejections: Dict[int, RejectionRecord]
    rms_limit: float
    max_limit: float
    cosine_limit: float
    derivative_limit: float


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
        composites: List[BEQComposite],
        rms_limit: float,
        max_limit: float,
        cosine_limit: float,
        derivative_limit: float,
        entry_idx: int,
        weights: np.ndarray | None = None
) -> AssignmentRecord:
    best_comp: BEQComposite | None = None
    best_rms: float = np.inf
    best_max: float = np.inf
    best_deriv: float = np.inf
    best_cos: float = -1.0
    rejection: RejectionReason | None = None

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

        if reject is None and entry_rms < best_rms:
            best_comp = comp
            best_comp_idx = comp_idx
            best_rms = entry_rms
            best_max = entry_max
            best_deriv = entry_deriv
            best_cos = entry_cos
            rejection = None
        elif reject is not None and rejection is None:
            rejection = reject
            rms_val, max_val, deriv_val, cos_val = entry_rms, entry_max, entry_deriv, entry_cos

    if best_comp is not None:
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
    else:
        return AssignmentRecord(
            entry_index=entry_idx,
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
def update_composite_shapes(catalogue: np.ndarray, composites: List[BEQComposite]) -> None:
    for comp in composites:
        if comp.assigned_indices:
            assigned = catalogue[comp.assigned_indices, :]
            comp.shape = np.median(assigned, axis=0)


# ------------------------------
# Compute non-overlapping fan curves
# ------------------------------
def compute_fan_curves(catalogue: np.ndarray, composites: List[BEQComposite], fan_counts: Tuple[int, ...] = (5,)) -> None:
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
) -> Tuple[BEQCompositePipelineResult, List[AssignmentRecord]]:
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
    composite_shapes: List[np.ndarray] = []
    for i in range(1, k + 1):
        cluster_curves: np.ndarray = prototypes[labels == i]
        median_shape: np.ndarray = np.median(cluster_curves, axis=0)
        composite_shapes.append(median_shape)
    composites: List[BEQComposite] = [BEQComposite(shape=c) for c in composite_shapes]

    # Step 4: assign all entries
    assignment_table: List[AssignmentRecord] = []
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

    result: BEQCompositePipelineResult = BEQCompositePipelineResult(
        composites=composites,
        rejections={r.entry_index: r for r in assignment_table if r.rejected},
        rms_limit=rms_limit,
        max_limit=max_limit,
        cosine_limit=cosine_limit,
        derivative_limit=derivative_limit
    )

    return result, assignment_table


from matplotlib.axes import Axes
from matplotlib.figure import Figure


# ------------------------------
# Assignment summary
# ------------------------------
def summarize_assignments(result: BEQCompositePipelineResult) -> None:
    total_entries: int = sum(len(comp.assigned_indices) for comp in result.composites) + len(result.rejections)
    print(f"Total catalogue entries: {total_entries}")

    print("\nComposite assignment counts:")
    for i, comp in enumerate(result.composites):
        print(f"  Composite {i + 1}: {len(comp.assigned_indices)} assigned")

    print(f"\nTotal assigned: {sum(len(c.assigned_indices) for c in result.composites)}")
    print(f"Total rejected: {len(result.rejections)}")

    reason_counts: Dict[RejectionReason, int] = {}
    for rej in result.rejections.values():
        reason_counts[rej.rejection_reason] = reason_counts.get(rej.rejection_reason, 0) + 1

    print("\nRejection breakdown:")
    for reason, count in reason_counts.items():
        print(f"  {reason.name}: {count}")


# ------------------------------
# Plot fan chart with unique curves per level
# ------------------------------
from matplotlib.axes import Axes
from matplotlib.figure import Figure

def plot_fan_chart_curves(
        catalogue: np.ndarray,
        result: BEQCompositePipelineResult,
        assignment_table: List[AssignmentRecord],
        freqs: np.ndarray
) -> None:
    n_comps: int = len(result.composites)
    ncols: int = min(3, n_comps)
    nrows: int = (n_comps + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 3*nrows), sharex=True, sharey=True)
    axes = np.array(axes).flatten() if n_comps > 1 else np.array([axes])

    for i, comp in enumerate(result.composites):
        ax: Axes = axes[i]

        # -------------------
        # Plot fan curves
        # -------------------
        n_levels = len(comp.fan_envelopes)
        for idx, fan_curves in enumerate(comp.fan_envelopes):
            if fan_curves.shape[0] == 0:
                continue
            alpha = 0.2 + 0.6 * idx / max(1, n_levels-1)
            for curve in fan_curves:
                ax.plot(freqs, curve, color='lightblue', lw=1, alpha=alpha, zorder=1)

        # -------------------
        # Plot rejected curves
        # -------------------
        rejected_indices = [r.entry_index for r in assignment_table
                            if r.rejected and r.assigned_composite == i]
        if rejected_indices:
            rejected_curves = catalogue[rejected_indices]
            rms_vals = np.array([rms(c - comp.shape) for c in rejected_curves])
            sort_idx = np.argsort(rms_vals)
            cmap = plt.cm.Reds
            n_rejected = len(sort_idx)
            for j, idx_r in enumerate(sort_idx):
                alpha = 0.2 + 0.6 * j / max(1, n_rejected-1)
                ax.plot(freqs, rejected_curves[idx_r], color=cmap(0.6 + 0.4*j/max(1,n_rejected-1)),
                        lw=1.5, alpha=alpha, zorder=2)

        # -------------------
        # Plot composite
        # -------------------
        ax.plot(freqs, comp.shape, color='black', lw=2, label='Composite', zorder=3)

        ax.set_title(f"Composite {i+1} ({len(comp.assigned_indices)} assigned)")
        ax.grid(True, alpha=0.3)
        if i % ncols == 0:
            ax.set_ylabel('Magnitude (dB)')
        if i >= ncols*(nrows-1):
            ax.set_xlabel('Frequency (Hz)')

        # -------------------
        # Inset RMS histogram (assigned vs rejected)
        # -------------------
        inset: Axes = ax.inset_axes([0.65, 0.65, 0.32, 0.32])
        assigned_rms = np.array([rms(c - comp.shape) for idx in comp.assigned_indices
                                 for c in [catalogue[idx]]])
        rejected_rms = np.array([rms(c - comp.shape) for idx in rejected_indices
                                 for c in [catalogue[idx]]])
        if len(assigned_rms) > 0:
            inset.hist(assigned_rms, bins=15, color='lightblue', alpha=0.7, label='Assigned')
        if len(rejected_rms) > 0:
            inset.hist(rejected_rms, bins=15, color='red', alpha=0.7, label='Rejected')
        inset.set_title('RMS dist', fontsize=8)
        inset.tick_params(axis='both', which='major', labelsize=6)
        inset.legend(fontsize=6)

    # Remove unused subplots
    for j in range(i+1, nrows*ncols):
        fig.delaxes(axes[j])

    # Legend once only
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', frameon=True)
    plt.tight_layout()
    plt.show()


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
def plot_histograms_from_table(assignment_table: List[AssignmentRecord]) -> None:
    rms_vals: List[float] = [r.rms_value for r in assignment_table if not r.rejected and r.rms_value is not None]
    max_vals: List[float] = [r.max_value for r in assignment_table if not r.rejected and r.max_value is not None]

    fig: Figure
    axes: np.ndarray
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(rms_vals, bins=30, color='skyblue', edgecolor='black')
    axes[0].set_xlabel('RMS deviation (dB)')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Histogram of RMS deviations')

    axes[1].hist(max_vals, bins=30, color='salmon', edgecolor='black')
    axes[1].set_xlabel('Max deviation (dB)')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Histogram of Max deviations')

    plt.tight_layout()
    plt.show()


def print_assignments(result: BEQCompositePipelineResult):
    with open('beq_composites.csv', 'w', newline='') as f:
        import csv
        writer = csv.writer(f)
        writer.writerow(['composite_id', 'digest', 'content type', 'author', 'title', 'year', 'beqc url'])
        for comp in result.composites:
            for idx in comp.assigned_indices:
                from beq_loader import BEQFilter
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
            result, assignment_table = build_beq_composites_pipeline(
                responses_db=responses_db,
                freqs=freqs,
                weights=weights,
                band=(min_freq, max_freq),
                k=i,
                rms_limit=10.0,
                max_limit=10.0,
                cosine_limit=0.95,
                derivative_limit=1.0,
                fan_counts=fan_counts,
                n_prototypes=100
            )

            # Diagnostics
            summarize_assignments(result)
            plot_histograms_from_table(assignment_table)
            plot_fan_chart_curves(responses_db[:, (freqs >= min_freq) & (freqs <= max_freq)], result,
                                  assignment_table,
                                  freqs[(freqs >= min_freq) & (freqs <= max_freq)])
            plot_rms_max_scatter(result)
