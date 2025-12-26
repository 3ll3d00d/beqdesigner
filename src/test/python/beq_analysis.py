import collections
import json
import time
from concurrent.futures import ProcessPoolExecutor
from itertools import groupby

from scipy.signal import unit_impulse

from model.catalogue import CatalogueEntry, load_catalogue
from model.iir import CompleteFilter
from model.signal import Signal

min_freq = 8.0
max_freq = 60.0
fs = 1000
max_band_deviation = 5.0

from dataclasses import dataclass
import numpy as np

@dataclass
class BEQComposite:
    # Inputs / configuration
    freqs: np.ndarray
    band_limits: tuple[float, float]
    shape_rms_limit: float
    max_dev_limit: float
    weights: np.ndarray
    fan_counts: list[int]

    # Catalogue statistics
    catalogue_size: int
    mapped_count: int
    unmapped_count: int

    # Core results (indexed by composite id)
    composites: list[np.ndarray]                 # median shapes
    assignments: list[list[int]]                 # indices of catalogue entries
    worst_case_curves: list[np.ndarray]          # per composite
    worst_case_devs: list[float]

    # Fan envelopes: [composite][fan_count] -> (lo, hi)
    fans: list[dict[int, tuple[np.ndarray, np.ndarray]]]

    # Optional diagnostics
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


def convert(entry: CatalogueEntry) -> dict | None:
    u_i = unit_impulse(fs * 4, 'mid') * 23453.66
    f = CompleteFilter(fs=fs, filters=entry.iir_filters(fs=fs), description=f'{entry}')
    signal = Signal('test', u_i, fs=fs)
    try:
        f_signal = signal.sosfilter(f.get_sos(), filtfilt=False)
        x, y = f_signal.mag_response
        return {'x': x, 'y': y, 'm': entry}
    except Exception as e:
        print(f'Unable to process entry {entry.title}')
        return None


class CatalogueEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, CatalogueEntry):
            return obj.for_search
        return super().default(obj)


class CatalogueDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)

    @staticmethod
    def object_hook(dct):
        if 'x' in dct and 'y' in dct and 'm' in dct:
            return {
                'x': np.array(dct['x']),
                'y': np.array(dct['y']),
                'm': CatalogueEntry('0', dct['m']),
            }
        return dct


import numpy as np

def derive_fan_styles(fan_counts):
    """
    Returns dict:
        n -> style kwargs for fill_between
    """
    fan_counts = sorted(fan_counts)
    n = len(fan_counts)

    # Opacity range: inner strongest, outer weakest
    alpha_max = 0.35
    alpha_min = 0.15

    if n == 1:
        alphas = [alpha_max]
    else:
        alphas = np.linspace(alpha_max, alpha_min, n)

    styles = {}
    for count, alpha in zip(fan_counts, alphas):
        styles[count] = {
            "alpha": alpha,
            "label": f"Fan ({count})",
        }

    return styles

# ============================================================
# 0. Simple diagnostic logger
# ============================================================

def log(msg):
    print(f"[BEQ] {msg}")


# ============================================================
# 1. Shape / strength separation
# ============================================================

def split_shape_strength(responses_db: np.ndarray):
    strengths = responses_db.mean(axis=1)
    shapes = responses_db - strengths[:, None]
    return shapes, strengths


# ============================================================
# 2. Distance metrics
# ============================================================

def weighted_rms(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> float:
    diff = a - b
    return np.sqrt(np.sum(w * diff**2) / np.sum(w))


def max_abs_dev(a: np.ndarray, b: np.ndarray) -> float:
    return np.max(np.abs(a - b))


def within_bounds(
        curve: np.ndarray,
        composite: np.ndarray,
        weights: np.ndarray,
        rms_limit: float,
        max_limit: float,
):
    rms = weighted_rms(curve, composite, weights)
    max_dev = max_abs_dev(curve, composite)
    return (rms <= rms_limit) and (max_dev <= max_limit), rms, max_dev


# ============================================================
# 3. Composite computation
# ============================================================

def median_composite(shapes: np.ndarray) -> np.ndarray:
    return np.median(shapes, axis=0)


# ============================================================
# 4. Bounded assignment
# ============================================================

def assign_curves(
        shapes: np.ndarray,
        composites: list,
        weights: np.ndarray,
        rms_limit: float,
        max_limit: float,
):
    K = len(composites)
    N = len(shapes)

    assignments = [[] for _ in range(K)]
    distances = np.full(N, np.inf)
    max_devs = np.zeros(N)
    unmapped = []

    for i, curve in enumerate(shapes):
        best_k = None
        best_rms = np.inf
        best_max = 0.0

        for k, comp in enumerate(composites):
            ok, rms, max_dev = within_bounds(
                curve, comp, weights, rms_limit, max_limit
            )
            if ok and rms < best_rms:
                best_rms = rms
                best_k = k
                best_max = max_dev

        if best_k is None:
            unmapped.append(i)
        else:
            assignments[best_k].append(i)
            distances[i] = best_rms
            max_devs[i] = best_max

    return assignments, distances, max_devs, unmapped


# ============================================================
# 5. Iterative bounded clustering
# ============================================================

def bounded_shape_clustering(
        shapes: np.ndarray,
        weights: np.ndarray,
        k: int,
        rms_limit: float,
        max_limit: float,
        max_iters: int = 20,
        random_state: int = 0,
):
    rng = np.random.default_rng(random_state)

    log(f"Clustering {len(shapes)} curves into k={k}")
    log(f"Bounds: RMS ≤ {rms_limit:.2f} dB, Max |Δ| ≤ {max_limit:.2f} dB")

    init_idx = rng.choice(len(shapes), size=k, replace=False)
    composites = [shapes[i].copy() for i in init_idx]

    for it in range(max_iters):
        assignments, distances, max_devs, unmapped = assign_curves(
            shapes, composites, weights, rms_limit, max_limit
        )

        mapped = len(shapes) - len(unmapped)
        log(
            f"Iter {it+1:02d}: mapped={mapped} "
            f"({mapped/len(shapes)*100:.1f}%), "
            f"unmapped={len(unmapped)}"
        )

        new_composites = []
        for idxs, old in zip(assignments, composites):
            if idxs:
                new_composites.append(median_composite(shapes[idxs]))
            else:
                new_composites.append(old)

        max_delta = max(
            np.max(np.abs(n - o))
            for n, o in zip(new_composites, composites)
        )

        composites = new_composites
        if max_delta < 1e-3:
            log(f"Converged after {it+1} iterations (Δmax={max_delta:.4f} dB)")
            break

    return composites, assignments, distances, max_devs, unmapped


# ============================================================
# 6. Fan chart envelopes (real curves, ordered)
# ============================================================

def fan_envelopes(
        shapes: np.ndarray,
        assigned_indices: list,
        composite: np.ndarray,
        weights: np.ndarray,
        fan_counts=(5, 10, 20, 50, 100, 200, 500),
):
    if not assigned_indices:
        return {}

    dists = [
        (i, weighted_rms(shapes[i], composite, weights))
        for i in assigned_indices
    ]
    dists.sort(key=lambda x: x[1])

    envelopes = {}
    for n in fan_counts:
        sel = [i for i, _ in dists[:n]]
        curves = shapes[sel]
        envelopes[n] = (
            curves.min(axis=0),
            curves.max(axis=0),
        )

    return envelopes


# ============================================================
# 7. FULL PIPELINE ENTRY POINT (FULL DIAGNOSTICS)
# ============================================================

def build_beq_composites_pipeline(
        responses_db: np.ndarray,
        freqs: np.ndarray,
        weights: np.ndarray,
        band=(5.0, 50.0),
        k: int = 8,
        rms_limit: float = 5.0,
        max_limit: float = 5.0,
        random_state: int = 0,
):
    log(f"Catalogue size: {responses_db.shape[0]} responses")
    log(f"Infra-bass band: {band[0]}–{band[1]} Hz")

    mask = (freqs >= band[0]) & (freqs <= band[1])
    log(f"Band bins: {mask.sum()}")

    responses_band = responses_db[:, mask]
    weights_band = weights[mask]

    shapes, strengths = split_shape_strength(responses_band)

    composites, assignments, distances, max_devs, unmapped = (
        bounded_shape_clustering(
            shapes,
            weights_band,
            k=k,
            rms_limit=rms_limit,
            max_limit=max_limit,
            random_state=random_state,
        )
    )

    total = len(shapes)
    mapped = total - len(unmapped)

    log("Final mapping summary:")
    log(f"  Total     : {total}")
    log(f"  Mapped    : {mapped} ({mapped/total*100:.1f}%)")
    log(f"  Unmapped  : {len(unmapped)} ({len(unmapped)/total*100:.1f}%)")

    # --------------------------------------------------------
    # Per-composite diagnostics
    # --------------------------------------------------------

    log("Per-composite diagnostics:")
    global_worst_rms = 0.0
    global_worst_max = 0.0

    for i, idxs in enumerate(assignments):
        if not idxs:
            log(f"  C{i:02d}: EMPTY")
            continue

        rms_vals = distances[idxs]
        max_vals = max_devs[idxs]

        worst_rms = rms_vals.max()
        worst_max = max_vals.max()

        global_worst_rms = max(global_worst_rms, worst_rms)
        global_worst_max = max(global_worst_max, worst_max)

        log(
            f"  C{i:02d}: "
            f"n={len(idxs):4d} "
            f"({len(idxs)/total*100:5.1f}%) | "
            f"worst RMS={worst_rms:5.2f} dB, "
            f"worst |Δ|={worst_max:5.2f} dB"
        )

    log("Global worst-case deviations:")
    log(f"  RMS max  : {global_worst_rms:.2f} dB")
    log(f"  |Δ| max  : {global_worst_max:.2f} dB")

    # --------------------------------------------------------
    # Single-line batch summary
    # --------------------------------------------------------

    log(
        "SUMMARY | "
        f"N={total} "
        f"k={k} "
        f"band={band[0]}–{band[1]}Hz "
        f"RMS≤{rms_limit} "
        f"|Δ|≤{max_limit} "
        f"mapped={mapped} "
        f"({mapped/total*100:.1f}%) "
        f"worstRMS={global_worst_rms:.2f} "
        f"worstΔ={global_worst_max:.2f}"
    )

    fans = [
        fan_envelopes(
            shapes,
            assignments[i],
            composites[i],
            weights_band,
        )
        for i in range(len(composites))
    ]

    return {
        "composites": composites,
        "assignments": assignments,
        "unmapped": unmapped,
        "distances": distances,
        "max_devs": max_devs,
        "fans": fans,
        "strengths": strengths,
        "band_mask": mask,
    }

import matplotlib.pyplot as plt
import math
import numpy as np

def plot_composites_with_fans(
        freqs: np.ndarray,
        composites: list,
        assignments: list,
        fans: list,
        distances: np.ndarray,
        max_devs: np.ndarray,
        band_mask: np.ndarray,
        shapes: np.ndarray,
        fan_counts=(5, 10, 20, 50, 100, 200, 500),
        ncols=3,
        figsize=(14, 8),
):
    band_freqs = freqs[band_mask]
    k = len(composites)
    total = sum(len(a) for a in assignments)

    nrows = math.ceil(k / ncols)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=figsize,
        sharex=True,
        sharey=True,
    )
    axes = np.atleast_1d(axes).ravel()

    # --------------------------------------------------------
    # Styles
    # --------------------------------------------------------
    composite_kw = dict(color="black", lw=2.0, label="Composite")
    worst_kw = dict(
        color="crimson",
        lw=1.5,
        ls="--",
        label="Worst-case curve",
    )

    fan_styles = derive_fan_styles(fan_counts)

    legend_handles = []
    legend_labels = []

    # --------------------------------------------------------
    # Per-composite subplots
    # --------------------------------------------------------
    for i, ax in enumerate(axes):
        if i >= k:
            ax.axis("off")
            continue

        comp = composites[i]
        idxs = assignments[i]

        # Fan envelopes
        for n in fan_counts:
            if n not in fans[i]:
                continue
            lo, hi = fans[i][n]
            poly = ax.fill_between(
                band_freqs,
                lo,
                hi,
                **fan_styles[n]
            )
            if i == 0:
                legend_handles.append(poly)
                legend_labels.append(f"Fan ({n})")

        # Composite
        comp_line, = ax.plot(band_freqs, comp, **composite_kw)
        if i == 0:
            legend_handles.append(comp_line)
            legend_labels.append("Composite")

        # Worst-case curve overlay
        if idxs:
            worst_idx = max(idxs, key=lambda j: distances[j])
            worst_curve = shapes[worst_idx]

            worst_line, = ax.plot(
                band_freqs,
                worst_curve,
                **worst_kw
            )
            if i == 0:
                legend_handles.append(worst_line)
                legend_labels.append("Worst-case curve")

            worst_rms = distances[worst_idx]
            worst_max = max_devs[worst_idx]
            pct = len(idxs) / total * 100

            title = (
                f"C{i:02d} | n={len(idxs)} ({pct:.1f}%)\n"
                f"worst RMS={worst_rms:.2f} dB, "
                f"worst |Δ|={worst_max:.2f} dB"
            )
        else:
            title = f"C{i:02d} | EMPTY"

        ax.set_title(title, fontsize=9)
        ax.grid(True, alpha=0.3)

    # --------------------------------------------------------
    # Shared labels
    # --------------------------------------------------------
    fig.supxlabel("Frequency (Hz)")
    fig.supylabel("Magnitude (dB, mean-removed)")

    # --------------------------------------------------------
    # Single shared legend (outside grid, unclipped)
    # --------------------------------------------------------
    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        ncol=len(legend_labels),
        frameon=False,
        bbox_to_anchor=(0.5, 0.98),
    )

    legend_rows = math.ceil(len(legend_labels) / 4)
    top = 0.88 - 0.04 * (legend_rows - 1)

    fig.subplots_adjust(top=top, hspace=0.35, wspace=0.25,)

    plt.show()


if __name__ == '__main__':
    a = time.time()

    try:
        with open('database.bin', 'r') as f:
            data: list[dict] = json.load(f, cls=CatalogueDecoder)['data']
    except Exception as e:
        entries: list[CatalogueEntry] = load_catalogue('/home/matt/.beq/database.json')
        with ProcessPoolExecutor() as executor:
            data: list[dict] = list(executor.map(convert, [e for e in entries if e.filters]))
        with open('database.bin', 'w') as f:
            json.dump({'data': data}, f, cls=CatalogueEncoder)

    b = time.time()
    print(f'Loaded catalogue in {b - a:.3g}s')

    freqs = data[0]['x']

    data_by_author = groupby(data, lambda x: x['m'].author)
    for author, data in data_by_author:
        if author != 'aron7awol':
            continue
        responses_db = np.array([f['y'] - f['y'][-1] for f in data])
        for i in range(6, 7, 1):
            result = build_beq_composites_pipeline(
                responses_db,
                freqs,
                np.ones_like(freqs),
                band=(5, 50),
                k=8,
                rms_limit=4.0,
                max_limit=10.0,
            )
            plot_composites_with_fans(
                freqs=freqs,
                composites=result["composites"],
                assignments=result["assignments"],
                fans=result["fans"],
                distances=result["distances"],
                max_devs=result["max_devs"],
                band_mask=result["band_mask"],
                shapes=(
                        responses_db[:, result["band_mask"]]
                        - responses_db[:, result["band_mask"]].mean(axis=1, keepdims=True)
                ),
            )

    pass
