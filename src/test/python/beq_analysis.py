import json
import time
from concurrent.futures import ProcessPoolExecutor
from itertools import groupby

from scipy.cluster.hierarchy import linkage, fcluster
import collections

from model.catalogue import CatalogueEntry, load_catalogue
from scipy.signal import unit_impulse

from model.iir import CompleteFilter
from model.signal import Signal

min_freq = 8.0
max_freq = 60.0
fs = 1000
max_band_deviation = 5.0


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


# -----------------------------
# Helper: automatic outlier threshold
# -----------------------------
def auto_outlier_threshold(rms_values, lambda_mad=2.5):
    med = np.median(rms_values)
    mad = np.median(np.abs(rms_values - med))
    return med + lambda_mad * mad


# -----------------------------
# Helper: extract infra descriptors
# -----------------------------
def extract_infra_descriptors_v2(freq, mag_db):
    f = freq
    m = mag_db

    bands = {
        "ultra": (f >= 5) & (f < 12),
        "low": (f >= 12) & (f < 20),
        "mid": (f >= 20) & (f < 35),
        "high": (f >= 35) & (f <= 50),
    }

    band_means = {k: np.mean(m[v]) for k, v in bands.items()}

    peak_db = np.max(m)
    peak_freq = f[np.argmax(m)]

    slope = np.polyfit(f, m, 1)[0] * (f[-1] - f[0])

    edge_mean = 0.5 * (band_means["ultra"] + band_means["high"])
    center_mean = band_means["low"] + band_means["mid"]
    curvature = center_mean - edge_mean

    return {
        "peak_db": peak_db,
        "peak_freq": peak_freq,
        "slope": slope,
        "curvature": curvature,
        **band_means,
    }


# -----------------------------
# Helper: percentile-based semantic labelling
# -----------------------------
def label_infra_semantics_v2(desc, stats):
    labels = []

    # Strength
    p = stats["peak_db"]
    if desc["peak_db"] >= p["p80"]:
        labels.append("very-strong")
    elif desc["peak_db"] >= p["p60"]:
        labels.append("strong")
    elif desc["peak_db"] >= p["p40"]:
        labels.append("moderate")
    elif desc["peak_db"] >= p["p20"]:
        labels.append("weak")
    else:
        labels.append("very-weak")

    # Depth balance
    ulf_delta = desc["ultra"] - desc["mid"]
    ref = stats["ultra"]["values"] - stats["mid"]["values"]
    if ulf_delta >= np.percentile(ref, 80):
        labels.append("ULF-heavy")
    elif ulf_delta >= np.percentile(ref, 60):
        labels.append("mid-ULF")
    elif ulf_delta >= np.percentile(ref, 40):
        labels.append("balanced-infra")
    elif ulf_delta >= np.percentile(ref, 20):
        labels.append("mid-upper")
    else:
        labels.append("upper-infra-heavy")

    # Shape
    c = stats["curvature"]
    if desc["curvature"] >= c["p80"]:
        labels.append("hump-shaped")
    elif desc["curvature"] >= c["p60"]:
        labels.append("moderately-hump")
    elif desc["curvature"] >= c["p40"]:
        labels.append("smooth-shelf")
    elif desc["curvature"] >= c["p20"]:
        labels.append("moderately-shelf")
    else:
        labels.append("shelf-shaped")

    # Aggressiveness
    s = abs(desc["slope"])
    sp = stats["slope"]
    if s >= sp["p80"]:
        labels.append("steep-rise")
    elif s >= sp["p60"]:
        labels.append("moderate-steep")
    elif s >= sp["p40"]:
        labels.append("moderate-rise")
    elif s >= sp["p20"]:
        labels.append("gentle-rise")
    else:
        labels.append("gradual-rise")

    # Peak region
    if desc["peak_freq"] < 10:
        labels.append("sub-10Hz")
    elif desc["peak_freq"] < 20:
        labels.append("10–20Hz")
    else:
        labels.append("20Hz+")

    return labels


# -----------------------------
# Build semantic reference from any set of responses
# -----------------------------
def build_semantic_reference(freq_sel, responses_db):
    descs = [extract_infra_descriptors_v2(freq_sel, r) for r in responses_db]
    keys = descs[0].keys()
    stats = {}
    for k in keys:
        values = np.array([d[k] for d in descs])
        stats[k] = {
            "values": values,
            "p20": np.percentile(values, 20),
            "p40": np.percentile(values, 40),
            "p60": np.percentile(values, 60),
            "p80": np.percentile(values, 80),
        }
    return descs, stats


# -----------------------------
# Label composites
# -----------------------------
def label_composites(freq_sel, composites_db, full_catalogue_stats):
    composite_descs = [extract_infra_descriptors_v2(freq_sel, c) for c in composites_db]
    composite_labels = [label_infra_semantics_v2(d, full_catalogue_stats) for d in composite_descs]
    return composite_labels


# -----------------------------
# Label catalogue entries
# -----------------------------
def label_catalogue_entries(freq_sel, responses_db, full_catalogue_stats):
    catalogue_labels = []
    label_counts = collections.Counter()
    for i in range(responses_db.shape[0]):
        mag_db = responses_db[i]
        desc = extract_infra_descriptors_v2(freq_sel, mag_db)
        sem = label_infra_semantics_v2(desc, full_catalogue_stats)
        catalogue_labels.append(sem)
        label_counts.update(sem)
    return catalogue_labels, dict(label_counts)


# -----------------------------
# Helper: plot CDFs
# -----------------------------
def plot_rms_cdf_core_vs_outlier(rms_core, rms_outlier, author):
    plt.figure(figsize=(6.8, 4.8))

    def plot_group(rms, label, color):
        rms = np.asarray(rms)
        if rms.size == 0:
            return
        rms_sorted = np.sort(rms)
        cdf = np.arange(1, len(rms_sorted) + 1) / len(rms_sorted)
        plt.plot(rms_sorted, cdf, label=label, linewidth=2)

        for i in [50, 90, 95]:
            p = np.percentile(rms_sorted, i)
            plt.axvline(p, linestyle=":", color=color, label=f"P{i} - {p:.2f} dB")

    plot_group(rms_core, "Core", "C0")
    plot_group(rms_outlier, "Outlier", "C1")
    plt.grid(True, alpha=0.3)
    plt.xlabel("RMS reconstruction error (dB)")
    plt.ylabel("Cumulative fraction")
    plt.title(f"Infra-band RMS reconstruction error (CDF) - {author}")
    plt.legend()
    plt.ylim(0, 1.02)
    xmax = max(np.max(rms_core) if len(rms_core) > 0 else 0,
               np.max(rms_outlier) if len(rms_outlier) > 0 else 0)
    plt.xlim(0, xmax * 1.05)
    plt.tight_layout()
    plt.show()


# -----------------------------
# Main function: build composites
# -----------------------------
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster


def build_beq_composites_pipeline(
        responses_db,
        freq_hz,
        author=None,
        infra_low_hz=10.0,
        infra_high_hz=40.0,
        k_core=8,
        k_outlier=4,
        alpha=1.0,
        pca_var=0.98,
        outlier_rms_thresh=None,
        max_assign_rms_db=4.0,
        max_band_dev_db=max_band_deviation,
):
    """
    Build BEQ composite curves with perceptually-weighted infra-band clustering
    and bandwise-safe assignment.

    Unmapped entries are labelled -1.
    """

    UNMAPPED = -1
    N, F = responses_db.shape

    # -------------------------------------------------
    # 1) Infra-band selection
    # -------------------------------------------------
    band_mask = (freq_hz >= infra_low_hz) & (freq_hz <= infra_high_hz)
    freq_sel = freq_hz[band_mask]
    resp = responses_db[:, band_mask]

    # -------------------------------------------------
    # 2) Shape / strength separation
    # -------------------------------------------------
    strength = resp.mean(axis=1, keepdims=True)
    shape = resp - strength

    # -------------------------------------------------
    # 3) Perceptual weighting (low-frequency emphasis)
    # -------------------------------------------------
    w = (infra_high_hz / freq_sel) ** alpha
    gate = 1.0 / (1.0 + np.exp(-(freq_sel - infra_low_hz * 1.2) / 2.0))
    w *= gate
    w /= np.sqrt(np.mean(w ** 2))

    shape_w = shape * w

    # -------------------------------------------------
    # 4) PCA compression
    # -------------------------------------------------
    X = shape_w - shape_w.mean(axis=0)
    U, S, _ = np.linalg.svd(X, full_matrices=False)
    var = S ** 2
    n_pca = np.searchsorted(np.cumsum(var) / var.sum(), pca_var) + 1
    Xp = U[:, :n_pca] * S[:n_pca]

    # -------------------------------------------------
    # 5) Core clustering (hierarchical / Ward)
    # -------------------------------------------------
    Z = linkage(Xp, method="ward")
    labels_core = fcluster(Z, k_core, criterion="maxclust") - 1

    # -------------------------------------------------
    # 6) Automatic outlier threshold (RMS within clusters)
    # -------------------------------------------------
    if outlier_rms_thresh is None:
        rms_all = []
        for k in range(k_core):
            idx = np.where(labels_core == k)[0]
            if len(idx) < 2:
                continue
            ref = np.median(shape_w[idx], axis=0)
            rms_all.append(
                np.sqrt(np.mean((shape_w[idx] - ref) ** 2, axis=1))
            )
        rms_all = np.concatenate(rms_all)
        med = np.median(rms_all)
        mad = np.median(np.abs(rms_all - med)) + 1e-9
        outlier_rms_thresh = med + 2.5 * mad

    # -------------------------------------------------
    # 7) Build core composites + collect outliers
    # -------------------------------------------------
    labels = np.full(N, UNMAPPED, dtype=int)
    composites = []
    outlier_idx = []

    for k in range(k_core):
        idx = np.where(labels_core == k)[0]
        if len(idx) == 0:
            composites.append(np.zeros(resp.shape[1]))
            continue

        ref = np.median(shape_w[idx], axis=0)
        rms = np.sqrt(np.mean((shape_w[idx] - ref) ** 2, axis=1))

        core = idx[rms <= outlier_rms_thresh]
        out = idx[rms > outlier_rms_thresh]
        outlier_idx.extend(out.tolist())

        if len(core) == 0:
            composites.append(np.zeros(resp.shape[1]))
            continue

        comp = (
                np.median(shape[core], axis=0)
                + np.median(strength[core])
        )
        composites.append(comp)
        labels[core] = k

    # -------------------------------------------------
    # 8) Reclustering outliers into dedicated composites
    # -------------------------------------------------
    if len(outlier_idx) and k_outlier > 0:
        outlier_idx = np.array(outlier_idx)
        Zo = linkage(Xp[outlier_idx], method="ward")
        k_eff = min(k_outlier, len(outlier_idx))
        out_labels = fcluster(Zo, k_eff, criterion="maxclust") - 1

        for j in range(k_eff):
            idx = outlier_idx[out_labels == j]
            comp = (
                    np.median(shape[idx], axis=0)
                    + np.median(strength[idx])
            )
            composites.append(comp)
            labels[idx] = k_core + j

    composites_db = np.vstack(composites)
    n_comp = composites_db.shape[0]

    # -------------------------------------------------
    # 9) BANDWISE-SAFE ASSIGNMENT
    # -------------------------------------------------
    # Precompute composite shapes
    comp_shape = np.zeros((n_comp, shape.shape[1]))
    for k in range(n_comp):
        members = np.where(labels == k)[0]
        if len(members):
            comp_shape[k] = np.median(shape[members], axis=0)

    comp_shape_w = comp_shape * w

    for i in range(N):
        k = labels[i]
        if k == UNMAPPED:
            continue

        delta_shape = shape[i] - comp_shape[k]
        delta_shape_w = shape_w[i] - comp_shape_w[k]

        rms = np.sqrt(np.mean(delta_shape_w ** 2))
        band_max = np.max(np.abs(delta_shape))

        if (rms > max_assign_rms_db) or (band_max > max_band_dev_db):
            labels[i] = UNMAPPED

    # -------------------------------------------------
    # 10) Reporting
    # -------------------------------------------------
    n_unmapped = np.sum(labels == UNMAPPED)
    print(f"Catalogue size:        {N}")
    print(f"Mapped entries:        {N - n_unmapped} ({100 * (N - n_unmapped) / N:.1f}%)")
    print(f"Unmapped entries:      {n_unmapped} ({100 * n_unmapped / N:.1f}%)")
    print(f"RMS limit:             {max_assign_rms_db:.2f} dB")
    print(f"Bandwise limit:        {max_band_dev_db:.2f} dB")

    return {
        "freq_sel": freq_sel,
        "composites_db": composites_db,
        "labels": labels,
        "unmapped_fraction": n_unmapped / N,
        "limits": {
            "rms_db": max_assign_rms_db,
            "band_db": max_band_dev_db,
        },
    }


def plot_composite_responses(freq_sel, composites_db, k_core, author):
    """
    Plot frequency response of each composite, separating core and outlier composites.

    Parameters
    ----------
    freq_sel : array_like
        Frequencies of infra-band (Hz)
    composites_db : array_like, shape (n_composites, n_freq)
        Composite magnitude responses in dB
    k_core : int
        Number of core composites (first k_core rows in composites_db)
    """
    n_composites = composites_db.shape[0]
    plt.figure(figsize=(8, 5))

    # Core composites
    for i in range(min(k_core, n_composites)):
        plt.semilogx(freq_sel, composites_db[i], label=f"Core {i}", color=f"C{i % 10}", linewidth=2)

    # Outlier composites
    for i in range(k_core, n_composites):
        plt.semilogx(freq_sel, composites_db[i], label=f"Outlier {i - k_core}", color=f"C{i % 10}", linewidth=1.5,
                     linestyle='--')

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.title(f"BEQ Composites - {author}")
    plt.grid(True, which="both", alpha=0.3)
    plt.ylim()
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.show()


import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt


def plot_composite_responses(
        freq_hz,
        responses_db,
        labels,
        composites_db,
        composite_idx,
        max_band_dev_db=None,
        percentiles=(0.50, 0.75, 0.90, 1.00),
        ax=None,
):
    """
    Plot a perceptually honest fan chart for a single composite.

    Envelopes are constructed from REAL curves, ordered by
    curve-wise maximum absolute deviation from the composite.

    Parameters
    ----------
    freq_hz : ndarray (F,)
        Frequency axis (Hz)
    responses_db : ndarray (N, F)
        Full catalogue responses (already band-limited if desired)
    labels : ndarray (N,)
        Composite assignment labels (-1 for unmapped)
    composites_db : ndarray (K, F)
        Composite responses
    composite_idx : int
        Which composite to plot
    max_band_dev_db : float or None
        Optional visual annotation of bandwise limit
    percentiles : tuple of floats
        Fractions of curves to include in fan layers (0–1)
    ax : matplotlib axis or None
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))

    # -------------------------------------------------
    # 1) Extract assigned responses
    # -------------------------------------------------
    idx = np.where(labels == composite_idx)[0]
    if len(idx) == 0:
        ax.set_title(f"Composite {composite_idx} (no members)")
        return ax

    curves = responses_db[idx]
    comp = composites_db[composite_idx]

    # -------------------------------------------------
    # 2) Curve-wise distance (bandwise max deviation)
    # -------------------------------------------------
    delta = curves - comp
    d = np.max(np.abs(delta), axis=1)

    order = np.argsort(d)
    curves = curves[order]
    d = d[order]

    # -------------------------------------------------
    # 3) Fan layers (REAL envelopes)
    # -------------------------------------------------
    base_color = np.array([0.2, 0.4, 0.8])  # blue
    n = len(curves)

    for i, p in enumerate(percentiles):
        k = max(1, int(np.ceil(p * n)))
        band = curves[:k]

        low = band.min(axis=0)
        high = band.max(axis=0)

        # darker = tighter core
        alpha = 0.15 + 0.6 * (1 - i / max(1, len(percentiles) - 1))

        ax.fill_between(
            freq_hz,
            low,
            high,
            color=base_color,
            alpha=alpha,
            linewidth=0,
            label=f"{int(p * 100)}% curves" if i == 0 else None,
        )

    # -------------------------------------------------
    # 4) Plot composite and median member
    # -------------------------------------------------
    ax.plot(freq_hz, comp, color="black", lw=2, label="Composite")

    median_curve = curves[len(curves) // 2]
    ax.plot(
        freq_hz,
        median_curve,
        color="black",
        lw=1,
        ls="--",
        alpha=0.8,
        label="Median member",
    )

    # -------------------------------------------------
    # 5) Optional bandwise limit annotation
    # -------------------------------------------------
    if max_band_dev_db is not None:
        ax.plot(
            freq_hz,
            comp + max_band_dev_db,
            color="red",
            lw=1,
            ls=":",
            alpha=0.6,
            label="Band limit",
        )
        ax.plot(
            freq_hz,
            comp - max_band_dev_db,
            color="red",
            lw=1,
            ls=":",
            alpha=0.6,
        )

    # -------------------------------------------------
    # 6) Cosmetics
    # -------------------------------------------------
    ax.set_xscale("log")
    ax.set_xlim(freq_hz[0], freq_hz[-1])
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dB)")
    ax.set_title(
        f"Composite {composite_idx} — {len(idx)} members\n"
        f"Max deviation range: {d.min():.2f}–{d.max():.2f} dB"
    )
    ax.grid(True, which="both", ls=":", lw=0.5)
    ax.legend(loc="best")

    return ax


def plot_composite_responses_on_grid(results, responses_db, freq_hz, author):
    n_comp = results['composites_db'].shape[0]
    cols = 3
    rows = int(np.ceil(n_comp / cols))

    band_mask = (freq_hz >= min_freq) & (freq_hz <= max_freq)

    responses_band = responses_db[:, band_mask]
    freq_band = freq_hz[band_mask]

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), sharex=True, sharey=True)
    axes = axes.flat

    legend_handles = None
    legend_labels = None

    for k in range(n_comp):
        ax = axes[k]
        plot_composite_responses(
            freq_hz=freq_band,
            responses_db=responses_band,
            labels=results['labels'],
            composites_db=results['composites_db'],
            composite_idx=k,
            max_band_dev_db=max_band_deviation,
            ax=ax,
        )
        if legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()
        # Remove per-axis legends
        ax.legend_.remove() if ax.legend_ else None

    for ax in axes[n_comp:]:
        ax.axis("off")

    # -------------------------------------------------
    # Single shared legend
    # -------------------------------------------------
    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower right",
        ncol=len(legend_labels),
        frameon=False,
        # bbox_to_anchor=(0.5, 1.02),
    )

    plt.tight_layout()
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

    freq_hz = data[0]['x']

    data_by_author = groupby(data, lambda x: x['m'].author)
    for author, data in data_by_author:
        if author != 'aron7awol':
            continue
        responses_db = np.array([f['y'] - f['y'][-1] for f in data])
        for i in range(6, 7, 1):
            results = build_beq_composites_pipeline(
                responses_db=responses_db,
                freq_hz=freq_hz,
                author=author,
                infra_low_hz=min_freq,
                infra_high_hz=max_freq,
                k_core=i,
                k_outlier=2,
                alpha=0.95,
                pca_var=0.98,
                outlier_rms_thresh=None
            )

            plot_composite_responses_on_grid(results, responses_db, freq_hz, author)

    pass
