import json
import time
from concurrent.futures import ProcessPoolExecutor
from itertools import groupby

import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster
import collections

from model.catalogue import CatalogueEntry, load_catalogue
from scipy.signal import unit_impulse

from model.iir import CompleteFilter
from model.signal import Signal

min_freq = 8.0
max_freq = 60.0
fs = 1000


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
def build_beq_composites_pipeline(
        responses_db,
        freq_hz,
        author=None,
        infra_low_hz=5.0,
        infra_high_hz=50.0,
        k_core=8,
        k_outlier=4,
        alpha=1.0,
        pca_var=0.98,
        outlier_rms_thresh=None
):
    """
    Integrated BEQ pipeline:
    - Infra-band selection
    - Shape/strength separation
    - PCA
    - Hierarchical clustering for core
    - Auto outlier threshold
    - Outlier reclustering
    - RMS calculation
    - CDF plotting
    - Semantic labeling
    - Catalogue mapping
    """
    # -----------------------------
    # 1) Infra-band
    mask = (freq_hz >= infra_low_hz) & (freq_hz <= infra_high_hz)
    freq_sel = freq_hz[mask]
    resp = responses_db[:, mask]  # (N,M)

    # -----------------------------
    # 2) Shape / strength
    strength = resp.mean(axis=1, keepdims=True)
    shape = resp - strength

    # -----------------------------
    # 3) Perceptual weighting
    w = (infra_high_hz / freq_sel) ** alpha
    low_gate = 1.0 / (1.0 + np.exp(-(freq_sel - infra_low_hz * 1.2) / 2.0))
    w *= low_gate
    w /= np.sqrt(np.mean(w ** 2))
    shape_w = shape * w

    # -----------------------------
    # 4) PCA
    X = shape_w - shape_w.mean(axis=0)
    U, S, _ = np.linalg.svd(X, full_matrices=False)
    var = S ** 2
    n_pca = np.searchsorted(np.cumsum(var) / var.sum(), pca_var) + 1
    Xp = U[:, :n_pca] * S[:n_pca]

    # -----------------------------
    # 5) Core clustering
    Z = linkage(Xp, method="ward")
    labels_core = fcluster(Z, k_core, criterion="maxclust") - 1

    # -----------------------------
    # 6) Auto global outlier threshold
    if outlier_rms_thresh is None:
        rms_all = []
        for k in range(k_core):
            idx = np.where(labels_core == k)[0]
            if len(idx) == 0:
                continue
            shape_k = np.median(shape_w[idx], axis=0)
            rms_k = np.sqrt(np.mean((shape_w[idx] - shape_k) ** 2, axis=1))
            rms_all.append(rms_k)
        rms_all = np.concatenate(rms_all)
        auto_thresh = auto_outlier_threshold(rms_all, lambda_mad=2.5)
    else:
        auto_thresh = outlier_rms_thresh

    # -----------------------------
    # 7) Core composite + outlier separation
    N, M = shape.shape
    labels = np.full(N, -1, dtype=int)
    composites_core = []
    core_members = []
    outlier_indices = []

    for k in range(k_core):
        idx = np.where(labels_core == k)[0]
        if len(idx) == 0:
            composites_core.append(np.zeros(M))
            core_members.append([])
            continue

        shape_k = np.median(shape_w[idx], axis=0)
        rms = np.sqrt(np.mean((shape_w[idx] - shape_k) ** 2, axis=1))
        is_outlier = rms > auto_thresh

        core_idx = idx[~is_outlier]
        out_idx = idx[is_outlier]
        outlier_indices.extend(out_idx.tolist())

        if len(core_idx) == 0:
            composites_core.append(np.zeros(M))
            core_members.append([])
            continue

        shape_k = np.median(shape[core_idx], axis=0)
        strength_k = np.median(strength[core_idx])
        composites_core.append(shape_k + strength_k)
        core_members.append(core_idx.tolist())
        labels[core_idx] = k

    # -----------------------------
    # 8) Outlier reclustering
    outlier_indices = np.array(outlier_indices, dtype=int)
    composites_out = []

    if len(outlier_indices) > 0 and k_outlier > 0:
        Xo = Xp[outlier_indices]
        k_eff = min(k_outlier, len(outlier_indices))
        Zo = linkage(Xo, method="ward")
        labels_out = fcluster(Zo, k_eff, criterion="maxclust") - 1
        for j in range(k_eff):
            idx = outlier_indices[labels_out == j]
            shape_j = np.median(shape[idx], axis=0)
            strength_j = np.median(strength[idx])
            composites_out.append(shape_j + strength_j)
            labels[idx] = k_core + j

    # -----------------------------
    # 9) Final composite array
    composites_db = np.vstack([composites_core, composites_out])

    # -----------------------------
    # 10) RMS calculation
    def rms_for(indices):
        return np.array([
            np.sqrt(np.mean((shape_w[i] - (composites_db[labels[i]] - strength[i]) * w) ** 2))
            for i in indices
        ])

    core_idx = np.where(labels < k_core)[0]
    out_idx = np.where(labels >= k_core)[0]
    rms_core = rms_for(core_idx)
    rms_outlier = rms_for(out_idx)

    # -----------------------------
    # 11) Core vs outlier fraction
    n_total = len(labels)
    pct_outlier = 100 * len(out_idx) / n_total
    pct_core = 100 * len(core_idx) / n_total

    print(f"Author: {author}")
    print('--------------')
    print(f"Catalogue size: {n_total}")
    print(f"Core curves:    {len(core_idx)} ({pct_core:.1f}%)")
    print(f"Outlier curves: {len(out_idx)} ({pct_outlier:.1f}%)")
    print(f"Auto RMS outlier threshold: {auto_thresh:.2f} dB")
    print('')

    # -----------------------------
    # 12) Plot CDF
    plot_rms_cdf_core_vs_outlier(rms_core, rms_outlier, author)
    #

    # -----------------------
    # Semantic labelling
    # -----------------------
    # Build percentile reference from full catalogue (not just composites)
    _, full_catalogue_stats = build_semantic_reference(freq_sel, resp)

    # Composite-level semantics (archetypes)
    composite_labels = label_composites(freq_sel, composites_db, full_catalogue_stats)
    n_composites = len(composite_labels)
    comp_counts = collections.Counter([tag for lbl in composite_labels for tag in lbl])
    print("\nComposite-level semantic distribution (archetypes):")
    for k, v in sorted(comp_counts.items(), key=lambda x: -x[1]):
        print(f"{k:18s}: {v:5d} ({100 * v / n_composites:.1f}%)")

    # Entry-level semantics (true catalogue distribution)
    catalogue_semantics, entry_counts = label_catalogue_entries(freq_sel, resp, full_catalogue_stats)
    print("\nCatalogue-level semantic distribution:")
    for k, v in sorted(entry_counts.items(), key=lambda x: -x[1]):
        print(f"{k:18s}: {v:5d} ({100 * v / n_total:.1f}%)")

    # -----------------------------
    # 14) Plot composite frequency responses
    # -----------------------------
    plot_composite_responses(freq_sel, composites_db, k_core, author)

    # Composite-level summary
    print("\nComposite semantic distribution:")
    n_composites = len(composite_labels)
    comp_counts = collections.Counter([tag for lbl in composite_labels for tag in lbl])
    for k, v in sorted(comp_counts.items(), key=lambda x: -x[1]):
        print(f"{k:18s}: {v:5d} ({100 * v / n_composites:.1f}%)")

    # Entry-level summary
    print("\nCatalogue semantic distribution:")
    n_entries = responses_db.shape[0]
    entry_counts = collections.Counter([tag for lbl in catalogue_semantics for tag in lbl])
    for k, v in sorted(entry_counts.items(), key=lambda x: -x[1]):
        print(f"{k:18s}: {v:5d} ({100 * v / n_entries:.1f}%)")

    return {
        "freq_sel": freq_sel,
        "composites_db": composites_db,
        "labels": labels,
        "rms_core": rms_core,
        "rms_outlier": rms_outlier,
        "composite_labels": composite_labels,
        "catalogue_semantics": catalogue_semantics,
        "entry_counts": entry_counts,
        "pct_outlier": pct_outlier
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


def plot_composite_responses_fan(freq_sel, composites_db, responses_db, labels,
                                 lower_percentiles=[50, 25, 20, 5, 0],
                                 upper_percentiles=[50, 75, 90, 95, 100]):
    """
    Plot each composite as a fan chart with asymmetric shading to handle skewed distributions.

    Lower percentiles are shaded from median downwards; upper percentiles are shaded from median upwards.

    Parameters
    ----------
    freq_sel : 1D np.array
        Frequencies of the responses (infra-band selected).
    composites_db : 2D np.array
        Composite curves (num_composites x num_freqs).
    responses_db : 2D np.array
        Original catalogue responses (num_entries x num_freqs).
    labels : 1D np.array of ints
        Mapping of each catalogue entry to a composite index.
    lower_percentiles : list
        Percentiles below median to shade (ascending, last should be median).
    upper_percentiles : list
        Percentiles above median to shade (ascending, first should be median).
    """
    n_composites = composites_db.shape[0]
    cmap_lower = plt.get_cmap("Blues")
    cmap_upper = plt.get_cmap("Reds")


    for k in range(n_composites):
        plt.figure(figsize=(10, 6))
        idx = np.where(labels == k)[0]
        if len(idx) == 0:
            continue

        entries = responses_db[idx]
        # --- Upper fan ---
        for i in range(len(upper_percentiles) - 1):
            low = np.percentile(entries, upper_percentiles[i], axis=0)
            high = np.percentile(entries, upper_percentiles[i + 1], axis=0)
            alpha = 0.1 + 0.05 * i
            plt.fill_between(freq_sel, low, high, color=cmap_upper(0.6), alpha=alpha, zorder=1)

        # --- Lower fan ---
        for i in range(len(lower_percentiles) - 1):
            low = np.percentile(entries, lower_percentiles[i], axis=0)
            high = np.percentile(entries, lower_percentiles[i + 1], axis=0)
            alpha = 0.1 + 0.05 * i
            plt.fill_between(freq_sel, low, high, color=cmap_lower(0.6), alpha=alpha, zorder=1)

        # Median
        median_curve = np.percentile(entries, 50, axis=0)
        plt.plot(freq_sel, median_curve, color="C0", linewidth=2, label=f"Composite {k}" if k == 0 else None, zorder=2)

        # Nominal composite
        plt.plot(freq_sel, composites_db[k], color="C1", linestyle="--", linewidth=1.8,
                 label=f"Composite nominal" if k == 0 else None, zorder=3)

        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Magnitude [dB]")
        plt.title("Composite BEQ Curves vs Underlying")
        plt.grid(True, alpha=0.3)
        plt.legend()
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
        for i in range(32, 33, 1):
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
            plot_composite_responses_fan(
                freq_sel=results["freq_sel"],
                composites_db=results["composites_db"],
                responses_db=responses_db[:, (freq_hz >= min_freq) & (freq_hz <= max_freq)],
                labels=results["labels"]
            )

    pass
