#!/usr/bin/env python3
"""
SignalPath-Finder feature extraction from chunked expression profiles.

Workflow
--------
1. Load a gene-expression matrix where rows are genes and columns are samples.
2. For each anchor gene (TOR1, LST8, RAPTOR2), determine the number of chunks
   from a target chunk size.
3. Reorder samples by anchor-gene expression and optimize chunk composition to
   improve within-chunk normality.
4. For every gene and chunk, compute four pattern signals relative to the anchor:
      - N: normality-like behavior
      - T: t-distribution-like behavior
      - S: similarity to the anchor-gene distribution
      - P: peak-position/width similarity after bell-shape reordering
5. Export:
      a. reordered chunk/sample assignments
      b. per-gene chunk-wise feature labels and statistics

Intended use
------------
This script is written for reproducible research use in GitHub repositories and
manuscript methods. Update the configuration block in `main()` or adapt it to
command-line arguments for your environment.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.ndimage import gaussian_filter1d
from scipy.stats import kstest, ks_2samp, kurtosis, shapiro, skew, t


def exclude_and_assign_above_threshold(
    series: pd.Series,
    threshold: float,
) -> Tuple[pd.Series, pd.Series]:
    """Split a numeric series into values below/equal to and above a threshold."""
    series_numeric = pd.to_numeric(series, errors="coerce")
    below_threshold = series_numeric[series_numeric <= threshold]
    above_threshold = series_numeric[series_numeric > threshold]
    return below_threshold, above_threshold



def sort_and_divide_series_into_n_chunks(
    series: pd.Series,
    total_chunks: int,
) -> List[pd.Series]:
    """Sort a series in descending order and divide it into balanced chunks."""
    sorted_series = series.sort_values(ascending=False)
    n = len(sorted_series)
    chunk_size = max(n // total_chunks, 1)
    extra = n % total_chunks

    chunks: List[pd.Series] = []
    start_index = 0
    for i in range(total_chunks):
        end_index = start_index + chunk_size + (1 if i < extra else 0)
        chunks.append(sorted_series.iloc[start_index:end_index])
        start_index = end_index

    return chunks



def analyze_chunk_normality(chunk: pd.DataFrame | pd.Series) -> Dict[str, float | bool]:
    """Evaluate whether a chunk approximates a normal distribution."""
    values = np.asarray(chunk).flatten()
    shapiro_stat, shapiro_p = shapiro(values)
    skewness = skew(values)
    kurt = kurtosis(values) + 3
    is_normal = (shapiro_p > 0.05) and (abs(skewness) < 0.5) and (2.5 < kurt < 3.5)

    return {
        "shapiro_stat": shapiro_stat,
        "shapiro_p_value": shapiro_p,
        "skewness": skewness,
        "kurtosis": kurt,
        "is_normal": is_normal,
    }



def optimize_chunks_for_normality_pandas(
    series_list: Sequence[pd.Series],
    num_new_chunks: int,
    seed: int,
    random_state: int,
    n_iterations: int = 10_000,
) -> Tuple[List[pd.DataFrame], List[Dict[str, float | bool]]]:
    """Shuffle and swap elements across chunks to maximize average Shapiro p-value."""
    combined_series = pd.concat(series_list)
    shuffled_series = combined_series.sample(frac=1, random_state=random_state)

    total_elements = len(shuffled_series)
    chunk_sizes = [
        total_elements // num_new_chunks + (1 if i < total_elements % num_new_chunks else 0)
        for i in range(num_new_chunks)
    ]

    starts = np.cumsum([0] + chunk_sizes[:-1])
    new_chunks = [
        shuffled_series.iloc[start:start + size].reset_index()
        for start, size in zip(starts, chunk_sizes)
    ]

    np.random.seed(seed)
    best_scores: List[Dict[str, float | bool]] = []
    best_mean_score = -np.inf

    for _ in range(n_iterations):
        i, j = np.random.randint(0, len(new_chunks), size=2)
        if len(new_chunks[i]) <= 3 or len(new_chunks[j]) <= 3:
            continue

        a = np.random.randint(0, len(new_chunks[i]))
        b = np.random.randint(0, len(new_chunks[j]))

        row_i = new_chunks[i].iloc[a].copy()
        row_j = new_chunks[j].iloc[b].copy()
        new_chunks[i].iloc[a] = row_j
        new_chunks[j].iloc[b] = row_i

        current_scores = [
            analyze_chunk_normality(chunk.drop(columns=chunk.columns[0]))
            for chunk in new_chunks
            if len(chunk) > 3
        ]
        current_mean_score = np.mean([score["shapiro_p_value"] for score in current_scores])

        if current_mean_score > best_mean_score:
            best_mean_score = current_mean_score
            best_scores = current_scores
        else:
            new_chunks[i].iloc[a] = row_i
            new_chunks[j].iloc[b] = row_j

    optimized_chunks = [chunk.set_index(chunk.columns[0]) for chunk in new_chunks]
    return optimized_chunks, best_scores



def smooth_signal(
    signal: np.ndarray | pd.Series,
    method: str = "none",
    smoothing_param: float = 2,
) -> np.ndarray | pd.Series:
    """Smooth a signal using Gaussian or LOWESS smoothing."""
    if method == "gaussian":
        return gaussian_filter1d(signal, sigma=smoothing_param)
    if method == "lowess":
        lowess = sm.nonparametric.lowess(signal, np.arange(len(signal)), frac=smoothing_param)
        return lowess[:, 1]
    return signal



def visualize_chunks_and_peaks_single_plot(
    chunk1: np.ndarray | pd.Series,
    chunk2: np.ndarray | pd.Series,
    smoothing: str = "none",
    smoothing_param: float = 2,
) -> None:
    """Plot two chunk profiles in a single figure for visual inspection."""
    chunk1_smoothed = smooth_signal(chunk1, method=smoothing, smoothing_param=smoothing_param)
    chunk2_smoothed = smooth_signal(chunk2, method=smoothing, smoothing_param=smoothing_param)

    plt.figure(figsize=(10, 6))
    plt.plot(chunk1_smoothed, label="Chunk 1", marker="o")
    plt.plot(chunk2_smoothed, label="Chunk 2", marker="o")
    plt.title(f"Chunk profiles with detected peaks ({smoothing.capitalize()} smoothing)")
    plt.xlabel("Index")
    plt.ylabel("Expression level")
    plt.legend()
    plt.tight_layout()
    plt.show()



def z_transform(chunk: np.ndarray | pd.Series) -> np.ndarray:
    """Standardize a chunk using the Z-transform."""
    return (chunk - np.mean(chunk)) / np.std(chunk)



def reorder_bell_shape(chunk: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    """Reorder values into a bell-shaped layout while retaining original indices."""
    original_indices = chunk.index.values
    sorted_values_with_indices = sorted(zip(chunk, original_indices), reverse=True)
    sorted_values, sorted_indices = zip(*sorted_values_with_indices)

    n = len(sorted_values)
    bell_shaped_order = np.zeros(n)
    bell_shaped_indices = np.zeros(n, dtype=sorted_indices[0].dtype if hasattr(sorted_indices[0], 'dtype') else object)

    center = n // 2
    left, right = center - 1, center + 1
    bell_shaped_order[center] = sorted_values[0]
    bell_shaped_indices[center] = sorted_indices[0]

    for i in range(1, n):
        if i % 2 == 1:
            bell_shaped_order[left] = sorted_values[i]
            bell_shaped_indices[left] = sorted_indices[i]
            left -= 1
        else:
            bell_shaped_order[right] = sorted_values[i]
            bell_shaped_indices[right] = sorted_indices[i]
            right += 1

    return bell_shaped_order, bell_shaped_indices



def apply_reordering_based_on_chunk1(
    chunk1: pd.Series,
    chunk2: pd.Series,
) -> Tuple[np.ndarray, np.ndarray]:
    """Bell-shape reorder chunk1 and apply the same ordering to chunk2."""
    reordered_chunk1, reordered_indices = reorder_bell_shape(chunk1)
    reordered_chunk2 = chunk2.loc[reordered_indices].values
    return reordered_chunk1, reordered_chunk2



def detect_peaks_from_derivative(
    signal: np.ndarray,
    first_derivative: np.ndarray,
) -> List[int]:
    """Detect local maxima from derivative sign changes."""
    peaks = []
    for i in range(1, len(first_derivative) - 1):
        if first_derivative[i - 1] > 0 and first_derivative[i] < 0:
            peaks.append(i)
    return peaks



def calculate_fwhm(signal: np.ndarray, peak_index: int) -> int:
    """Calculate full width at half maximum (FWHM) for a peak."""
    peak_value = signal[peak_index]
    half_max = peak_value / 2.0

    left_idx = peak_index
    while left_idx > 0 and signal[left_idx] > half_max:
        left_idx -= 1

    right_idx = peak_index
    while right_idx < len(signal) - 1 and signal[right_idx] > half_max:
        right_idx += 1

    return right_idx - left_idx



def compute_similarity_score(
    peak1_index: int,
    peak1_width: int,
    peak2_index: int,
    peak2_width: int,
    alpha: float = 0.1,
    beta: float = 1.0,
) -> Tuple[int, float, float]:
    """Score peak similarity using index proximity and width ratio."""
    index_diff = abs(peak2_index - peak1_index)
    index_similarity = np.exp(-alpha * index_diff)
    width_ratio = peak2_width / peak1_width if peak1_width > 0 else 0
    width_similarity = np.exp(-beta * abs(width_ratio - 1))
    final_score = index_similarity * width_similarity
    return index_diff, width_ratio, final_score



def compare_chunks_for_peaks_and_widths(
    chunk1: np.ndarray,
    chunk2: np.ndarray,
    alpha: float = 0.005,
    beta: float = 1.0,
    similarity_threshold: float = 0.5,
    min_width_threshold: int = 15,
) -> Tuple[int | None, float | None, float, bool, int]:
    """Compare peak position and width similarity between two chunks."""
    first_derivative_1 = np.diff(chunk1)
    first_derivative_2 = np.diff(chunk2)

    peaks1 = detect_peaks_from_derivative(chunk1, first_derivative_1)
    peaks2 = detect_peaks_from_derivative(chunk2, first_derivative_2)

    if len(peaks1) == 0 or len(peaks2) == 0:
        return None, None, 0.0, False, 0

    peak1_index = peaks1[0]
    peak1_width = calculate_fwhm(chunk1, peak1_index)

    best_score = 0.0
    best_index_diff = None
    best_width_ratio = None
    count_large_peaks = 0

    for peak2_index in peaks2:
        peak2_width = calculate_fwhm(chunk2, peak2_index)
        if peak2_width >= min_width_threshold:
            count_large_peaks += 1

        if peak2_width > 0:
            index_diff, width_ratio, score = compute_similarity_score(
                peak1_index,
                peak1_width,
                peak2_index,
                peak2_width,
                alpha,
                beta,
            )
            if score > best_score:
                best_score = score
                best_index_diff = index_diff
                best_width_ratio = width_ratio

    is_match = best_score >= similarity_threshold
    return best_index_diff, best_width_ratio, best_score, is_match, count_large_peaks



def analyze_chunk_normality_t_similarity_peak(
    df: pd.DataFrame,
    gene_id: str,
    gene_of_interest: str,
    valid_chunks: Sequence[int],
    optimized_chunks: Sequence[pd.DataFrame],
) -> Dict[str, object]:
    """Compute chunk-wise N/T/S/P feature labels and associated statistics."""
    results: Dict[str, object] = {"gene_id": gene_id, "GeneOfInterest": gene_of_interest}

    for chunk_num in valid_chunks:
        try:
            gene_chunk_data = df.loc[gene_id, optimized_chunks[chunk_num].index].values
            goi_chunk_data = df.loc[gene_of_interest, optimized_chunks[chunk_num].index].values
        except KeyError:
            continue

        passes_normality = False
        passes_t_dist = False
        passes_similarity = False

        shapiro_stat, shapiro_p = shapiro(gene_chunk_data)
        skewness = skew(gene_chunk_data)
        kurt = kurtosis(gene_chunk_data, fisher=False)
        is_normal = (shapiro_p > 0.05) and (abs(skewness) < 0.5) and (2.5 < kurt < 3.5)
        if is_normal:
            passes_normality = True

        df_estimated, loc, scale = t.fit(gene_chunk_data)
        ks_stat_t, ks_p_t = kstest(gene_chunk_data, "t", args=(df_estimated, loc, scale))

        if df_estimated > 4:
            theoretical_kurtosis = 3 + (6 / (df_estimated - 4))
        else:
            theoretical_kurtosis = np.inf

        is_t_dist = (
            ks_p_t > 0.05
            and abs(skewness) < 0.5
            and abs(kurt - theoretical_kurtosis) < 1
        )
        if is_t_dist:
            passes_t_dist = True

        ks_stat, ks_p_value = ks_2samp(goi_chunk_data, gene_chunk_data)
        if ks_p_value > 0.05:
            passes_similarity = True

        chunk1_transformed = pd.Series(z_transform(goi_chunk_data))
        chunk2_transformed = pd.Series(z_transform(gene_chunk_data))
        chunk1_reordered, chunk2_reordered = apply_reordering_based_on_chunk1(
            chunk1_transformed,
            chunk2_transformed,
        )
        best_index_diff, best_width_ratio, best_score, peak_result, count_large_peaks = (
            compare_chunks_for_peaks_and_widths(
                chunk1_reordered,
                chunk2_reordered,
                alpha=0.005,
                beta=1.0,
            )
        )

        label_parts = []
        if passes_normality:
            label_parts.append("N")
        if passes_t_dist:
            label_parts.append("T")
        if passes_similarity:
            label_parts.append("S")
        if peak_result:
            label_parts.append("P")
        result_label = "+".join(label_parts) if label_parts else "None"

        min_pval = 1e-16
        results[f"chunk{chunk_num}_result"] = result_label
        results[f"chunk{chunk_num}_shapiro_p"] = shapiro_p
        results[f"chunk{chunk_num}_ks_p_t_dist"] = ks_p_t
        results[f"chunk{chunk_num}_ks_p_same_dist"] = ks_p_value
        results[f"chunk{chunk_num}_shapiro_p_neglog"] = -np.log10(max(shapiro_p, min_pval))
        results[f"chunk{chunk_num}_ks_p_t_dist_neglog"] = -np.log10(max(ks_p_t, min_pval))
        results[f"chunk{chunk_num}_ks_p_same_dist_neglog"] = -np.log10(max(ks_p_value, min_pval))
        results[f"chunk{chunk_num}_kurtosis"] = kurt
        results[f"chunk{chunk_num}_skewness"] = skewness
        results[f"chunk{chunk_num}_best_index_diff"] = best_index_diff
        results[f"chunk{chunk_num}_best_width_ratio"] = best_width_ratio
        results[f"chunk{chunk_num}_best_score"] = best_score
        results[f"chunk{chunk_num}_count_large_peaks"] = count_large_peaks

    return results



def load_expression_matrix(input_file: Path) -> pd.DataFrame:
    """Load expression matrix and set the first column as gene index."""
    df = pd.read_csv(input_file, sep="\t")
    df.set_index(df.columns[0], inplace=True)
    return df



def run_anchor_analysis(
    df: pd.DataFrame,
    gene_of_interest: str,
    gene_name: str,
    chunk_size: int,
    output_dir: Path,
    seed: int = 15,
    random_state: int = 15,
) -> None:
    """Run full feature extraction for one anchor gene and one chunk size."""
    if gene_of_interest not in df.index:
        print(f"Skipping {gene_of_interest} ({gene_name}): not found in expression matrix.")
        return

    print(f"Processing {gene_of_interest} ({gene_name}) | chunk size = {chunk_size}")
    goi_expression = df.loc[gene_of_interest]
    threshold = goi_expression.max()

    total_columns = df.shape[1]
    desired_chunks = math.ceil(total_columns / chunk_size)

    below_threshold, _ = exclude_and_assign_above_threshold(goi_expression, threshold)
    chunks = sort_and_divide_series_into_n_chunks(below_threshold, desired_chunks + 1)
    optimized_chunks, _ = optimize_chunks_for_normality_pandas(
        chunks,
        desired_chunks,
        seed=seed,
        random_state=random_state,
    )

    all_data = []
    for i, chunk in enumerate(optimized_chunks):
        chunk_df = chunk.copy()
        chunk_df["Chunk_Number"] = i
        all_data.append(chunk_df)

    reordered_df = pd.concat(all_data)
    reorder_file = output_dir / f"{gene_of_interest}_{gene_name}_sdx_{chunk_size}samples_peak_reorder_numbered.csv"
    reordered_df.to_csv(reorder_file)

    valid_chunks = list(range(desired_chunks))
    if not valid_chunks:
        print(f"No valid chunks available for {gene_of_interest} ({gene_name}).")
        return

    results = [
        analyze_chunk_normality_t_similarity_peak(
            df,
            gene,
            gene_of_interest,
            valid_chunks,
            optimized_chunks,
        )
        for gene in df.index.unique()
    ]

    results_df = pd.DataFrame(results)
    results_file = output_dir / (
        f"{gene_of_interest}_{gene_name}_featureCounts_"
        f"normality_t_similarity_height_width_results_sdx_{chunk_size}samples.csv"
    )
    results_df.to_csv(results_file, index=False)

    print(f"Saved: {reorder_file.name}")
    print(f"Saved: {results_file.name}")



def main() -> None:
    """Run chunk-based feature extraction for all anchors and chunk sizes."""
    input_dir = Path("/mnt/research/home/campus24/mislam69/populus_DTW/python/PNdtwFE/normal_t_similar_peak/feature_counts_data/counts")
    input_file = input_dir / "SDX_95_samples.txt"
    output_dir = Path("/mnt/research/home/campus24/mislam69/populus_DTW/python/PNdtwFE/normal_t_similar_peak/feature_counts_data/counts/multiple_chunk")
    output_dir.mkdir(parents=True, exist_ok=True)

    genes_of_interest = {
        "Potri.001G289200": "TOR1",
        "Potri.016G052800": "Lst8",
        "Potri.016G132000": "Raptor2",
    }
    chunk_sample_sizes = [20, 40, 60, 80]

    df = load_expression_matrix(input_file)
    print(f"Expression matrix loaded: {df.shape[0]} genes x {df.shape[1]} samples")

    for chunk_size in chunk_sample_sizes:
        print(f"\n=== Running analysis for chunk size {chunk_size} ===")
        for gene_of_interest, gene_name in genes_of_interest.items():
            run_anchor_analysis(
                df=df,
                gene_of_interest=gene_of_interest,
                gene_name=gene_name,
                chunk_size=chunk_size,
                output_dir=output_dir,
            )


if __name__ == "__main__":
    main()
