#!/usr/bin/env python3
"""
Multi-seed K-means clustering with known downstream-gene enrichment analysis.

Workflow
--------
1. Load the gene-by-feature matrix from a tab-delimited file.
2. Retain numeric feature columns and remove metadata and constant columns.
3. Standardize the feature matrix.
4. For each random seed and each k in the specified range:
   - run K-means clustering,
   - compute inertia and silhouette score,
   - quantify enrichment of known downstream genes in each cluster using the
     hypergeometric test,
   - adjust p-values with the Benjamini-Hochberg procedure,
   - save per-k cluster summaries and gene-to-cluster assignments.
5. For each seed, select the optimal k:
   - prioritize k values with at least one significantly enriched cluster,
   - among those, choose the one with the highest silhouette score,
   - otherwise choose the k with the highest silhouette score overall.
6. Save per-seed diagnostics, the selected clustering outputs, and a master
   summary across all tested seeds.

Outputs
-------
- One directory per seed containing:
  * cluster_summary_k{k}.csv
  * gene_cluster_assignments_k{k}.csv
  * k_diagnostics.csv
  * cluster_summary_CHOSEN_K.csv
  * gene_cluster_assignments_CHOSEN_K.csv
  * chosen_k_report.json
- A master summary across seeds:
  * sdx_all_seeds_summary.csv
  * sdx_overall_best_seed.json
"""

import json
import os
from pathlib import Path

# Limit threaded BLAS/OpenMP backends before importing NumPy/scikit-learn.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import hypergeom
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


DATA_PATH = "merged_tor_binary_40_sdx_weighted_filtered.tsv"
OUTPUT_DIR = "sdx_clustering_outputs"
SEED_RANGE = range(1, 101)
K_RANGE = range(5, 31)
N_INIT = 10
N_JOBS = 3

KNOWN_DOWNSTREAM_GENES = [
    "Potri.016G138400", "Potri.012G016500", "Potri.019G043400", "Potri.013G075300",
    "Potri.003G205200", "Potri.019G010700", "Potri.012G004400", "Potri.014G107000",
    "Potri.019G056200", "Potri.004G200800", "Potri.004G200900", "Potri.010G115800",
    "Potri.009G018100", "Potri.001G220100", "Potri.001G142500", "Potri.016G090800",
    "Potri.006G127100", "Potri.010G186400", "Potri.005G162600", "Potri.004G059000",
    "Potri.011G068600", "Potri.001G301800", "Potri.006G102200", "Potri.016G119700",
    "Potri.008G191600", "Potri.010G039700", "Potri.014G031000", "Potri.016G081000",
    "Potri.018G139400", "Potri.009G148900", "Potri.012G091900", "Potri.004G188200",
    "Potri.001G386900", "Potri.011G106800", "Potri.003G026600", "Potri.004G062400",
    "Potri.011G071800", "Potri.005G083500", "Potri.001G298500", "Potri.009G093200",
    "Potri.001G251600", "Potri.008G068200", "Potri.009G046400", "Potri.015G012200",
    "Potri.006G180800", "Potri.006G109600", "Potri.009G129700", "Potri.001G099400",
    "Potri.003G132300", "Potri.016G104900", "Potri.015G020500", "Potri.006G209500",
    "Potri.005G137700", "Potri.017G111800", "Potri.004G103700", "Potri.002G059800",
    "Potri.005G202000", "Potri.005G109400", "Potri.007G063600", "Potri.002G050800",
    "Potri.005G211500", "Potri.008G154600", "Potri.010G085700", "Potri.008G174100",
    "Potri.010G062900", "Potri.002G099500", "Potri.005G072700", "Potri.007G096300",
    "Potri.001G311000", "Potri.009G161900", "Potri.001G417300", "Potri.003G205700",
    "Potri.001G018400", "Potri.001G007800", "Potri.003G217900", "Potri.012G107100",
    "Potri.015G106000", "Potri.003G162700", "Potri.001G031400", "Potri.007G113100",
    "Potri.007G037800", "Potri.001G309800", "Potri.019G007200", "Potri.019G011300",
    "Potri.019G011600", "Potri.002G181000", "Potri.006G092900", "Potri.018G103000",
    "Potri.013G000300", "Potri.002G096900", "Potri.005G164700", "Potri.010G071200",
    "Potri.002G179800", "Potri.008G046200", "Potri.010G215200", "Potri.012G005900",
    "Potri.015G002300", "Potri.005G114700", "Potri.007G012100",
]


def benjamini_hochberg(pvalues: np.ndarray) -> np.ndarray:
    """Return Benjamini-Hochberg adjusted p-values in the original order."""
    pvalues = np.asarray(pvalues, dtype=float)
    n = len(pvalues)
    if n == 0:
        return np.array([], dtype=float)

    order = np.argsort(pvalues)
    adjusted = np.empty(n, dtype=float)
    running_min = 1.0

    for rank_from_end, idx in enumerate(order[::-1], start=1):
        rank = n - rank_from_end + 1
        bh_value = (pvalues[idx] * n) / rank
        running_min = min(running_min, bh_value)
        adjusted[idx] = running_min

    return np.clip(adjusted, 0.0, 1.0)


def safe_silhouette_score(X: np.ndarray, labels: np.ndarray) -> float:
    """Return the silhouette score, or NaN when it cannot be computed safely."""
    unique_labels, counts = np.unique(labels, return_counts=True)
    if len(unique_labels) < 2 or (counts < 2).any():
        return float("nan")

    try:
        return float(silhouette_score(X, labels, metric="euclidean"))
    except Exception:
        return float("nan")


def load_feature_matrix(data_path: str) -> tuple[np.ndarray, pd.Index, pd.Series]:
    """
    Load and preprocess the feature matrix.

    Returns
    -------
    X_scaled : np.ndarray
        Standardized numeric feature matrix.
    gene_ids : pd.Index
        Gene identifiers.
    is_known_downstream : pd.Series
        Boolean series indicating whether each gene is a known downstream gene.
    """
    df = pd.read_csv(data_path, sep="\t", index_col=0)
    gene_ids = df.index.astype(str)

    X = df.select_dtypes(include=[np.number]).copy()

    metadata_columns = {"score_total", "score", "is_KDG", "is_known_downstream"}
    drop_columns = [
        col for col in X.columns
        if col in metadata_columns or col.endswith("_score")
    ]
    if drop_columns:
        X = X.drop(columns=drop_columns, errors="ignore")

    non_constant_columns = X.nunique(axis=0) > 1
    X = X.loc[:, non_constant_columns]

    if X.shape[1] == 0:
        raise ValueError("No usable numeric feature columns remain after filtering.")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    known_downstream_set = set(KNOWN_DOWNSTREAM_GENES)
    is_known_downstream = pd.Series(gene_ids.isin(known_downstream_set), index=gene_ids)

    return X_scaled, gene_ids, is_known_downstream


def summarize_clusters(
    labels: np.ndarray,
    gene_ids: pd.Index,
    is_known_downstream: pd.Series,
    k: int,
) -> pd.DataFrame:
    """Create a per-cluster enrichment summary table."""
    total_genes = len(gene_ids)
    total_known = int(is_known_downstream.sum())
    background_rate = (total_known / total_genes) if total_genes > 0 else 0.0

    cluster_series = pd.Series(labels, index=gene_ids, name="cluster")
    cluster_sizes = cluster_series.value_counts().sort_index()
    known_counts = cluster_series[is_known_downstream.values].value_counts().reindex(
        cluster_sizes.index, fill_value=0
    )

    enrichments = []
    pvalues = []

    for cluster_id in cluster_sizes.index:
        cluster_size = int(cluster_sizes[cluster_id])
        known_in_cluster = int(known_counts.get(cluster_id, 0))

        if cluster_size > 0 and background_rate > 0:
            enrichment_ratio = (known_in_cluster / cluster_size) / background_rate
        else:
            enrichment_ratio = np.nan
        enrichments.append(enrichment_ratio)

        if (
            total_genes <= 0
            or total_known <= 0
            or cluster_size <= 0
            or known_in_cluster == 0
        ):
            pvalue = 1.0
        else:
            pvalue = float(hypergeom.sf(known_in_cluster - 1, total_genes, total_known, cluster_size))
        pvalues.append(pvalue)

    fdr_values = benjamini_hochberg(pvalues)

    summary = pd.DataFrame(
        {
            "k": k,
            "cluster_No": cluster_sizes.index,
            "total_genes": cluster_sizes.values,
            "No_downstream_genes": known_counts.reindex(cluster_sizes.index, fill_value=0).values,
            "enrichment_ratio": enrichments,
            "pvalue_hypergeom": pvalues,
            "fdr_bh": fdr_values,
        }
    )

    no_known_mask = summary["No_downstream_genes"] == 0
    summary.loc[no_known_mask, ["pvalue_hypergeom", "fdr_bh"]] = 1.0

    return summary[
        [
            "k",
            "cluster_No",
            "total_genes",
            "No_downstream_genes",
            "enrichment_ratio",
            "pvalue_hypergeom",
            "fdr_bh",
        ]
    ]


def choose_optimal_k(
    summaries_by_k: dict[int, pd.DataFrame],
    silhouette_by_k: dict[int, float],
    k_range: range,
) -> int:
    """
    Select the optimal k.

    Priority:
    1. Any k with at least one significantly enriched cluster (FDR <= 0.05).
    2. Among eligible k values, choose the highest silhouette score.
    3. If none are eligible, choose the highest silhouette score overall.
    """
    candidate_ks = []
    for k, summary in summaries_by_k.items():
        has_significant_cluster = (
            (summary["fdr_bh"] <= 0.05) & (summary["No_downstream_genes"] > 0)
        ).any()
        if has_significant_cluster:
            candidate_ks.append(k)

    if candidate_ks:
        return max(
            candidate_ks,
            key=lambda current_k: np.nan_to_num(silhouette_by_k.get(current_k, np.nan), nan=-1e9),
        )

    return max(
        k_range,
        key=lambda current_k: np.nan_to_num(silhouette_by_k.get(current_k, np.nan), nan=-1e9),
    )


def run_single_seed(
    seed: int,
    X_scaled: np.ndarray,
    gene_ids: pd.Index,
    is_known_downstream: pd.Series,
    output_dir: Path,
    k_range: range,
) -> dict:
    """Run clustering, enrichment analysis, and k-selection for a single seed."""
    output_dir.mkdir(parents=True, exist_ok=True)

    total_genes = len(gene_ids)
    total_known = int(is_known_downstream.sum())
    background_rate = (total_known / total_genes) if total_genes > 0 else 0.0

    summaries_by_k = {}
    inertia_by_k = {}
    silhouette_by_k = {}

    for k in k_range:
        model = KMeans(n_clusters=k, random_state=seed, n_init=N_INIT)
        labels = model.fit_predict(X_scaled)

        inertia_by_k[k] = float(model.inertia_)
        silhouette_by_k[k] = safe_silhouette_score(X_scaled, labels)

        summary = summarize_clusters(labels, gene_ids, is_known_downstream, k)
        summaries_by_k[k] = summary
        summary.to_csv(output_dir / f"cluster_summary_k{k}.csv", index=False)

        assignments = pd.DataFrame(
            {
                "gene_id": gene_ids,
                "cluster": labels,
                "is_known_downstream": is_known_downstream.values.astype(int),
            }
        )
        assignments.to_csv(output_dir / f"gene_cluster_assignments_k{k}.csv", index=False)

    chosen_k = choose_optimal_k(summaries_by_k, silhouette_by_k, k_range)

    diagnostics = pd.DataFrame(
        {
            "k": list(k_range),
            "inertia": [inertia_by_k[k] for k in k_range],
            "silhouette": [silhouette_by_k[k] for k in k_range],
        }
    )
    diagnostics.to_csv(output_dir / "k_diagnostics.csv", index=False)

    chosen_summary = summaries_by_k[chosen_k].copy().sort_values(
        ["fdr_bh", "enrichment_ratio", "No_downstream_genes", "total_genes"],
        ascending=[True, False, False, False],
    )
    chosen_summary.to_csv(output_dir / "cluster_summary_CHOSEN_K.csv", index=False)

    chosen_assignments_path = output_dir / f"gene_cluster_assignments_k{chosen_k}.csv"
    chosen_assignments_copy = output_dir / "gene_cluster_assignments_CHOSEN_K.csv"
    pd.read_csv(chosen_assignments_path).to_csv(chosen_assignments_copy, index=False)

    significant_clusters = chosen_summary[
        (chosen_summary["fdr_bh"] <= 0.05)
        & (chosen_summary["No_downstream_genes"] > 0)
    ].copy()

    report = {
        "seed": int(seed),
        "chosen_k": int(chosen_k),
        "silhouette": float(np.nan_to_num(silhouette_by_k.get(chosen_k, np.nan), nan=0.0)),
        "background_rate_known_downstream": float(background_rate),
        "N_genes": int(total_genes),
        "K_known_downstreams_found": int(total_known),
        "num_significant_clusters": int(len(significant_clusters)),
        "significant_clusters": significant_clusters[
            [
                "cluster_No",
                "total_genes",
                "No_downstream_genes",
                "enrichment_ratio",
                "pvalue_hypergeom",
                "fdr_bh",
            ]
        ].to_dict(orient="records"),
        "paths": {
            "k_diagnostics": str((output_dir / "k_diagnostics.csv").resolve()),
            "chosen_summary": str((output_dir / "cluster_summary_CHOSEN_K.csv").resolve()),
            "chosen_assignments": str(chosen_assignments_copy.resolve()),
            "per_k_dir": str(output_dir.resolve()),
        },
    }

    with open(output_dir / "chosen_k_report.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    top_cluster = chosen_summary.iloc[0]
    return {
        "seed": int(seed),
        "chosen_k": int(chosen_k),
        "silhouette": float(report["silhouette"]),
        "N": int(total_genes),
        "K": int(total_known),
        "bg_rate": float(background_rate),
        "num_sig_clusters": int(len(significant_clusters)),
        "best_cluster_No": int(top_cluster["cluster_No"]),
        "best_cluster_fdr": float(top_cluster["fdr_bh"]),
        "best_cluster_enrichment": float(top_cluster["enrichment_ratio"]),
        "seed_dir": str(output_dir.resolve()),
    }


def main() -> None:
    """Run the full multi-seed clustering and enrichment workflow."""
    base_output_dir = Path(OUTPUT_DIR)
    base_output_dir.mkdir(parents=True, exist_ok=True)

    X_scaled, gene_ids, is_known_downstream = load_feature_matrix(DATA_PATH)

    n_jobs = max(1, min(N_JOBS, os.cpu_count() or 1))
    print(f"Running multi-seed clustering with n_jobs={n_jobs} (CPU count={os.cpu_count()})")
    print(
        "Thread limits:",
        f"OMP={os.environ.get('OMP_NUM_THREADS')},",
        f"MKL={os.environ.get('MKL_NUM_THREADS')},",
        f"OPENBLAS={os.environ.get('OPENBLAS_NUM_THREADS')},",
        f"NUMEXPR={os.environ.get('NUMEXPR_NUM_THREADS')}",
    )

    def process_seed(seed: int) -> dict:
        seed_output_dir = base_output_dir / f"seed_{seed:03d}"
        return run_single_seed(
            seed=seed,
            X_scaled=X_scaled,
            gene_ids=gene_ids,
            is_known_downstream=is_known_downstream,
            output_dir=seed_output_dir,
            k_range=K_RANGE,
        )

    all_results = Parallel(
        n_jobs=n_jobs,
        backend="loky",
        verbose=10,
        max_nbytes="50M",
    )(delayed(process_seed)(seed) for seed in SEED_RANGE)

    master_summary = pd.DataFrame(all_results).sort_values(
        ["num_sig_clusters", "silhouette", "chosen_k"],
        ascending=[False, False, True],
    )
    master_summary_path = base_output_dir / "sdx_all_seeds_summary.csv"
    master_summary.to_csv(master_summary_path, index=False)

    best_seed_record = master_summary.iloc[0].to_dict()
    best_seed_path = base_output_dir / "sdx_overall_best_seed.json"
    with open(best_seed_path, "w", encoding="utf-8") as handle:
        json.dump(best_seed_record, handle, indent=2)

    print("\nMulti-seed run complete.")
    print(
        f"Best seed: {int(best_seed_record['seed'])} | "
        f"chosen_k={int(best_seed_record['chosen_k'])} | "
        f"silhouette={best_seed_record['silhouette']:.4f} | "
        f"significant_clusters={int(best_seed_record['num_sig_clusters'])}"
    )
    print(f"Master summary: {master_summary_path.resolve()}")
    print(f"Best-seed report: {best_seed_path.resolve()}")
    print(f"Per-seed output directory: {base_output_dir.resolve()}")


if __name__ == "__main__":
    main()
