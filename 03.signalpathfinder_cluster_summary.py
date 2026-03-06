"""
Summarize significant clusters across seed-specific clustering outputs.

Workflow
--------
1. Scan the clustering output directory for `seed_*` folders.
2. For each seed, load the selected cluster summary and gene-to-cluster assignments.
3. Identify significant clusters using the configured FDR threshold.
4. Extract Populus gene IDs for each significant cluster.
5. Map Populus IDs to Arabidopsis TAIR IDs using the annotation file.
6. Save per-cluster gene tables, a combined significant-cluster gene table,
   and a per-seed summary report.

Expected inputs
---------------
- <BASE_OUT_DIR>/seed_*/cluster_summary_CHOSEN_K.csv
- <BASE_OUT_DIR>/seed_*/gene_cluster_assignments_CHOSEN_K.csv
- Mapping file with Populus and Arabidopsis identifier columns

Generated outputs
-----------------
- <seed_dir>/sig_clusters/cluster_##_genes_with_arabidopsis.csv
- <seed_dir>/sig_clusters/all_sig_clusters_genes_with_arabidopsis.csv
- <seed_dir>/sig_clusters/significant_clusters_table.csv
- <BASE_OUT_DIR>/sig_cluster_mapping_summary.csv
"""

from pathlib import Path

import pandas as pd

BASE_OUT_DIR = Path("sdx_clustering_outputs/")
MAPPING_PATH = Path("Ptrichocarpa_533_v4.1.annotation_info_tair10_symbol_TFs.txt")
MAPPING_SEPARATOR = "\t"
FDR_THRESHOLD = 0.05

SIGNIFICANT_CLUSTER_COLUMNS = [
    "cluster_No",
    "total_genes",
    "No_downstream_genes",
    "pvalue_hypergeom",
    "fdr_bh",
    "enrichment_ratio",
]


def load_mapping(mapping_path: Path, separator: str = MAPPING_SEPARATOR) -> pd.DataFrame:
    """Load and standardize the Populus-to-Arabidopsis mapping table."""
    mapping_df = pd.read_csv(mapping_path, sep=separator, dtype=str, encoding="latin1")
    column_lookup = {column.lower(): column for column in mapping_df.columns}

    potri_column = (
        column_lookup.get("locusname")
        or column_lookup.get("potri")
        or column_lookup.get("gene")
        or column_lookup.get("geneid")
    )
    tair_column = (
        column_lookup.get("best_arabi_gene")
        or column_lookup.get("arabidopsis")
        or column_lookup.get("tair")
        or column_lookup.get("best_arabi")
    )

    if not potri_column or not tair_column:
        raise ValueError(
            "Required mapping columns were not found. "
            f"Available columns: {list(mapping_df.columns)}"
        )

    mapping_df = mapping_df.rename(
        columns={potri_column: "locusName", tair_column: "best_arabi_gene"}
    )
    mapping_df["locusName"] = mapping_df["locusName"].astype(str).str.strip()
    mapping_df["best_arabi_gene"] = (
        mapping_df["best_arabi_gene"].astype(str).str.strip()
    )

    return mapping_df[["locusName", "best_arabi_gene"]].drop_duplicates()


def find_seed_directories(base_output_dir: Path) -> list[Path]:
    """Return all seed-specific output directories."""
    return sorted(path for path in base_output_dir.glob("seed_*") if path.is_dir())


def load_significant_clusters(summary_df: pd.DataFrame, fdr_threshold: float) -> pd.DataFrame:
    """Filter and format significant clusters from a cluster summary table."""
    if "pvalue_hypergeom" not in summary_df.columns:
        summary_df["pvalue_hypergeom"] = pd.NA

    available_columns = [
        column for column in SIGNIFICANT_CLUSTER_COLUMNS if column in summary_df.columns
    ]

    significant_df = (
        summary_df.loc[summary_df["fdr_bh"] <= fdr_threshold, available_columns]
        .sort_values(
            ["fdr_bh", "enrichment_ratio", "total_genes"],
            ascending=[True, False, False],
        )
        .copy()
    )

    for column in SIGNIFICANT_CLUSTER_COLUMNS:
        if column not in significant_df.columns:
            significant_df[column] = pd.NA

    return significant_df[SIGNIFICANT_CLUSTER_COLUMNS]


def extract_cluster_gene_table(
    assignment_df: pd.DataFrame,
    cluster_number: int,
    mapping_df: pd.DataFrame,
) -> pd.DataFrame:
    """Return the Populus-to-TAIR mapping table for one cluster."""
    genes_df = (
        assignment_df[["gene_id", "cluster"]]
        .rename(columns={"gene_id": "Potri_ID"})
        .copy()
    )
    genes_df["Potri_ID"] = genes_df["Potri_ID"].astype(str).str.strip()

    cluster_df = genes_df.loc[genes_df["cluster"] == cluster_number, ["Potri_ID"]].copy()
    cluster_df = cluster_df.merge(
        mapping_df,
        left_on="Potri_ID",
        right_on="locusName",
        how="left",
    )
    cluster_df = cluster_df[["Potri_ID", "best_arabi_gene"]].rename(
        columns={"best_arabi_gene": "TAIR_ID"}
    )
    cluster_df.insert(0, "cluster_No", cluster_number)
    return cluster_df


def process_seed_directory(
    seed_dir: Path,
    mapping_df: pd.DataFrame,
    fdr_threshold: float = FDR_THRESHOLD,
) -> dict | None:
    """Process one seed directory and export significant-cluster mapping outputs."""
    summary_path = seed_dir / "cluster_summary_CHOSEN_K.csv"
    assignments_path = seed_dir / "gene_cluster_assignments_CHOSEN_K.csv"

    if not summary_path.exists() or not assignments_path.exists():
        print(f"[skip] Missing required files in {seed_dir}")
        return None

    summary_df = pd.read_csv(summary_path)
    assignment_df = pd.read_csv(assignments_path, dtype={"gene_id": str, "cluster": int})
    significant_df = load_significant_clusters(summary_df, fdr_threshold)

    if significant_df.empty:
        print(f"[{seed_dir.name}] No significant clusters at FDR ≤ {fdr_threshold}.")
        return {"seed_dir": str(seed_dir), "n_sig_clusters": 0, "out_dir": None}

    output_dir = seed_dir / "sig_clusters"
    output_dir.mkdir(parents=True, exist_ok=True)

    combined_cluster_tables = []
    for cluster_number in significant_df["cluster_No"].astype(int).tolist():
        cluster_df = extract_cluster_gene_table(assignment_df, cluster_number, mapping_df)
        cluster_df.to_csv(
            output_dir / f"cluster_{cluster_number:02d}_genes_with_arabidopsis.csv",
            index=False,
        )
        combined_cluster_tables.append(cluster_df)

    pd.concat(combined_cluster_tables, ignore_index=True).to_csv(
        output_dir / "all_sig_clusters_genes_with_arabidopsis.csv",
        index=False,
    )
    significant_df.to_csv(output_dir / "significant_clusters_table.csv", index=False)

    print(f"[{seed_dir.name}] Saved {len(significant_df)} significant clusters → {output_dir}")
    return {
        "seed_dir": str(seed_dir),
        "n_sig_clusters": int(len(significant_df)),
        "out_dir": str(output_dir),
    }


def main() -> None:
    """Run significant-cluster extraction and mapping for all seeds."""
    mapping_df = load_mapping(MAPPING_PATH, separator=MAPPING_SEPARATOR)

    results = []
    for seed_dir in find_seed_directories(BASE_OUT_DIR):
        result = process_seed_directory(seed_dir, mapping_df, fdr_threshold=FDR_THRESHOLD)
        if result is not None:
            results.append(result)

    if results:
        summary_output = BASE_OUT_DIR / "sig_cluster_mapping_summary.csv"
        pd.DataFrame(results).to_csv(summary_output, index=False)
        print(f"\nWrote summary: {summary_output}")


if __name__ == "__main__":
    main()
