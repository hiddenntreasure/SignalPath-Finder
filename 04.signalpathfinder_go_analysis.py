"""
Run GO enrichment for significant clusters identified across seed-specific outputs.

Workflow
--------
1. Load a global Arabidopsis TAIR background from the Populus annotation file.
2. Scan the clustering output directory for `seed_*` folders.
3. For each seed, read the exported table of significant clusters.
4. For each significant cluster, load the cluster gene list with TAIR mappings.
5. Use clusterProfiler::enrichGO through rpy2 with the global TAIR background.
6. Save per-cluster GO result tables and dotplot PDFs.
7. Compile a master summary across all processed seeds and clusters.

Expected inputs
---------------
- <BASE_OUT_DIR>/seed_*/sig_clusters/significant_clusters_table.csv
- <BASE_OUT_DIR>/seed_*/sig_clusters/cluster_##_genes_with_arabidopsis.csv
- Mapping file containing a `best_arabi_gene` column

Generated outputs
-----------------
- <seed_dir>/GO_BY_CLUSTER/cluster_#/cluster_#_GO_enrichment.csv
- <seed_dir>/GO_BY_CLUSTER/cluster_#/cluster_#_GO_enrichment_dotplot.pdf
- <BASE_OUT_DIR>/master_GO_summary.csv
"""

import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects import default_converter, r
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr

BASE_OUT_DIR = Path("sdx_clustering_outputs/")
MAP_FILE = Path("Ptrichocarpa_533_v4.1.annotation_info_tair10_symbol_TFs.txt")

ONTOLOGY = "BP"
P_ADJUST_METHOD = "BH"
PVALUE_CUTOFF = 0.05
QVALUE_CUTOFF = 0.05
MIN_GENESET_SIZE = 10
MAX_GENESET_SIZE = 2000
DOTPLOT_CATEGORIES = 20
DOTPLOT_FONT_SIZE = 6

TAIR_PATTERN = re.compile(r"^AT[1-5CM]G\d{5}$", re.I)


def load_global_background(map_file: Path) -> list[str]:
    """Load unique TAIR identifiers from the mapping file."""
    mapping_df = pd.read_csv(map_file, sep="\t", dtype=str, encoding="latin1")
    tair_ids = (
        mapping_df["best_arabi_gene"].dropna().astype(str).str.strip().str.upper()
    )
    tair_ids = tair_ids[tair_ids.ne("---")]
    tair_ids = tair_ids[tair_ids.str.match(TAIR_PATTERN)]
    background = sorted(tair_ids.unique())
    print(f"[global background] unique TAIR IDs = {len(background)}")
    return background


def find_seed_directories(base_output_dir: Path) -> list[Path]:
    """Return all seed-specific output directories."""
    return sorted(path for path in base_output_dir.glob("seed_*") if path.is_dir())


def read_significant_clusters_table(seed_dir: Path) -> Optional[pd.DataFrame]:
    """Load the exported significant-cluster table for one seed."""
    table_path = seed_dir / "sig_clusters" / "significant_clusters_table.csv"
    if not table_path.exists():
        return None
    return pd.read_csv(table_path)


def read_cluster_gene_file(seed_dir: Path, cluster_number: int) -> Optional[pd.DataFrame]:
    """Load the TAIR-mapped gene table for one cluster."""
    candidate_paths = [
        seed_dir / "sig_clusters" / f"cluster_{cluster_number:02d}_genes_with_arabidopsis.csv",
        seed_dir / "sig_clusters" / f"cluster_{cluster_number}_genes_with_arabidopsis.csv",
    ]

    for file_path in candidate_paths:
        if file_path.exists():
            return pd.read_csv(file_path, dtype=str)
    return None


def load_r_dependencies():
    """Import required R packages for GO enrichment and plotting."""
    cluster_profiler = importr("clusterProfiler")
    r("library(org.At.tair.db)")
    enrichplot = importr("enrichplot")
    grdevices = importr("grDevices")
    orgdb = r("org.At.tair.db")
    return cluster_profiler, orgdb, enrichplot, grdevices


def run_go_enrichment(
    foreground_tair: list[str],
    background_tair: list[str],
    output_pdf: Path,
    ontology: str = ONTOLOGY,
) -> pd.DataFrame:
    """Run enrichGO and save a dotplot PDF for the foreground gene set."""
    cluster_profiler, orgdb, enrichplot, grdevices = load_r_dependencies()

    enrich_result = cluster_profiler.enrichGO(
        gene=ro.StrVector(sorted(set(foreground_tair))),
        OrgDb=orgdb,
        keyType="TAIR",
        universe=ro.StrVector(sorted(set(background_tair))),
        ont=ontology,
        pAdjustMethod=P_ADJUST_METHOD,
        pvalueCutoff=PVALUE_CUTOFF,
        qvalueCutoff=QVALUE_CUTOFF,
        minGSSize=MIN_GENESET_SIZE,
        maxGSSize=MAX_GENESET_SIZE,
    )

    with localconverter(default_converter + pandas2ri.converter):
        enrich_df = ro.conversion.rpy2py(r("as.data.frame")(enrich_result))

    grdevices.pdf(str(output_pdf), width=9, height=6)
    r["print"](
        enrichplot.dotplot(
            enrich_result,
            showCategory=DOTPLOT_CATEGORIES,
            **{"font.size": DOTPLOT_FONT_SIZE},
        )
    )
    grdevices.dev_off()

    if enrich_df is None:
        return pd.DataFrame()
    return enrich_df


def collect_foreground_tair_ids(cluster_df: pd.DataFrame) -> list[str]:
    """Extract valid TAIR IDs from a cluster gene table."""
    tair_series = cluster_df["TAIR_ID"].fillna("").astype(str).str.upper().str.strip()
    return sorted({tair_id for tair_id in tair_series if tair_id != "---" and TAIR_PATTERN.match(tair_id)})


def summarize_cluster(
    seed_dir: Path,
    cluster_row: pd.Series,
    global_background: list[str],
) -> dict | None:
    """Run GO enrichment for one significant cluster and return summary metadata."""
    cluster_number = int(cluster_row["cluster_No"])
    prev_total_genes = int(cluster_row.get("total_genes", 0))
    prev_no_downstream = int(cluster_row.get("No_downstream_genes", 0))
    prev_enrichment_ratio = float(cluster_row.get("enrichment_ratio", np.nan))
    prev_pvalue_hypergeom = float(cluster_row.get("pvalue_hypergeom", np.nan))
    prev_fdr_bh = float(cluster_row.get("fdr_bh", np.nan))

    cluster_df = read_cluster_gene_file(seed_dir, cluster_number)
    if cluster_df is None or cluster_df.empty:
        print(f"[{seed_dir.name}] cluster {cluster_number}: missing cluster file — skip")
        return None

    foreground_tair = collect_foreground_tair_ids(cluster_df)
    n_tair_foreground = len(foreground_tair)
    n_potri_foreground = cluster_df["Potri_ID"].dropna().astype(str).str.strip().nunique()

    output_dir = seed_dir / "GO_BY_CLUSTER" / f"cluster_{cluster_number}"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / f"cluster_{cluster_number}_GO_enrichment.csv"

    if n_tair_foreground == 0:
        pd.DataFrame().to_csv(output_csv, index=False)
        return {
            "seed": seed_dir.name,
            "cluster_No": cluster_number,
            "prev_total_genes": prev_total_genes,
            "prev_No_downstream_genes": prev_no_downstream,
            "prev_enrichment_ratio": prev_enrichment_ratio,
            "prev_pvalue_hypergeom": prev_pvalue_hypergeom,
            "prev_fdr_bh": prev_fdr_bh,
            "n_potri_fg": n_potri_foreground,
            "n_tair_fg": n_tair_foreground,
            "n_GO_pvalue_<=0.05": 0,
            "n_GO_padj_<=0.05": 0,
            "ont": ONTOLOGY,
            "p_adjust_method": P_ADJUST_METHOD,
            "seed_dir": str(seed_dir.resolve()),
            "cluster_dir": str(output_dir.resolve()),
        }

    output_pdf = output_dir / f"cluster_{cluster_number}_GO_enrichment_dotplot.pdf"
    enrich_df = run_go_enrichment(foreground_tair, global_background, output_pdf, ontology=ONTOLOGY)

    if enrich_df.empty:
        enrich_df.to_csv(output_csv, index=False)
        n_go_raw = 0
        n_go_adjusted = 0
    else:
        enrich_df.to_csv(output_csv, index=False)
        n_go_raw = int((enrich_df["pvalue"].astype(float) <= PVALUE_CUTOFF).sum())
        n_go_adjusted = int((enrich_df["p.adjust"].astype(float) <= QVALUE_CUTOFF).sum())

    return {
        "seed": seed_dir.name,
        "cluster_No": cluster_number,
        "prev_total_genes": prev_total_genes,
        "prev_No_downstream_genes": prev_no_downstream,
        "prev_enrichment_ratio": prev_enrichment_ratio,
        "prev_pvalue_hypergeom": prev_pvalue_hypergeom,
        "prev_fdr_bh": prev_fdr_bh,
        "n_potri_fg": n_potri_foreground,
        "n_tair_fg": n_tair_foreground,
        "n_GO_pvalue_<=0.05": n_go_raw,
        "n_GO_padj_<=0.05": n_go_adjusted,
        "ont": ONTOLOGY,
        "p_adjust_method": P_ADJUST_METHOD,
        "seed_dir": str(seed_dir.resolve()),
        "cluster_dir": str(output_dir.resolve()),
    }


def main() -> None:
    """Run GO enrichment for all significant clusters across all seeds."""
    global_background = load_global_background(MAP_FILE)
    master_rows = []

    for seed_dir in find_seed_directories(BASE_OUT_DIR):
        significant_df = read_significant_clusters_table(seed_dir)
        if significant_df is None or significant_df.empty:
            print(f"[{seed_dir.name}] no significant cluster table found — skip")
            continue

        for _, cluster_row in significant_df.iterrows():
            summary_row = summarize_cluster(seed_dir, cluster_row, global_background)
            if summary_row is not None:
                master_rows.append(summary_row)

    if master_rows:
        master_df = pd.DataFrame(master_rows).sort_values(["seed", "cluster_No"])
        output_path = BASE_OUT_DIR / "master_GO_summary.csv"
        master_df.to_csv(output_path, index=False)
        print(f"\nMaster summary written to: {output_path}")
    else:
        print("\nNo GO results were generated. Check inputs and paths.")


if __name__ == "__main__":
    main()
