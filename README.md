# SignalPath-Finder

SignalPath-Finder is a computational framework for identifying **downstream genes regulated by a gene complex** using transcriptomic data and prior biological knowledge. The method integrates distribution-based feature extraction, clustering guided by known downstream genes, and autoencoder-based gene ranking to prioritize candidate downstream genes.

This repository contains the analysis pipeline used to identify **TOR complex downstream genes** from *Populus trichocarpa* transcriptomic datasets.

---

# Pipeline Overview

The SignalPath-Finder pipeline consists of five main stages:

1. Pseudo peak based feature extraction from gene expression profiles
2. Multi-seed clustering with downstream gene enrichment testing
3. Significant cluster summarization
4. GO enrichment analysis
5. Autoencoder-based gene ranking

---
## Requirements

Python >= 3.10  
R >= 4.3

### Python packages
- numpy == 1.26.4
- pandas == 2.2.3
- scipy == 1.7.3
- scikit-learn == 1.6.1
- tensorflow == 2.18.1
- matplotlib == 3.8.3
- rpy2 == 3.5.11

### R / Bioconductor packages
- BiocManager
- clusterProfiler == 4.10.0
- org.At.tair.db == 3.18.0
- GO.db == 3.18.0
- enrichplot == 1.22.0
- ggplot2 == 3.5.1

## Usage
Each scripts require user-defined file paths and parameter settings.
Run the scripts in the following order:

```bash
python3 01.signalpathfinder_psedu_peak_based_feature_extraction.py
python3 02.signalpathfinder_cluster_selection.py
python3 03.signalpathfinder_cluster_summary.py
python3 04.signalpathfinder_go_analysis.py
python3 05.signalpathfinder_gene_ranking.py
```

# Repository Structure
```text
SignalPath-Finder/
│
├── 01_signalpathfinder_feature_extraction.py
├── 02_signalpathfinder_cluster_selection.py
├── 03_signalpathfinder_cluster_summary.py
├── 04_signalpathfinder_go_analysis.py
├── 05_signalpathfinder_gene_ranking.py
│
├── data/
│   ├── expression_matrix.txt
│   ├── feature_matrix.tsv
│   ├── gene_cluster_assignments_CHOSEN_K.csv
│   └── gene_mapping_file.txt
│
├── results/
│   │
│   ├── 01_feature_extraction/
│   │   └── feature_matrix_outputs/
│   │
│   ├── 02_cluster_selection/
│   │   ├── all_seeds_summary.csv
│   │   ├── overall_best_seed.json
│   │   ├── seed_001/
│   │   │   ├── k_diagnostics.csv
│   │   │   ├── cluster_summary_CHOSEN_K.csv
│   │   │   ├── gene_cluster_assignments_CHOSEN_K.csv
│   │   │   ├── chosen_k_report.json
│   │   │   └── gene_cluster_assignments_k*.csv
│   │   └── seed_002/
│   │       └── ...
│   │
│   ├── 03_cluster_summary/
│   │   ├── sig_cluster_mapping_summary.csv
│   │   ├── seed_001/
│   │   │   └── sig_clusters/
│   │   │       ├── significant_clusters_table.csv
│   │   │       ├── all_sig_clusters_genes_with_arabidopsis.csv
│   │   │       ├── cluster_01_genes_with_arabidopsis.csv
│   │   │       └── cluster_02_genes_with_arabidopsis.csv
│   │   └── seed_002/
│   │       └── ...
│   │
│   ├── 04_go_analysis/
│   │   ├── master_GO_summary.csv
│   │   ├── seed_001/
│   │   │   └── GO_BY_CLUSTER/
│   │   │       ├── cluster_1/
│   │   │       │   ├── cluster_1_GO_enrichment.csv
│   │   │       │   └── cluster_1_GO_dotplot.pdf
│   │   │       └── cluster_2/
│   │   │           └── ...
│   │   └── seed_002/
│   │       └── ...
│   │
│   └── 05_gene_ranking/
│       ├── multi_cluster_summary.csv
│       ├── cluster_002/
│       │   ├── AE_stageA_results_cluster_2.csv
│       │   ├── AE_master_grid_results_cluster_2.csv
│       │   ├── per_seed_ranks/
│       │   │   └── <config_id>__seed-<seed>.csv
│       │   └── top5_seeds/
│       │       ├── top5_seeds_cluster_2_summary.csv
│       │       └── <config_id>__seed-<seed>.csv
│       ├── cluster_003/
│       │   └── ...
│       └── cluster_022/
│           └── ...
│
└── README.md
```





## Key Outputs

Important output files include:
- **feature matrix** generated from pseudo-peak profiles  
- **cluster summaries** and selected seed reports  
- **significant cluster gene tables**  
- **GO enrichment summaries**  
- **autoencoder-based gene ranking tables**  
- **multi-cluster summary** of final ranking performance

## Notes
- The pipeline was developed for identifying **TOR complex downstream genes** in *Populus trichocarpa*.
- Some scripts require user-defined file paths and parameter settings before execution.
