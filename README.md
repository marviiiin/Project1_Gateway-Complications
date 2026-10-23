# Gateway Complications: Mining Comorbidity Networks in Type 2 Diabetes

Network analysis of Type 2 Diabetes (T2D) comorbidity patterns using the EHRShot dataset. This project constructs weighted co-occurrence networks from electronic health records and applies social network analysis techniques to reveal complication clusters, bridge diagnoses, and demographic disparities.

**Course:** CIS 4930 / CAI 5155 — Network Analysis & ML with Graphs, University of South Florida (Fall 2026)

**Authors:** Marvin Osei-Kuffour, Javier Varela

**[Read the Full Report (PDF)](https://drive.google.com/file/d/1ce9Y4KZxc52GAjXPEc92j5AZIEQiV87-/view?usp=drive_link)**

## Key Findings

- **12 comorbidity communities** detected via Louvain algorithm (modularity Q = 0.317), organized around clinically coherent disease domains (cardiometabolic, renal, hematologic, acute care, etc.)
- **52 bridge diagnoses** identified using betweenness centrality + participation coefficient — led by dyspnea, ECG abnormalities, and pleural effusion — that connect separate disease clusters
- **Metabolic triad** (hypertension, T2D, hyperlipidemia) confirmed as the primary temporal gateway to downstream complications
- **Demographic disparities:** female patients show more fragmented comorbidity landscapes (9 vs 7 communities); elderly patients (75+) show merging of clusters into 3 broad communities
- **Spectral vs Louvain comparison:** Louvain outperforms spectral graph partitioning (Q = 0.317 vs 0.266) with more balanced, clinically interpretable communities

## Project Structure

```
t2d_project/
├── 01_cohort_extraction.py        # Extract T2D cohort from EHRShot (OMOP CDM)
├── 02_network_construction.py     # Build co-occurrence network with Jaccard weights
├── 03_network_analysis.py         # Centrality, community detection, bridge analysis
├── 04_visualization.py            # Generate all figures
├── 05_gnn_link_prediction.py      # GNN-based link prediction (Project 2)
├── spectral_comparison.py         # Spectral partitioning vs Louvain comparison
├── regenerate_figs.py             # Network diagram regeneration (fig1, fig1b, fig1c)
├── regenerate_heatmap.py          # Jaccard heatmap regeneration (fig8)
├── Project1_Report.tex            # LaTeX report source
├── Project1_Report.pdf            # Compiled report (8 pages)
├── data/                          # Generated CSV data files
│   ├── network_nodes.csv          # 578 condition nodes
│   ├── network_cooccurrence_edges.csv  # 15,259 Jaccard-weighted edges
│   ├── analysis_node_metrics.csv  # Centrality & community assignments
│   ├── analysis_communities.csv   # Louvain community summaries
│   ├── analysis_bridge_diagnoses.csv   # Bridge diagnosis list
│   ├── analysis_spectral_comparison.csv # Spectral vs Louvain results
│   ├── network_subgroup_edges_*.csv    # Demographic subgroup edges
│   └── ...
├── figures/                       # All generated figures (PNG, 300 DPI)
│   ├── fig1_network_overview.png  # Main network diagram (top 80 nodes)
│   ├── fig1b_community_zoomin.png # Cardiometabolic community zoom-in
│   ├── fig1c_subgroup_networks.png # Male vs Female network comparison
│   ├── fig2_degree_distribution.png
│   ├── fig3_centrality_rankings.png
│   ├── fig4_community_composition.png
│   ├── fig5_bridge_diagnoses.png
│   ├── fig6_temporal_gateways.png
│   ├── fig7_demographic_comparison.png
│   ├── fig8_jaccard_heatmap.png
│   ├── fig_spectral_vs_louvain.png
│   └── ...
└── verify_*.py                    # Verification/validation scripts
```

## Pipeline

Run scripts in order:

```bash
python 01_cohort_extraction.py      # Requires EHRShot dataset access
python 02_network_construction.py
python 03_network_analysis.py
python 04_visualization.py
python spectral_comparison.py       # Spectral vs Louvain comparison
python regenerate_figs.py           # High-quality network diagrams
python regenerate_heatmap.py        # Jaccard heatmap
```

## Methods

| Step | Method | Details |
|------|--------|---------|
| Cohort | SNOMED concept hierarchy expansion | 1,354 T2D patients from EHRShot |
| Nodes | Conditions with >= 50 patients | 578 nodes |
| Edges | Jaccard similarity + Fisher's exact test (Bonferroni-corrected) | 15,259 edges |
| Communities | Louvain modularity optimization | 12 communities, Q = 0.317 |
| Bridges | Betweenness >= Q75 AND participation coefficient >= Q75 | 52 bridge diagnoses |
| Temporal | First-occurrence directed transitions | Gateway condition identification |
| Subgroups | Same node set, recomputed Jaccard per demographic | Male/Female, Under 60/60-74/75+ |

## Network Summary

| Metric | Value |
|--------|-------|
| Nodes | 578 |
| Edges | 15,259 |
| Density | 0.0915 |
| Avg Degree | 52.8 |
| Clustering Coefficient | 0.1267 |
| Transitivity | 0.4265 |
| Avg Path Length (LCC) | 2.197 |
| Diameter | 5 |

## Requirements

```
python >= 3.9
numpy
pandas
networkx
scikit-learn
matplotlib
scipy
```

For GNN link prediction (Project 2): `torch`, `torch-geometric`

## Dataset

This project uses the [EHRShot](https://github.com/som-shahlab/ehrshot-benchmark) dataset (Wornow et al., NeurIPS 2023), which contains de-identified longitudinal EHR data from Stanford Medicine in the OMOP Common Data Model format. Access requires a data use agreement.

## References

1. Hidalgo et al. (2009). "A dynamic network approach for the study of human phenotypes." *PLoS Computational Biology*. [DOI:10.1371/journal.pcbi.1000353](https://doi.org/10.1371/journal.pcbi.1000353)
2. Jensen et al. (2014). "Temporal disease trajectories condensed from population-wide registry data." *Nature Communications*. [DOI:10.1038/ncomms5022](https://doi.org/10.1038/ncomms5022)
3. Barabasi et al. (2011). "Network medicine: a network-based approach to human disease." *Nature Reviews Genetics*. [DOI:10.1038/nrg2918](https://doi.org/10.1038/nrg2918)
4. Blondel et al. (2008). "Fast unfolding of communities in large networks." *JSTAT*. [DOI:10.1088/1742-5468/2008/10/P10008](https://doi.org/10.1088/1742-5468/2008/10/P10008)
5. Guimera & Amaral (2005). "Functional cartography of metabolic networks." *Nature*. [DOI:10.1038/nature03288](https://doi.org/10.1038/nature03288)
6. Wornow et al. (2023). "EHRSHOT: An EHR benchmark for few-shot evaluation of foundation models." *NeurIPS*. [DOI:10.48550/arXiv.2307.02028](https://doi.org/10.48550/arXiv.2307.02028)
