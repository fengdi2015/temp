"""
Reprocess the raw LuCA T-cell AnnData from scratch (QC -> normalize -> HVG ->
PCA -> Harmony batch correction -> neighbors -> UMAP -> Leiden clustering),
save the new UMAP coordinates back into the h5ad, and generate three
standard scanpy visualizations against a canonical T-cell marker panel:
  1. UMAP colored by cluster/subset
  2. Marker-gene heatmap (mean expression per cluster, z-scored)
  3. Marker-gene dot plot (% expressing + mean expression per cluster)

Input:  luca_tcells_raw.h5ad  (3,500 T cells x4 disease groups from the
        CELLxGENE Census LuCA lung cancer atlas; see 01_fetch_data.py)
Output: luca_tcells_reprocessed.h5ad  (new checkpoint with X_umap, leiden,
        subset annotation)
        umap_by_cluster.png
        marker_heatmap.png
        marker_dotplot.png
"""
import scanpy as sc
import numpy as np
import pandas as pd
import harmonypy
import matplotlib.pyplot as plt

sc.settings.verbosity = 1

# ---------------------------------------------------------------------
# 1. Load + QC
# ---------------------------------------------------------------------
adata = sc.read_h5ad("luca_tcells_raw.h5ad")
adata.var_names = adata.var["feature_name"].astype(str)
adata.var_names_make_unique()
adata.var.index.name = None  # avoid index-name/column collision on h5ad write (feature_name is both index and a column)
adata.layers["counts"] = adata.X.copy()

adata.var["mt"] = adata.var_names.str.startswith("MT-")
sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True, percent_top=None)

sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)
upper = adata.obs.total_counts.quantile(0.995)
adata = adata[adata.obs.total_counts <= upper].copy()
# Drop Smart-seq2 cells: non-UMI counts are not comparable in magnitude to 10x UMI counts.
adata = adata[adata.obs.assay.astype(str) != "Smart-seq2"].copy()
print("cells after QC:", adata.n_obs)

# ---------------------------------------------------------------------
# 2. Normalize + HVG + PCA
# ---------------------------------------------------------------------
adata.X = adata.layers["counts"].copy()
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
adata.layers["lognorm"] = adata.X.copy()

sc.pp.highly_variable_genes(adata, n_top_genes=2000, batch_key="assay")
adata.raw = adata
adata_hvg = adata[:, adata.var.highly_variable].copy()
sc.pp.scale(adata_hvg, max_value=10)
sc.tl.pca(adata_hvg, n_comps=50, svd_solver="arpack")
adata.obsm["X_pca"] = adata_hvg.obsm["X_pca"]
print("PCA:", adata.obsm["X_pca"].shape)

# ---------------------------------------------------------------------
# 3. Harmony batch correction (manual call — the scanpy external wrapper
#    had a compatibility bug at the time of this analysis)
# ---------------------------------------------------------------------
ho = harmonypy.run_harmony(adata.obsm["X_pca"], adata.obs, ["assay"], max_iter_harmony=20)
Zc = np.asarray(ho.Z_corr)
# harmonypy returns Z_corr as (n_pcs, n_cells); transpose to (n_cells, n_pcs)
adata.obsm["X_pca_harmony"] = Zc.T if Zc.shape[0] != adata.n_obs else Zc

# ---------------------------------------------------------------------
# 4. Neighbors / UMAP / Leiden -- THIS IS THE NEW UMAP EMBEDDING
# ---------------------------------------------------------------------
sc.pp.neighbors(adata, use_rep="X_pca_harmony", n_neighbors=15)
sc.tl.umap(adata)
sc.tl.leiden(adata, resolution=1.0, key_added="leiden", flavor="igraph", n_iterations=2)
print(adata.obs.leiden.value_counts())

# ---------------------------------------------------------------------
# 5. Cluster annotation (marker-based; re-derive with rank_genes_groups if
#    you change clustering resolution / need to re-check cluster identity)
# ---------------------------------------------------------------------
annotation = {
    "0": "CD4 T (resting, low-marker)",
    "1": "CD4 T (Ig-ambient/plasma-contam)",
    "2": "T cell (stress-response, HSP-high)",
    "3": "Regulatory T (Treg)",
    "4": "CD4 Naive/Tcm",
    "5": "T cell (RBC-contam, low quality)",
    "6": "CD8 Effector Memory (GZMK+)",
    "7": "CD8 Terminal Effector (GZMH+GNLY+)",
    "8": "CD4 Naive/resting",
    "9": "CD8 Memory (NK-like, KLRC1+)",
    "10": "CD8 Exhausted/Tumor-reactive (CXCL13+GZMB+)",
    "11": "Proliferating T (MKI67+)",
    "12": "CD4/Treg CXCL13+ (dysfunctional, tumor-assoc.)",
    "13": "T cell (Ig-ambient, low quality)",
    "14": "T cell (high mito, low quality)",
    "15": "Innate-like T (MAIT/NKT-like, KLRB1+)",
    "16": "Doublet/contam (myeloid markers)",
}
# Clusters not in the mapping (e.g. if re-run with a different resolution)
# fall back to a generic "Cluster N" label rather than silently dropping cells.
present = adata.obs.leiden.astype(str).unique()
for c in present:
    annotation.setdefault(c, f"Cluster {c} (unannotated)")
adata.obs["subset"] = adata.obs.leiden.astype(str).map(annotation).astype("category")

low_qual = {"2", "5", "13", "14", "16"}
adata.obs["is_low_quality"] = adata.obs.leiden.astype(str).isin(low_qual)

# ---------------------------------------------------------------------
# 6. Save the reprocessed checkpoint (new UMAP coordinates included)
# ---------------------------------------------------------------------
adata.write_h5ad("luca_tcells_reprocessed.h5ad")
print("saved luca_tcells_reprocessed.h5ad with X_umap shape:", adata.obsm["X_umap"].shape)

clean = adata[~adata.obs.is_low_quality].copy()
print("clean cells for plotting:", clean.n_obs)

# ---------------------------------------------------------------------
# 7. UMAP visualization colored by subset
# ---------------------------------------------------------------------
subset_order = clean.obs["subset"].value_counts().index.tolist()
palette = plt.get_cmap("tab20")(np.linspace(0, 1, len(subset_order)))
color_map = dict(zip(subset_order, palette))

fig, ax = plt.subplots(figsize=(8, 6.5))
coords = clean.obsm["X_umap"]
for s in subset_order:
    m = clean.obs["subset"] == s
    ax.scatter(coords[m, 0], coords[m, 1], s=3, alpha=0.7, color=color_map[s], label=s, linewidths=0)
ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")
ax.set_title("T-cell subsets (reprocessed UMAP)", loc="left")
ax.legend(fontsize=6, markerscale=3, loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
fig.tight_layout()
fig.savefig("umap_by_cluster.png", dpi=200, bbox_inches="tight")
print("saved umap_by_cluster.png")

# ---------------------------------------------------------------------
# 8. Marker gene panel (canonical T-cell subset markers)
# ---------------------------------------------------------------------
marker_genes = ["CCR7", "SELL", "TCF7", "IL7R",             # naive/Tcm
                "FOXP3", "CTLA4", "IL2RA",                   # Treg
                "GZMK", "GZMA",                              # effector memory
                "GZMH", "GZMB", "GNLY", "NKG7", "PRF1",       # cytotoxic/terminal effector
                "CXCL13", "HAVCR2", "PDCD1", "LAG3", "TIGIT", # exhaustion
                "MKI67",                                     # proliferation
                "KLRB1", "KLRC1"]                             # innate-like / NK-like
marker_genes = [g for g in marker_genes if g in adata.raw.var_names]

X = clean.raw[:, marker_genes].X
expr = pd.DataFrame(X.toarray() if hasattr(X, "toarray") else X,
                     columns=marker_genes, index=clean.obs_names)
expr["subset"] = clean.obs["subset"].values

mean_expr = expr.groupby("subset", observed=True).mean()
mean_expr = mean_expr.loc[[s for s in subset_order if s in mean_expr.index]]
pct_expr = (expr.drop(columns="subset") > 0).groupby(expr["subset"], observed=True).mean() * 100
pct_expr = pct_expr.loc[mean_expr.index]

# --- 8a. Heatmap: z-scored mean expression per cluster ---
zexpr = (mean_expr - mean_expr.mean(axis=0)) / (mean_expr.std(axis=0) + 1e-9)
fig, ax = plt.subplots(figsize=(9, 5.5))
im = ax.imshow(zexpr.values, cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
ax.set_xticks(range(len(marker_genes))); ax.set_xticklabels(marker_genes, rotation=90, fontsize=7)
ax.set_yticks(range(len(zexpr))); ax.set_yticklabels(zexpr.index, fontsize=7)
ax.set_title("Marker expression by subset (z-scored mean, reprocessed clustering)", loc="left")
cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
cbar.set_label("z-score", fontsize=7)
fig.tight_layout()
fig.savefig("marker_heatmap.png", dpi=200, bbox_inches="tight")
print("saved marker_heatmap.png")

# --- 8b. Dot plot: size = % expressing, color = mean expression among all cells ---
fig, ax = plt.subplots(figsize=(9, 5.5))
n_genes, n_subsets = len(marker_genes), len(mean_expr)
xs, ys, sizes, colors = [], [], [], []
for i, s in enumerate(mean_expr.index):
    for j, g in enumerate(marker_genes):
        xs.append(j); ys.append(i)
        sizes.append(pct_expr.loc[s, g] * 6)
        colors.append(mean_expr.loc[s, g])
sc_plot = ax.scatter(xs, ys, s=sizes, c=colors, cmap="viridis", edgecolors="black", linewidths=0.3)
ax.set_xticks(range(n_genes)); ax.set_xticklabels(marker_genes, rotation=90, fontsize=7)
ax.set_yticks(range(n_subsets)); ax.set_yticklabels(mean_expr.index, fontsize=7)
ax.invert_yaxis()
ax.set_title("Marker expression dot plot (reprocessed clustering)", loc="left")
for pct_val in [10, 30, 60]:
    ax.scatter([], [], s=pct_val * 6, c="grey", edgecolors="black", linewidths=0.3, label=f"{pct_val}%")
ax.legend(title="% expressing", loc="upper left", bbox_to_anchor=(1.12, 1.0), fontsize=7, title_fontsize=7.5, frameon=False)
cbar = fig.colorbar(sc_plot, ax=ax, fraction=0.03, pad=0.09)
cbar.set_label("mean expr (all cells)", fontsize=7)
fig.tight_layout()
fig.savefig("marker_dotplot.png", dpi=200, bbox_inches="tight")
print("saved marker_dotplot.png")
