#!/usr/bin/env python3
"""
Generate all figures (vector PDFs) and LaTeX tables (booktabs) for the paper
directly from results/ and results_aggregate/ -- numbers in the paper text
must trace back to these files, never hand-typed, so table/text numbers can't
drift apart from the underlying data.
"""
import json
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import networkx as nx
import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
AGG_DIR = os.path.join(REPO_ROOT, "results_aggregate")
FIG_DIR = os.path.join(REPO_ROOT, "latex", "figures")
TAB_DIR = os.path.join(REPO_ROOT, "latex", "tables")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)

plt.rcParams.update({
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})

OUTCOME_LABELS = {
    "aki_label": "AKI",
    "delirium_label": "Delirium (proxy)",
    "bleeding_label": "Bleeding (proxy)",
}
FEATURE_SET_LABELS = {
    "embedding": "Graph embedding",
    "graph_theoretic": "Graph-theoretic",
    "tabular": "Tabular baseline",
}

# ============================================================
# Figure 1: Cohort attrition flow
# ============================================================
attrition = pd.read_csv(os.path.join(AGG_DIR, "cohort_attrition.csv"))
fig, ax = plt.subplots(figsize=(6.5, 2.6))
steps = attrition["step"].tolist()
ns = attrition["n"].tolist()
y_pos = np.arange(len(steps))[::-1]
ax.barh(y_pos, ns, color="#4C72B0", height=0.5)
for y, n in zip(y_pos, ns):
    ax.text(n, y, f"  {n:,}", va="center", fontsize=8)
wrapped = [s if len(s) < 38 else s[:35] + "..." for s in steps]
ax.set_yticks(y_pos)
ax.set_yticklabels(wrapped, fontsize=7.5)
ax.set_xlabel("Admissions (n)")
ax.set_title("Cohort attrition")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "cohort_attrition.pdf"))
plt.close(fig)
print("Saved cohort_attrition.pdf")

# ============================================================
# Figure 2: Population polypharmacy network (top-N nodes by degree)
# ============================================================
with open(os.path.join(RESULTS_DIR, "population_graphs.pkl"), "rb") as f:
    graph_bundle = pickle.load(f)
G = graph_bundle["G_weighted"]

TOP_N_NODES = 60
degree_dict = dict(G.degree())
top_nodes = [n for n, _ in sorted(degree_dict.items(), key=lambda kv: -kv[1])[:TOP_N_NODES]]
subG = G.subgraph(top_nodes).copy()

fig, ax = plt.subplots(figsize=(6.8, 6.8))
pos = nx.spring_layout(subG, seed=42, k=0.5)
node_sizes = [80 + 12 * subG.degree(n) for n in subG.nodes()]
edge_weights = [subG[u][v]["weight"] for u, v in subG.edges()]
max_w = max(edge_weights) if edge_weights else 1
edge_widths = [0.3 + 2.5 * (w / max_w) for w in edge_weights]

nx.draw_networkx_edges(subG, pos, width=edge_widths, alpha=0.25, edge_color="#888888", ax=ax)
nx.draw_networkx_nodes(subG, pos, node_size=node_sizes, node_color="#4C72B0", alpha=0.85, ax=ax)
labels = {n: n if len(n) < 16 else n[:14] + "..." for n in subG.nodes()}
nx.draw_networkx_labels(subG, pos, labels=labels, font_size=5.5, ax=ax)
ax.set_title(f"Population co-administration network (top {TOP_N_NODES} drugs by degree)")
ax.axis("off")
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "population_network.pdf"))
plt.close(fig)
print("Saved population_network.pdf")

# ============================================================
# Figure 3: Degree distribution (log-log)
# ============================================================
deg_df = pd.read_csv(os.path.join(AGG_DIR, "degree_distribution_full.csv"))
degree_col = deg_df.columns[-1]
degrees = deg_df[degree_col].values
degrees = degrees[degrees > 0]

fig, ax = plt.subplots(figsize=(3.4, 2.8))
ax.hist(degrees, bins=np.logspace(np.log10(degrees.min()), np.log10(degrees.max()), 30),
        color="#4C72B0", edgecolor="white", linewidth=0.3)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Node degree (log scale)")
ax.set_ylabel("Count (log scale)")
ax.set_title("Degree distribution")
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "degree_distribution.pdf"))
plt.close(fig)
print("Saved degree_distribution.pdf")

# ============================================================
# Figure 4: Grouped bar -- AUROC by feature-set x outcome, with bootstrap CI
# ============================================================
model_results = pd.read_csv(os.path.join(AGG_DIR, "model_results.csv"))
best = (
    model_results.sort_values("auroc", ascending=False)
    .groupby(["outcome", "feature_set"])
    .first()
    .reset_index()
)

outcomes = ["aki_label", "delirium_label", "bleeding_label"]
feature_sets = ["embedding", "graph_theoretic", "tabular"]
colors = {"embedding": "#4C72B0", "graph_theoretic": "#DD8452", "tabular": "#55A868"}

fig, ax = plt.subplots(figsize=(6.5, 3.2))
x = np.arange(len(outcomes))
width = 0.25
for i, fs in enumerate(feature_sets):
    vals, los, his = [], [], []
    for oc in outcomes:
        row = best[(best.outcome == oc) & (best.feature_set == fs)].iloc[0]
        vals.append(row["auroc"])
        los.append(row["auroc"] - row["auroc_ci_low"])
        his.append(row["auroc_ci_high"] - row["auroc"])
    offset = (i - 1) * width
    ax.bar(x + offset, vals, width, yerr=[los, his], capsize=2.5,
           label=FEATURE_SET_LABELS[fs], color=colors[fs])

ax.set_xticks(x)
ax.set_xticklabels([OUTCOME_LABELS[o] for o in outcomes])
ax.set_ylabel("AUROC (best model per feature set)")
ax.set_ylim(0.5, 1.0)
ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.7)
ax.legend(fontsize=7.5, loc="upper right", frameon=False)
ax.set_title("Best-model AUROC by feature set and outcome (bootstrap 95% CI)")
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "auroc_comparison.pdf"))
plt.close(fig)
print("Saved auroc_comparison.pdf")

# ============================================================
# Figure 5: Top-15 risk-associated drugs per outcome (horizontal bars)
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(9.5, 4.2))
for ax, oc in zip(axes, outcomes):
    df = pd.read_csv(os.path.join(AGG_DIR, f"top_risk_drugs_{oc}.csv")).sort_values(
        "mean_risk_contribution"
    )
    y_pos = np.arange(len(df))
    ax.barh(y_pos, df["mean_risk_contribution"], color=colors["embedding"])
    ax.set_yticks(y_pos)
    labels = [d if len(d) < 22 else d[:19] + "..." for d in df["drug"]]
    ax.set_yticklabels(labels, fontsize=6.5)
    ax.set_xlabel("Mean risk contribution\n(perturbation delta)", fontsize=7.5)
    ax.set_title(OUTCOME_LABELS[oc], fontsize=8.5)
fig.suptitle("Top risk-associated drugs per outcome ($\\geq$10 high-risk patients/drug)", fontsize=9)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(os.path.join(FIG_DIR, "top_risk_drugs.pdf"))
plt.close(fig)
print("Saved top_risk_drugs.pdf")

# ============================================================
# Table 1: Cohort attrition (booktabs)
# ============================================================
def escape(s):
    return str(s).replace("_", "\\_").replace("%", "\\%").replace("&", "\\&")

with open(os.path.join(TAB_DIR, "table_attrition.tex"), "w") as f:
    f.write("\\begin{table}[t]\n\\centering\n\\caption{Cohort attrition.}\n\\label{tab:attrition}\n")
    f.write("\\begin{tabular}{lr}\n\\toprule\n")
    f.write("Filtering step & $n$ (admissions) \\\\\n\\midrule\n")
    for _, row in attrition.iterrows():
        f.write(f"{escape(row['step'])} & {int(row['n']):,} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
print("Saved table_attrition.tex")

# ============================================================
# Table 2: Full model results grid (booktabs)
# ============================================================
with open(os.path.join(TAB_DIR, "table_model_results.tex"), "w") as f:
    f.write("\\begin{table*}[t]\n\\centering\n")
    f.write("\\caption{AUROC (95\\% bootstrap CI), AUPRC, F1, and Brier score for every "
            "outcome $\\times$ feature-set $\\times$ model combination. "
            "delirium\\_label / tabular rows are the leakage-corrected refit "
            "(see Section~\\ref{sec:methods}).}\n")
    f.write("\\label{tab:model_results}\n")
    f.write("\\begin{tabular}{llllllll}\n\\toprule\n")
    f.write("Outcome & Feature set & Model & AUROC & 95\\% CI & AUPRC & F1 & Brier \\\\\n\\midrule\n")
    for oc in outcomes:
        for fs in feature_sets:
            sub = model_results[(model_results.outcome == oc) & (model_results.feature_set == fs)]
            sub = sub.sort_values("auroc", ascending=False)
            for i, (_, row) in enumerate(sub.iterrows()):
                oc_label = OUTCOME_LABELS[oc] if i == 0 else ""
                fs_label = FEATURE_SET_LABELS[fs] if i == 0 else ""
                model_name = escape(row["model"]).replace("\\_", " ")
                f.write(
                    f"{oc_label} & {fs_label} & {model_name} & "
                    f"{row['auroc']:.3f} & [{row['auroc_ci_low']:.3f}, {row['auroc_ci_high']:.3f}] & "
                    f"{row['auprc']:.3f} & {row['f1']:.3f} & {row['brier']:.3f} \\\\\n"
                )
        f.write("\\addlinespace\n")
    f.write("\\bottomrule\n\\end{tabular}\n\\end{table*}\n")
print("Saved table_model_results.tex")

# ============================================================
# Table 3: Graph summary statistics
# ============================================================
with open(os.path.join(AGG_DIR, "graph_stats.json")) as f:
    gstats = json.load(f)

with open(os.path.join(TAB_DIR, "table_graph_stats.tex"), "w") as f:
    f.write("\\begin{table}[t]\n\\centering\n\\caption{Population co-administration graph summary statistics.}\n")
    f.write("\\label{tab:graph_stats}\n\\begin{tabular}{lr}\n\\toprule\n")
    f.write("Statistic & Value \\\\\n\\midrule\n")
    f.write(f"Nodes (distinct drugs) & {gstats['node_count']:,} \\\\\n")
    f.write(f"Edges (co-administered pairs) & {gstats['edge_count']:,} \\\\\n")
    f.write(f"Density & {gstats['density']:.4f} \\\\\n")
    f.write(f"Avg.\\ clustering coefficient & {gstats['avg_clustering_coefficient']:.4f} \\\\\n")
    f.write(f"Connected components & {gstats['num_connected_components']} \\\\\n")
    dd = gstats["degree_distribution_summary"]
    f.write(f"Mean degree & {dd['mean']:.1f} \\\\\n")
    f.write(f"Median degree & {dd['median']:.1f} \\\\\n")
    f.write(f"Max degree & {dd['max']} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
print("Saved table_graph_stats.tex")

# ============================================================
# Table 4: Delirium leakage correction + paired significance test
# ============================================================
with open(os.path.join(AGG_DIR, "delirium_paired_significance_test.json")) as f:
    sig = json.load(f)

with open(os.path.join(TAB_DIR, "table_significance.tex"), "w") as f:
    f.write("\\begin{table}[t]\n\\centering\n")
    f.write("\\caption{Paired bootstrap significance test: node2vec embedding vs.\\ "
            "leakage-corrected tabular baseline, delirium outcome, identical "
            "5-fold CV splits, 2{,}000 resamples.}\n")
    f.write("\\label{tab:significance}\n\\begin{tabular}{lr}\n\\toprule\n")
    f.write("Quantity & Value \\\\\n\\midrule\n")
    f.write(f"AUROC (embedding) & {sig['auroc_embedding']:.4f} \\\\\n")
    f.write(f"AUROC (tabular, leak-free) & {sig['auroc_tabular_leakfree']:.4f} \\\\\n")
    f.write(f"Observed $\\Delta$AUROC & {sig['observed_delta']:.4f} \\\\\n")
    f.write(f"95\\% CI on $\\Delta$ & [{sig['delta_ci_95_low']:.4f}, {sig['delta_ci_95_high']:.4f}] \\\\\n")
    f.write(f"$p$ (approx., two-sided) & {sig['approx_two_sided_p']:.4f} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
print("Saved table_significance.tex")

# ============================================================
# Replication table (only if the confirmatory second-subsample run finished)
# ============================================================
repl_path = os.path.join(RESULTS_DIR, "delirium_replication_subsample2.json")
if os.path.exists(repl_path):
    with open(repl_path) as f:
        repl = json.load(f)
    with open(os.path.join(TAB_DIR, "table_replication.tex"), "w") as f:
        f.write("\\begin{table}[t]\n\\centering\n")
        f.write("\\caption{Confirmatory replication on an independent, "
                "zero-overlap second 10{,}000-admission subsample "
                "(graph and embeddings rebuilt from scratch).}\n")
        f.write("\\label{tab:replication}\n\\begin{tabular}{lrr}\n\\toprule\n")
        f.write("Quantity & Original sample & Replication \\\\\n\\midrule\n")
        f.write(f"$\\Delta$AUROC (emb.\\ $-$ tab.) & "
                f"{repl['original_subsample_delta']:.4f} & {repl['observed_delta']:.4f} \\\\\n")
        f.write(f"95\\% CI & [{repl['original_subsample_ci'][0]:.4f}, {repl['original_subsample_ci'][1]:.4f}] "
                f"& [{repl['delta_ci_95_low']:.4f}, {repl['delta_ci_95_high']:.4f}] \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    print("Saved table_replication.tex (replication run had completed)")
else:
    print("NOTE: delirium_replication_subsample2.json not yet present -- "
          "table_replication.tex NOT generated. Re-run this script after the "
          "replication finishes, or the paper will omit that table.")

print("\nAll available figures/tables generated.")
