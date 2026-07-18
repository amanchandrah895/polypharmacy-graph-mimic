#!/usr/bin/env python3
"""
Confirmatory replication: second, non-overlapping 10k subsample, delirium outcome only.

Purpose: the headline claim (node2vec embeddings beat a leak-free tabular
baseline for delirium, AUROC delta +0.0178, 95% CI [0.0112,0.0249], p<0.0001)
was measured on ONE random 10k subsample of the 68,013-admission eligible
cohort. A reviewer can reasonably ask whether that generalizes. This script
draws a SECOND subsample with zero overlap with the first, rebuilds the
co-administration graph and node2vec embeddings from scratch on that
independent data (not reusing the original graph/embeddings -- that would be
testing on the same graph, not a real replication), and re-runs the same
paired-bootstrap significance test.

Only delirium is replicated here (the headline claim) -- AKI/bleeding, the
full 4-classifier grid, graph-theoretic features, and SHAP are out of scope
for this confirmatory check to keep runtime/cost down.
"""
import os
import re
import json
import pickle
import warnings
from itertools import combinations
from collections import defaultdict

import numpy as np
import pandas as pd
import networkx as nx
from tqdm.auto import tqdm
from google.cloud import bigquery

warnings.filterwarnings("ignore")

PROJECT_ID = "unified-v2-494419"
client = bigquery.Client(project=PROJECT_ID)
print(f"BigQuery client initialized for billing project: {PROJECT_ID}")

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ------------------------------------------------------------------
# 1. Rebuild the full eligible cohort (same query as the original run --
#    deterministic given the same filters, so this reproduces the same
#    68,013-admission frame without needing to cache it beforehand).
# ------------------------------------------------------------------
HOSP_DS = "mimiciv_3_1_hosp"
ICU_DS = "mimiciv_3_1_icu"

cohort_query = f"""
WITH first_icu AS (
  SELECT
    i.subject_id, i.hadm_id, i.stay_id, i.intime, i.outtime,
    TIMESTAMP_DIFF(i.outtime, i.intime, HOUR) AS icu_los_hours,
    ROW_NUMBER() OVER (PARTITION BY i.hadm_id ORDER BY i.intime ASC) AS icu_rank
  FROM `physionet-data.{ICU_DS}.icustays` i
)
SELECT
  p.subject_id, p.gender, p.anchor_age, p.anchor_year,
  a.hadm_id, a.admittime, a.dischtime, a.deathtime, a.race,
  f.stay_id, f.intime, f.outtime, f.icu_los_hours
FROM `physionet-data.{HOSP_DS}.patients` p
JOIN `physionet-data.{HOSP_DS}.admissions` a ON p.subject_id = a.subject_id
JOIN first_icu f ON a.hadm_id = f.hadm_id AND f.icu_rank = 1
WHERE p.anchor_age >= 18
"""
eligible_df = client.query(cohort_query).to_dataframe()
eligible_df = eligible_df[eligible_df["icu_los_hours"] >= 24].copy()
print(f"Full eligible cohort (adult, first ICU stay, LOS>=24h): {eligible_df['hadm_id'].nunique()}")

# ------------------------------------------------------------------
# 2. Draw a SECOND subsample, zero overlap with the original.
# ------------------------------------------------------------------
original_cohort = pd.read_csv("results/cohort.csv")
original_hadm_ids = set(original_cohort["hadm_id"].unique())
print(f"Original sample size (for exclusion): {len(original_hadm_ids)}")

non_overlapping_pool = eligible_df[~eligible_df["hadm_id"].isin(original_hadm_ids)].copy()
print(f"Non-overlapping pool available: {non_overlapping_pool['hadm_id'].nunique()}")

SUBSAMPLE2_N = 10_000
SUBSAMPLE2_SEED = 123
cohort_df = non_overlapping_pool.sample(n=SUBSAMPLE2_N, random_state=SUBSAMPLE2_SEED).copy()
print(f"Second subsample drawn: {cohort_df['hadm_id'].nunique()} admissions "
      f"(seed={SUBSAMPLE2_SEED}, zero overlap with original sample verified: "
      f"{len(set(cohort_df['hadm_id']) & original_hadm_ids) == 0})")

# ------------------------------------------------------------------
# 3. Drug normalization (identical logic to the original pipeline)
# ------------------------------------------------------------------
BRAND_TO_GENERIC = {
    "tylenol": "acetaminophen", "toradol": "ketorolac", "lasix": "furosemide",
    "zofran": "ondansetron", "protonix": "pantoprazole", "pepcid": "famotidine",
    "coumadin": "warfarin", "lopressor": "metoprolol", "levophed": "norepinephrine",
    "narcan": "naloxone", "ativan": "lorazepam", "versed": "midazolam",
    "haldol": "haloperidol", "benadryl": "diphenhydramine",
    "zosyn": "piperacillin-tazobactam", "vanco": "vancomycin",
    "neurontin": "gabapentin", "cardizem": "diltiazem",
}
DOSE_UNIT_RE = re.compile(r"\b\d+(\.\d+)?\s?(mg|mcg|g|ml|units?|meq|%)\b", re.IGNORECASE)
FORM_ROUTE_RE = re.compile(
    r"\b(tablet|tab|capsule|cap|injection|inj|solution|soln|suspension|susp|"
    r"syrup|cream|ointment|patch|drip|bag|vial|syringe|elixir|oral|iv|po|sl|"
    r"extended release|er|xr|sr|dr)\b", re.IGNORECASE,
)
PUNCT_RE = re.compile(r"[^a-z0-9\-\s]")
WS_RE = re.compile(r"\s+")

def normalize_drug_name(raw):
    if raw is None:
        return None
    s = raw.lower().strip()
    s = DOSE_UNIT_RE.sub(" ", s)
    s = FORM_ROUTE_RE.sub(" ", s)
    s = PUNCT_RE.sub(" ", s)
    s = WS_RE.sub(" ", s).strip()
    if s in BRAND_TO_GENERIC:
        s = BRAND_TO_GENERIC[s]
    for brand, generic in BRAND_TO_GENERIC.items():
        if brand in s:
            s = generic
            break
    return s if s else None

# ------------------------------------------------------------------
# 4. Stream prescriptions for the second subsample (same memory-safe pattern)
# ------------------------------------------------------------------
hadm_ids = cohort_df["hadm_id"].unique().tolist()
hadm_df = pd.DataFrame({"hadm_id": hadm_ids})
temp_table_id = f"{PROJECT_ID}.temp_polypharm.replicate2_hadm_ids"

client.query("CREATE SCHEMA IF NOT EXISTS temp_polypharm").result()
load_job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
job = client.load_table_from_dataframe(hadm_df, temp_table_id, job_config=load_job_config)
job.result()

check_df = client.query(
    f"SELECT COUNT(*) AS n, COUNT(DISTINCT hadm_id) AS n_distinct FROM `{temp_table_id}`"
).to_dataframe()
assert int(check_df["n"][0]) == int(check_df["n_distinct"][0]) == len(hadm_df)
print(f"Temp table verified: {len(hadm_df)} rows, all distinct.")

rx_query = f"""
SELECT r.subject_id, r.hadm_id, r.drug, r.starttime
FROM `physionet-data.{HOSP_DS}.prescriptions` r
JOIN `{temp_table_id}` c ON r.hadm_id = c.hadm_id
"""
count_query = f"""
SELECT COUNT(*) AS n
FROM `physionet-data.{HOSP_DS}.prescriptions` r
JOIN `{temp_table_id}` c ON r.hadm_id = c.hadm_id
"""
expected_total_rows = int(client.query(count_query).to_dataframe()["n"][0])
print(f"Expected prescription rows: {expected_total_rows}")

query_job = client.query(rx_query)
page_iter = query_job.result(page_size=100_000).to_dataframe_iterable()

normalized_chunks = []
with tqdm(total=expected_total_rows, desc="Streaming prescriptions", unit="rows", unit_scale=True) as pbar:
    for page_df in page_iter:
        page_df = page_df.copy()
        page_df["subject_id"] = page_df["subject_id"].astype("int32")
        page_df["hadm_id"] = page_df["hadm_id"].astype("int32")
        page_df["generic_drug"] = page_df["drug"].apply(normalize_drug_name)
        page_df = page_df.dropna(subset=["generic_drug"])
        page_df = page_df[page_df["generic_drug"] != ""]
        normalized_chunks.append(page_df)
        pbar.update(len(page_df))

rx_df = pd.concat(normalized_chunks, ignore_index=True)
del normalized_chunks
print(f"Normalized prescription rows: {len(rx_df)}")

# ------------------------------------------------------------------
# 5. >=3 distinct drugs filter
# ------------------------------------------------------------------
drug_counts = rx_df.groupby("hadm_id")["generic_drug"].nunique()
keep_hadm = drug_counts[drug_counts >= 3].index
cohort_df = cohort_df[cohort_df["hadm_id"].isin(keep_hadm)].copy()
rx_df = rx_df[rx_df["hadm_id"].isin(keep_hadm)].copy()
print(f"Cohort after >=3-drug filter: {cohort_df['hadm_id'].nunique()}")

# ------------------------------------------------------------------
# 6. Delirium label (drug proxy AND CAM-ICU positive -- same logic/itemids)
# ------------------------------------------------------------------
camicu_query = f"""
SELECT itemid, label, category
FROM `physionet-data.{ICU_DS}.d_items`
WHERE LOWER(label) LIKE '%cam-icu%' OR LOWER(label) LIKE '%cam icu%'
   OR LOWER(label) LIKE '%confusion assessment%'
"""
camicu_items = client.query(camicu_query).to_dataframe()
print(f"CAM-ICU items found: {len(camicu_items)}")

DELIRIUM_DRUG_KEYWORDS = [
    "haloperidol", "quetiapine", "olanzapine", "risperidone", "ziprasidone",
    "lorazepam", "midazolam", "diazepam",
]
delirium_drug_hadm = rx_df[
    rx_df["generic_drug"].str.contains("|".join(DELIRIUM_DRUG_KEYWORDS), na=False)
]["hadm_id"].unique()

if len(camicu_items) > 0:
    itemids = camicu_items["itemid"].tolist()
    ids_str = ",".join(str(i) for i in itemids)
    cam_query = f"""
    SELECT ce.stay_id, ce.value
    FROM `physionet-data.{ICU_DS}.chartevents` ce
    WHERE ce.itemid IN ({ids_str})
      AND ce.stay_id IN (SELECT stay_id FROM `{temp_table_id}` c
                         JOIN `physionet-data.{ICU_DS}.icustays` i ON c.hadm_id = i.hadm_id)
    """
    cam_df = client.query(cam_query).to_dataframe()
    positive_values = {"positive", "1", "1.0", "yes"}
    cam_df["is_positive"] = cam_df["value"].astype(str).str.lower().isin(positive_values)
    camicu_positive_stays = set(cam_df.loc[cam_df["is_positive"], "stay_id"].unique())
    print(f"CAM-ICU positive stays: {len(camicu_positive_stays)}")
else:
    camicu_positive_stays = set()

cohort_df["delirium_label"] = (
    cohort_df["hadm_id"].isin(delirium_drug_hadm)
    & (cohort_df["stay_id"].isin(camicu_positive_stays) if camicu_positive_stays else True)
).astype(int)
print(f"Delirium-proxy positive: {cohort_df['delirium_label'].sum()} / {len(cohort_df)} "
      f"({100*cohort_df['delirium_label'].mean():.1f}%)")

cohort_df.to_csv(os.path.join(RESULTS_DIR, "cohort_subsample2.csv"), index=False)
rx_df.to_csv(os.path.join(RESULTS_DIR, "prescriptions_normalized_subsample2.csv"), index=False)

# ------------------------------------------------------------------
# 7. Build population co-administration graph (fresh, from subsample 2 only)
# ------------------------------------------------------------------
rx_df["starttime"] = pd.to_datetime(rx_df["starttime"])
rx_df["admin_day"] = rx_df["starttime"].dt.date
stay_drug_days = rx_df.groupby(["hadm_id", "admin_day"])["generic_drug"].apply(set)

pair_stays = defaultdict(set)
for (hadm_id, day), drugs in stay_drug_days.items():
    for u, v in combinations(sorted(drugs), 2):
        pair_stays[(u, v)].add(hadm_id)

G_weighted = nx.Graph()
for (u, v), stays in pair_stays.items():
    G_weighted.add_edge(u, v, weight=len(stays))
print(f"Graph (subsample 2): {G_weighted.number_of_nodes()} nodes, {G_weighted.number_of_edges()} edges")

hadm_drugs = rx_df.groupby("hadm_id")["generic_drug"].apply(lambda s: sorted(set(s)))

# ------------------------------------------------------------------
# 8. node2vec (same hyperparameters as the original run)
# ------------------------------------------------------------------
from node2vec import Node2Vec

N2V_DIMENSIONS = 64
node2vec_model = Node2Vec(
    G_weighted, dimensions=N2V_DIMENSIONS, walk_length=40, num_walks=20,
    weight_key="weight", workers=4,
)
n2v_fitted = node2vec_model.fit(window=10, min_count=1, batch_words=4)
drug_embeddings = {drug: n2v_fitted.wv[drug] for drug in G_weighted.nodes()}
print(f"node2vec trained on subsample 2: {len(drug_embeddings)} drug embeddings")

def patient_embedding_feature(drugs):
    vecs = [drug_embeddings[d] for d in drugs if d in drug_embeddings]
    return np.mean(vecs, axis=0) if vecs else np.zeros(N2V_DIMENSIONS)

emb_rows = []
for hadm_id, drugs in hadm_drugs.items():
    emb_rows.append({"hadm_id": hadm_id, **{f"emb_{i}": v for i, v in enumerate(patient_embedding_feature(drugs))}})
embedding_features_df = pd.DataFrame(emb_rows).set_index("hadm_id")
print(f"Embedding features (subsample 2): {embedding_features_df.shape}")

# ------------------------------------------------------------------
# 9. Tabular baseline (leak-free from the start, since we already know which
#    columns to exclude for delirium)
# ------------------------------------------------------------------
diag_query = f"""
SELECT hadm_id, COUNT(DISTINCT icd_code) AS n_diagnoses
FROM `physionet-data.{HOSP_DS}.diagnoses_icd`
WHERE hadm_id IN (SELECT hadm_id FROM `{temp_table_id}`)
GROUP BY hadm_id
"""
diag_df = client.query(diag_query).to_dataframe().set_index("hadm_id")

drug_onehot = pd.crosstab(rx_df["hadm_id"], rx_df["generic_drug"])
drug_onehot = (drug_onehot > 0).astype(int)

demo_df = cohort_df.set_index("hadm_id")[["anchor_age", "gender"]].copy()
demo_df["gender_male"] = (demo_df["gender"] == "M").astype(int)
demo_df = demo_df.drop(columns=["gender"])
demo_df = demo_df.join(diag_df, how="left").fillna({"n_diagnoses": 0})
tabular_features_df = demo_df.join(drug_onehot, how="left").fillna(0)

leaky_cols = [c for c in tabular_features_df.columns if any(kw in c for kw in DELIRIUM_DRUG_KEYWORDS)]
tabular_clean = tabular_features_df.drop(columns=leaky_cols)
print(f"Tabular features (subsample 2, leak-free): {tabular_clean.shape} "
      f"({len(leaky_cols)} delirium-defining columns excluded)")

# ------------------------------------------------------------------
# 10. Paired bootstrap significance test (identical method to the original)
# ------------------------------------------------------------------
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import xgboost as xgb

y_full = cohort_df.set_index("hadm_id")["delirium_label"]
common_idx = sorted(set(embedding_features_df.index) & set(tabular_clean.index) & set(y_full.index))
y = y_full.loc[common_idx].values
X_emb = embedding_features_df.loc[common_idx].values
X_tab = tabular_clean.loc[common_idx].values

RANDOM_STATE = 42
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
fold_assignments = list(skf.split(X_emb, y))

def get_xgb():
    return xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                              eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1)

def oof_predict(X, folds):
    oof = np.zeros(len(y))
    for train_idx, test_idx in folds:
        scaler = StandardScaler().fit(X[train_idx])
        model = get_xgb()
        model.fit(scaler.transform(X[train_idx]), y[train_idx])
        oof[test_idx] = model.predict_proba(scaler.transform(X[test_idx]))[:, 1]
    return oof

oof_emb = oof_predict(X_emb, fold_assignments)
oof_tab = oof_predict(X_tab, fold_assignments)

auroc_emb = roc_auc_score(y, oof_emb)
auroc_tab = roc_auc_score(y, oof_tab)
print(f"\n[SUBSAMPLE 2] Embedding AUROC: {auroc_emb:.4f}")
print(f"[SUBSAMPLE 2] Tabular (leak-free) AUROC: {auroc_tab:.4f}")
print(f"[SUBSAMPLE 2] Observed delta (emb - tab): {auroc_emb - auroc_tab:.4f}")

rng = np.random.RandomState(RANDOM_STATE)
n = len(y)
deltas = []
for _ in range(2000):
    idx = rng.randint(0, n, n)
    if len(np.unique(y[idx])) < 2:
        continue
    deltas.append(roc_auc_score(y[idx], oof_emb[idx]) - roc_auc_score(y[idx], oof_tab[idx]))
deltas = np.array(deltas)
ci_lo, ci_hi = np.percentile(deltas, [2.5, 97.5])
p_value = 2 * min((deltas > 0).mean(), (deltas < 0).mean())

print(f"[SUBSAMPLE 2] Paired bootstrap 95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]  (excludes 0: {ci_lo > 0 or ci_hi < 0})")
print(f"[SUBSAMPLE 2] Approx two-sided p: {p_value:.4f}")

original = json.load(open("results/delirium_paired_significance_test.json"))
result = {
    "subsample": 2,
    "seed": SUBSAMPLE2_SEED,
    "n_admissions": len(cohort_df),
    "overlap_with_original_sample": len(set(cohort_df["hadm_id"]) & original_hadm_ids),
    "auroc_embedding": float(auroc_emb),
    "auroc_tabular_leakfree": float(auroc_tab),
    "observed_delta": float(auroc_emb - auroc_tab),
    "delta_ci_95_low": float(ci_lo),
    "delta_ci_95_high": float(ci_hi),
    "ci_excludes_zero": bool(ci_lo > 0 or ci_hi < 0),
    "approx_two_sided_p": float(p_value),
    "original_subsample_delta": original["observed_delta"],
    "original_subsample_ci": [original["delta_ci_95_low"], original["delta_ci_95_high"]],
    "effect_direction_replicated": bool((auroc_emb - auroc_tab) > 0),
}
with open("results/delirium_replication_subsample2.json", "w") as f:
    json.dump(result, f, indent=2)

print("\n=== REPLICATION SUMMARY ===")
print(f"Original subsample:  delta=+{original['observed_delta']:.4f}  CI=[{original['delta_ci_95_low']:.4f},{original['delta_ci_95_high']:.4f}]")
print(f"Second subsample:    delta=+{auroc_emb-auroc_tab:.4f}  CI=[{ci_lo:.4f},{ci_hi:.4f}]")
print(f"Effect direction replicated: {result['effect_direction_replicated']}")
print("Saved results/delirium_replication_subsample2.json")
