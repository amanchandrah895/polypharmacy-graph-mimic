#!/usr/bin/env python3
"""
Polypharmacy risk-network pipeline (INDICon paper) -- local/offline runner.

This is a straight port of notebooks/01_polypharmacy_pipeline.ipynb for running
outside Colab, on your own machine, against your own GCP project. It keeps the
exact same cohort/label/graph/modeling logic (including the streaming-prescriptions
and per-day-co-administration memory fixes already applied to the notebook) --
only the Colab-specific auth cell and the final files.download() call are swapped
for local equivalents.

Prerequisites (one-time, run in Terminal before this script):
    brew install --cask google-cloud-sdk
    gcloud auth application-default login \
        --scopes=https://www.googleapis.com/auth/cloud-platform,\
https://www.googleapis.com/auth/bigquery

    # then create+activate a venv and install deps:
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r notebooks/requirements.txt

Run:
    source .venv/bin/activate
    python3 -u notebooks/run_pipeline_local.py 2>&1 | tee notebooks/run.log

Runs to completion end-to-end (no cell-by-cell interaction); progress prints
and tqdm bars go to stdout, and everything under results/ + results.zip is left
on disk when it finishes -- no download step needed since it already ran locally.
"""

import warnings
warnings.filterwarnings("ignore")

# --- Local auth: uses Application Default Credentials set up via
# `gcloud auth application-default login` (see module docstring). This is the
# direct local equivalent of the Colab `auth.authenticate_user()` cell -- both
# resolve to your own PhysioNet-linked Google identity, which is what
# `physionet-data` authorization is actually tied to.
from google.cloud import bigquery

PROJECT_ID = "unified-v2-494419"  # GCP project used to bill BigQuery queries
client = bigquery.Client(project=PROJECT_ID)

print(f"BigQuery client initialized for billing project: {PROJECT_ID}")


import os
import json
import time
import pickle
import zipfile
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
sns.set_context("notebook")

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

MANIFEST = []  # list of (filename, description) appended to as files are saved

def save_manifest_entry(path, description):
    MANIFEST.append({"file": path, "description": description})

print("Results directory ready:", os.path.abspath(RESULTS_DIR))

# --- Schema resolution: try versioned MIMIC-IV 3.1 schema first, fall back to unversioned ---

def resolve_dataset(candidates, probe_table):
    """Try each candidate dataset id by running a trivial LIMIT 1 query against
    probe_table. Returns the first dataset id that resolves without error."""
    for ds in candidates:
        try:
            q = f"SELECT 1 FROM `physionet-data.{ds}.{probe_table}` LIMIT 1"
            client.query(q).result()
            print(f"Resolved dataset '{ds}' (probe table '{probe_table}' OK)")
            return ds
        except Exception as e:
            print(f"Dataset '{ds}' failed probe on '{probe_table}': {e}")
    raise RuntimeError(f"None of {candidates} resolved for probe table {probe_table}")

HOSP_DS = resolve_dataset(["mimiciv_3_1_hosp", "mimiciv_hosp"], "patients")
ICU_DS = resolve_dataset(["mimiciv_3_1_icu", "mimiciv_icu"], "icustays")

# Derived concept dataset (used later for KDIGO AKI staging); optional.
DERIVED_DS = None
for ds in ["mimiciv_3_1_derived", "mimiciv_derived"]:
    try:
        client.query(f"SELECT 1 FROM `physionet-data.{ds}.kdigo_stages` LIMIT 1").result()
        DERIVED_DS = ds
        print(f"Resolved derived-concepts dataset '{ds}' (kdigo_stages available)")
        break
    except Exception as e:
        print(f"Derived dataset '{ds}' not available: {e}")

if DERIVED_DS is None:
    print("No derived KDIGO table resolved -- will compute AKI manually from labevents.")

print()
print(f"HOSP_DS   = {HOSP_DS}")
print(f"ICU_DS    = {ICU_DS}")
print(f"DERIVED_DS = {DERIVED_DS}")

attrition = []

def log_attrition(step, n, note=""):
    attrition.append({"step": step, "n": n, "note": note})
    print(f"[{len(attrition)}] {step}: n={n}  {note}")

# --- Step 1: base cohort frame (patients x admissions x icustays), adult, first ICU stay ---

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

cohort_df = client.query(cohort_query).to_dataframe()
log_attrition("Adult patients, first ICU stay per admission", cohort_df["hadm_id"].nunique())

# --- Step 2: ICU LOS >= 24h ---

cohort_df = cohort_df[cohort_df["icu_los_hours"] >= 24].copy()
log_attrition("ICU LOS >= 24h", cohort_df["hadm_id"].nunique())

# --- Step 2b: random subsample for compute/cost tractability ---
# The full eligible cohort (68,013 admissions) drives a 7.48M-row prescriptions
# pull plus comparably large labevents/chartevents/inputevents queries, which is
# both expensive to scan in BigQuery and heavy to process locally. We take a
# fixed-seed random subsample so the run is reproducible; this MUST be reported
# in the paper's Methods/Limitations as a subsample of the full eligible MIMIC-IV
# cohort, not the full population.
COHORT_SUBSAMPLE_N = 10_000
COHORT_SUBSAMPLE_SEED = 42

if len(cohort_df) > COHORT_SUBSAMPLE_N:
    cohort_df = cohort_df.sample(n=COHORT_SUBSAMPLE_N, random_state=COHORT_SUBSAMPLE_SEED).copy()
    log_attrition(
        f"Random subsample (seed={COHORT_SUBSAMPLE_SEED}, compute/cost constraint)",
        cohort_df["hadm_id"].nunique(),
        note=f"subsampled from full eligible cohort down to {COHORT_SUBSAMPLE_N}",
    )

import re

BRAND_TO_GENERIC = {
    "tylenol": "acetaminophen",
    "toradol": "ketorolac",
    "lasix": "furosemide",
    "zofran": "ondansetron",
    "protonix": "pantoprazole",
    "pepcid": "famotidine",
    "coumadin": "warfarin",
    "lopressor": "metoprolol",
    "levophed": "norepinephrine",
    "narcan": "naloxone",
    "ativan": "lorazepam",
    "versed": "midazolam",
    "haldol": "haloperidol",
    "benadryl": "diphenhydramine",
    "zosyn": "piperacillin-tazobactam",
    "vanco": "vancomycin",
    "neurontin": "gabapentin",
    "cardizem": "diltiazem",
}

DOSE_UNIT_RE = re.compile(
    r"\b\d+(\.\d+)?\s?(mg|mcg|g|ml|units?|meq|%)\b", re.IGNORECASE
)
FORM_ROUTE_RE = re.compile(
    r"\b(tablet|tab|capsule|cap|injection|inj|solution|soln|suspension|susp|"
    r"syrup|cream|ointment|patch|drip|bag|vial|syringe|elixir|oral|iv|po|sl|"
    r"extended release|er|xr|sr|dr)\b",
    re.IGNORECASE,
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

print(f"Brand->generic synonym dictionary ({len(BRAND_TO_GENERIC)} entries):")
for k, v in BRAND_TO_GENERIC.items():
    print(f"  {k} -> {v}")

# --- Step 3: pull prescriptions for the cohort, normalize drug names ---
# Rewritten for memory safety: the previous version pulled the ENTIRE joined
# prescriptions result into one pandas DataFrame via .to_dataframe() before doing
# any normalization. For this cohort that is 7.48M raw rows -- holding the full
# raw pull AND the intermediate copies .apply()/.dropna()/filtering each create
# (3-4x the base size, momentarily) is what exhausted Colab's RAM. This version:
#   1. only selects columns actually used downstream (drops `route`/`stoptime`,
#      which nothing later in the notebook reads),
#   2. downcasts subject_id/hadm_id to int32,
#   3. streams the BigQuery result page-by-page (to_dataframe_iterable) instead
#      of materializing the full result at once,
#   4. normalizes + drops empty rows immediately per page, so only the smaller
#      normalized pages accumulate in memory,
# with a tqdm progress bar tracking rows-completed/rows-total (not just an
# unbounded page counter) -- a quick COUNT(*) upfront gives the denominator.
# If you still see RAM pressure, lower PAGE_SIZE further (e.g. 50_000) --
# runtime goes up, peak memory goes down.
#
# IMPORTANT: load_table_from_dataframe() defaults to WRITE_APPEND if the temp
# table already exists from a previous run. Re-running this notebook without
# WRITE_TRUNCATE silently accumulates duplicate hadm_id rows in the temp table
# across runs, which fans out the downstream JOIN to many times the correct row
# count (observed: 68,013 distinct hadm_ids but 826,156 accumulated rows after
# repeated runs, fanning a 10k-admission subsample query out to 90M+ joined
# rows). WRITE_TRUNCATE makes every run start from a clean, exact-match temp
# table regardless of prior runs.

from tqdm.auto import tqdm

hadm_ids = cohort_df["hadm_id"].unique().tolist()

hadm_df = pd.DataFrame({"hadm_id": hadm_ids})
temp_table_id = f"{PROJECT_ID}.temp_polypharm.cohort_hadm_ids"

PAGE_SIZE = 100_000  # rows pulled from BigQuery per page/chunk; lower if RAM is still tight

def normalize_and_filter(df):
    df = df.copy()
    df["subject_id"] = df["subject_id"].astype("int32")
    df["hadm_id"] = df["hadm_id"].astype("int32")
    df["generic_drug"] = df["drug"].apply(normalize_drug_name)
    df = df.dropna(subset=["generic_drug"])
    df = df[df["generic_drug"] != ""]
    return df

normalized_chunks = []
total_raw_rows = 0

try:
    client.query("CREATE SCHEMA IF NOT EXISTS temp_polypharm").result()
    load_job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
    job = client.load_table_from_dataframe(hadm_df, temp_table_id, job_config=load_job_config)
    job.result()
    print(f"Loaded {len(hadm_df)} hadm_ids to temp table {temp_table_id} (WRITE_TRUNCATE)")

    # sanity check: temp table row count must exactly match the cohort size, or
    # a stale/duplicated table would silently fan out the join below
    check_df = client.query(
        f"SELECT COUNT(*) AS n, COUNT(DISTINCT hadm_id) AS n_distinct FROM `{temp_table_id}`"
    ).to_dataframe()
    n_rows, n_distinct = int(check_df["n"][0]), int(check_df["n_distinct"][0])
    assert n_rows == n_distinct == len(hadm_df), (
        f"Temp table row count mismatch after WRITE_TRUNCATE: "
        f"{n_rows} rows, {n_distinct} distinct, expected {len(hadm_df)}"
    )
    print(f"Temp table verified: {n_rows} rows, all distinct, matches cohort size.")

    # route/stoptime dropped from SELECT: not read anywhere downstream in this notebook
    rx_query = f"""
    SELECT r.subject_id, r.hadm_id, r.drug, r.starttime
    FROM `physionet-data.{HOSP_DS}.prescriptions` r
    JOIN `{temp_table_id}` c ON r.hadm_id = c.hadm_id
    """

    # quick COUNT(*) upfront so the progress bar can show rows-completed/rows-total
    # (and as a second sanity check on the expected join size before pulling data)
    count_query = f"""
    SELECT COUNT(*) AS n
    FROM `physionet-data.{HOSP_DS}.prescriptions` r
    JOIN `{temp_table_id}` c ON r.hadm_id = c.hadm_id
    """
    expected_total_rows = int(client.query(count_query).to_dataframe()["n"][0])
    print(f"Expected prescription rows for this cohort: {expected_total_rows}")

    query_job = client.query(rx_query)
    page_iter = query_job.result(page_size=PAGE_SIZE).to_dataframe_iterable()

    with tqdm(total=expected_total_rows, desc="Streaming + normalizing prescriptions", unit="rows", unit_scale=True) as pbar:
        for page_df in page_iter:
            total_raw_rows += len(page_df)
            normalized_chunks.append(normalize_and_filter(page_df))
            pbar.update(len(page_df))

except Exception as e:
    print(f"Temp-table join failed ({e}); falling back to chunked IN-clause queries.")
    chunks = [hadm_ids[i:i+5000] for i in range(0, len(hadm_ids), 5000)]
    for chunk in tqdm(chunks, desc="Streaming + normalizing prescriptions (chunked fallback)", unit="chunk"):
        ids_str = ",".join(str(x) for x in chunk)
        q = f"""
        SELECT subject_id, hadm_id, drug, starttime
        FROM `physionet-data.{HOSP_DS}.prescriptions`
        WHERE hadm_id IN ({ids_str})
        """
        chunk_df = client.query(q).to_dataframe()
        total_raw_rows += len(chunk_df)
        normalized_chunks.append(normalize_and_filter(chunk_df))

rx_df = pd.concat(normalized_chunks, ignore_index=True)
del normalized_chunks

print(f"Raw prescription rows pulled: {total_raw_rows}")
print(f"Prescription rows after normalization/drop-empty: {len(rx_df)}")
print(f"Distinct raw drug strings: {rx_df['drug'].nunique()}  ->  distinct generic names: {rx_df['generic_drug'].nunique()}")

# --- Step 4: exclude admissions with < 3 distinct generic drugs ---

drug_counts = rx_df.groupby("hadm_id")["generic_drug"].nunique()
keep_hadm = drug_counts[drug_counts >= 3].index

cohort_df = cohort_df[cohort_df["hadm_id"].isin(keep_hadm)].copy()
rx_df = rx_df[rx_df["hadm_id"].isin(keep_hadm)].copy()

log_attrition(">= 3 distinct generic drugs administered", cohort_df["hadm_id"].nunique())

# --- AKI label: prefer derived KDIGO table, else compute manually from labevents creatinine ---

if DERIVED_DS is not None:
    aki_query = f"""
    SELECT DISTINCT k.stay_id
    FROM `physionet-data.{DERIVED_DS}.kdigo_stages` k
    WHERE k.aki_stage >= 1
    """
    aki_stays = client.query(aki_query).to_dataframe()["stay_id"]
    cohort_df["aki_label"] = cohort_df["stay_id"].isin(aki_stays).astype(int)
    print(f"AKI label computed via derived table `{DERIVED_DS}.kdigo_stages` (KDIGO stage >= 1).")
else:
    print("Computing AKI manually from labevents creatinine using KDIGO criteria:")
    print("  - >=0.3 mg/dL rise within 48h, OR >=1.5x rise from baseline within 7 days")
    stay_ids = cohort_df["stay_id"].tolist()
    cr_query = f"""
    SELECT le.subject_id, le.hadm_id, le.charttime, le.valuenum
    FROM `physionet-data.{HOSP_DS}.labevents` le
    JOIN `physionet-data.{HOSP_DS}.d_labitems` d ON le.itemid = d.itemid
    WHERE LOWER(d.label) LIKE '%creatinine%' AND le.valuenum IS NOT NULL
      AND le.hadm_id IN (SELECT hadm_id FROM `{temp_table_id}`)
    """
    cr_df = client.query(cr_query).to_dataframe()
    cr_df = cr_df.merge(cohort_df[["hadm_id", "stay_id", "intime", "outtime"]], on="hadm_id")
    cr_df["charttime"] = pd.to_datetime(cr_df["charttime"])
    cr_df["intime"] = pd.to_datetime(cr_df["intime"])
    cr_df["outtime"] = pd.to_datetime(cr_df["outtime"])
    cr_df = cr_df[(cr_df["charttime"] >= cr_df["intime"]) & (cr_df["charttime"] <= cr_df["outtime"])]

    def kdigo_flag(g):
        g = g.sort_values("charttime")
        baseline = g["valuenum"].iloc[0]
        vals = g[["charttime", "valuenum"]].values
        for i in range(len(vals)):
            t_i, v_i = vals[i]
            for j in range(i + 1, len(vals)):
                t_j, v_j = vals[j]
                dt_hours = (t_j - t_i).total_seconds() / 3600
                if dt_hours <= 48 and (v_j - v_i) >= 0.3:
                    return 1
                if dt_hours <= 24 * 7 and baseline > 0 and (v_j / baseline) >= 1.5:
                    return 1
        return 0

    aki_flags = cr_df.groupby("stay_id").apply(kdigo_flag)
    cohort_df["aki_label"] = cohort_df["stay_id"].map(aki_flags).fillna(0).astype(int)

print(f"AKI positive: {cohort_df['aki_label'].sum()} / {len(cohort_df)} "
      f"({100*cohort_df['aki_label'].mean():.1f}%)")

# --- Delirium proxy label: antipsychotic/benzodiazepine-for-agitation order AND CAM-ICU positive flag ---
# First, search d_items for CAM-ICU related item labels so itemids are not hardcoded blind.

camicu_items_query = f"""
SELECT itemid, label, category
FROM `physionet-data.{ICU_DS}.d_items`
WHERE LOWER(label) LIKE '%cam-icu%' OR LOWER(label) LIKE '%cam icu%'
   OR LOWER(label) LIKE '%confusion assessment%'
"""
camicu_items = client.query(camicu_items_query).to_dataframe()
print("CAM-ICU related d_items found:")
print(camicu_items)

# Delirium-relevant drugs: antipsychotics + benzodiazepines commonly used for agitation
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
    SELECT ce.stay_id, ce.charttime, ce.value, ce.valuenum
    FROM `physionet-data.{ICU_DS}.chartevents` ce
    WHERE ce.itemid IN ({ids_str})
      AND ce.stay_id IN (SELECT stay_id FROM `{temp_table_id}` c
                         JOIN `physionet-data.{ICU_DS}.icustays` i ON c.hadm_id = i.hadm_id)
    """
    cam_df = client.query(cam_query).to_dataframe()
    positive_values = {"positive", "1", "1.0", "yes"}
    cam_df["is_positive"] = cam_df["value"].astype(str).str.lower().isin(positive_values)
    camicu_positive_stays = set(cam_df.loc[cam_df["is_positive"], "stay_id"].unique())
    print(f"CAM-ICU chartevents rows: {len(cam_df)}; positive stays: {len(camicu_positive_stays)}")
else:
    camicu_positive_stays = set()
    print("No CAM-ICU itemids found in d_items -- delirium label will rely on drug proxy only "
          "(this weakens the label; report as a limitation).")

cohort_df["delirium_label"] = (
    cohort_df["hadm_id"].isin(delirium_drug_hadm)
    & (cohort_df["stay_id"].isin(camicu_positive_stays) if camicu_positive_stays else True)
).astype(int)

print(f"Delirium-proxy positive: {cohort_df['delirium_label'].sum()} / {len(cohort_df)} "
      f"({100*cohort_df['delirium_label'].mean():.1f}%)")

# --- Bleeding proxy label: new PRBC transfusion order in inputevents after ICU day 2 ---

prbc_items_query = f"""
SELECT itemid, label
FROM `physionet-data.{ICU_DS}.d_items`
WHERE LOWER(label) LIKE '%packed red blood cells%' OR LOWER(label) LIKE '%prbc%'
   OR LOWER(label) LIKE '%red blood cells%'
"""
prbc_items = client.query(prbc_items_query).to_dataframe()
print("PRBC-related d_items found:")
print(prbc_items)

itemids = prbc_items["itemid"].tolist()
ids_str = ",".join(str(i) for i in itemids)
prbc_query = f"""
SELECT ie.stay_id, ie.starttime
FROM `physionet-data.{ICU_DS}.inputevents` ie
WHERE ie.itemid IN ({ids_str})
  AND ie.stay_id IN (SELECT stay_id FROM `{temp_table_id}` c
                     JOIN `physionet-data.{ICU_DS}.icustays` i ON c.hadm_id = i.hadm_id)
"""
prbc_df = client.query(prbc_query).to_dataframe()
prbc_df = prbc_df.merge(cohort_df[["stay_id", "intime"]], on="stay_id")
prbc_df["starttime"] = pd.to_datetime(prbc_df["starttime"])
prbc_df["intime"] = pd.to_datetime(prbc_df["intime"])
prbc_df["icu_day"] = (prbc_df["starttime"] - prbc_df["intime"]).dt.total_seconds() / 86400

bleed_stays = set(prbc_df.loc[prbc_df["icu_day"] > 2, "stay_id"].unique())
cohort_df["bleeding_label"] = cohort_df["stay_id"].isin(bleed_stays).astype(int)

print(f"Bleeding-proxy positive: {cohort_df['bleeding_label'].sum()} / {len(cohort_df)} "
      f"({100*cohort_df['bleeding_label'].mean():.1f}%)")

# --- Final attrition entry + save ---

log_attrition("Final cohort (all labels attached)", len(cohort_df))

attrition_df = pd.DataFrame(attrition)
attrition_path = os.path.join(RESULTS_DIR, "cohort_attrition.csv")
attrition_df.to_csv(attrition_path, index=False)
save_manifest_entry(attrition_path, "Cohort attrition table (Table-1 style filtering flow)")

cohort_path = os.path.join(RESULTS_DIR, "cohort.csv")
cohort_df.to_csv(cohort_path, index=False)
save_manifest_entry(cohort_path, "Final cohort frame with demographics + 3 outcome labels")

rx_path = os.path.join(RESULTS_DIR, "prescriptions_normalized.csv")
rx_df.to_csv(rx_path, index=False)
save_manifest_entry(rx_path, "Normalized prescriptions (generic_drug column) for the final cohort")

print(attrition_df)

rx_df["starttime"] = pd.to_datetime(rx_df["starttime"])
rx_df["admin_day"] = rx_df["starttime"].dt.date

# drug -> set of (hadm_id, day) tuples, used to determine same-day co-administration
stay_drug_days = rx_df.groupby(["hadm_id", "admin_day"])["generic_drug"].apply(set)

from itertools import combinations
from collections import defaultdict

pair_weight = defaultdict(int)   # (drug_u, drug_v) -> count of stay-days co-administered... but we want per-stay, not per-day
pair_stays = defaultdict(set)    # (drug_u, drug_v) -> set of hadm_id where co-administered on some overlapping day

for (hadm_id, day), drugs in stay_drug_days.items():
    for u, v in combinations(sorted(drugs), 2):
        pair_stays[(u, v)].add(hadm_id)

# population edge weight = number of distinct stays with >=1 day of co-administration
pop_edges = [(u, v, len(stays)) for (u, v), stays in pair_stays.items()]
print(f"Distinct co-administered drug pairs (population): {len(pop_edges)}")

# --- Build population-level graphs ---

G_weighted = nx.Graph()
for u, v, w in pop_edges:
    G_weighted.add_edge(u, v, weight=w)

G_unweighted = nx.Graph()
for u, v, w in pop_edges:
    G_unweighted.add_edge(u, v)

print(f"Weighted graph:   {G_weighted.number_of_nodes()} nodes, {G_weighted.number_of_edges()} edges")
print(f"Unweighted graph: {G_unweighted.number_of_nodes()} nodes, {G_unweighted.number_of_edges()} edges")

# --- Per-stay graphs (dict: hadm_id -> nx.Graph), used later for patient-level graph-theoretic features ---
# Rewritten for memory/runtime safety: the previous version iterated all
# C(n_drugs_in_stay, 2) pairs per admission and checked membership against the
# population-level pair_stays sets. For admissions with many distinct drugs across
# a long stay this blows up combinatorially (e.g. 100 drugs -> 4,950 pairs per stay)
# and crashed the Colab runtime. This version builds edges directly from
# stay_drug_days (per admission, per calendar day), so the combinatorics are bounded
# by drugs given on a single day (much smaller) rather than drugs given across the
# whole stay.

per_stay_graphs = defaultdict(nx.Graph)

hadm_drugs = rx_df.groupby("hadm_id")["generic_drug"].apply(lambda s: sorted(set(s)))

# add every distinct drug the stay received as a node (even if never co-administered
# on the same calendar day), so downstream subgraph-density/feature code sees the
# full drug set per stay
for hadm_id, drugs in hadm_drugs.items():
    per_stay_graphs[hadm_id].add_nodes_from(drugs)

# accumulate edge weight = number of days that pair was co-administered within the stay
for (hadm_id, day), drugs in stay_drug_days.items():
    if len(drugs) < 2:
        continue
    g = per_stay_graphs[hadm_id]
    for u, v in combinations(sorted(drugs), 2):
        if g.has_edge(u, v):
            g[u][v]["weight"] += 1
        else:
            g.add_edge(u, v, weight=1)

per_stay_graphs = dict(per_stay_graphs)
print(f"Built {len(per_stay_graphs)} per-stay graphs.")

# --- Graph summary statistics (population-level, weighted graph is primary) ---

degree_dict = dict(G_weighted.degree())
degree_series = pd.Series(degree_dict).sort_values(ascending=False)

top20_degree = degree_series.head(20).reset_index()
top20_degree.columns = ["drug", "degree"]

edge_weights = [(u, v, d["weight"]) for u, v, d in G_weighted.edges(data=True)]
edge_weights_sorted = sorted(edge_weights, key=lambda x: -x[2])[:20]
top20_edges = pd.DataFrame(edge_weights_sorted, columns=["drug_u", "drug_v", "co_admin_stays"])

graph_stats = {
    "node_count": G_weighted.number_of_nodes(),
    "edge_count": G_weighted.number_of_edges(),
    "density": nx.density(G_weighted),
    "avg_clustering_coefficient": nx.average_clustering(G_weighted, weight="weight"),
    "num_connected_components": nx.number_connected_components(G_weighted),
    "degree_distribution_summary": {
        "mean": float(degree_series.mean()),
        "median": float(degree_series.median()),
        "max": int(degree_series.max()),
        "min": int(degree_series.min()),
    },
    "top20_highest_degree_drugs": top20_degree.to_dict(orient="records"),
    "top20_highest_weight_edges": top20_edges.to_dict(orient="records"),
    "unweighted_comparison": {
        "node_count": G_unweighted.number_of_nodes(),
        "edge_count": G_unweighted.number_of_edges(),
        "density": nx.density(G_unweighted),
        "avg_clustering_coefficient": nx.average_clustering(G_unweighted),
        "num_connected_components": nx.number_connected_components(G_unweighted),
    },
}

graph_stats_path = os.path.join(RESULTS_DIR, "graph_stats.json")
with open(graph_stats_path, "w") as f:
    json.dump(graph_stats, f, indent=2, default=str)
save_manifest_entry(graph_stats_path, "Population graph summary stats: degree/edge-weight top-20, density, clustering, components")

full_degree_path = os.path.join(RESULTS_DIR, "degree_distribution_full.csv")
degree_series.reset_index().rename(columns={"index": "drug", 0: "degree"}).to_csv(full_degree_path, index=False)
save_manifest_entry(full_degree_path, "Full node degree list (for degree-distribution histogram in paper)")

print(json.dumps({k: v for k, v in graph_stats.items() if not isinstance(v, list)}, indent=2))

# --- Persist graph objects for Section 4/5 and for Phase-2 figure generation ---

graph_pickle_path = os.path.join(RESULTS_DIR, "population_graphs.pkl")
with open(graph_pickle_path, "wb") as f:
    pickle.dump({
        "G_weighted": G_weighted,
        "G_unweighted": G_unweighted,
        "per_stay_graphs": per_stay_graphs,
    }, f)
save_manifest_entry(graph_pickle_path, "Pickled networkx graphs: population weighted/unweighted + per-stay drug subgraphs")

print("Saved graph objects to", graph_pickle_path)

from node2vec import Node2Vec

N2V_DIMENSIONS = 64
N2V_WALK_LENGTH = 40
N2V_NUM_WALKS = 20

node2vec = Node2Vec(
    G_weighted,
    dimensions=N2V_DIMENSIONS,
    walk_length=N2V_WALK_LENGTH,
    num_walks=N2V_NUM_WALKS,
    weight_key="weight",
    workers=4,
)
n2v_model = node2vec.fit(window=10, min_count=1, batch_words=4)

drug_embeddings = {drug: n2v_model.wv[drug] for drug in G_weighted.nodes()}
print(f"Trained node2vec embeddings for {len(drug_embeddings)} drug nodes, dim={N2V_DIMENSIONS}")

emb_pickle_path = os.path.join(RESULTS_DIR, "node2vec_embeddings.pkl")
with open(emb_pickle_path, "wb") as f:
    pickle.dump({
        "embeddings": drug_embeddings,
        "hyperparameters": {
            "dimensions": N2V_DIMENSIONS,
            "walk_length": N2V_WALK_LENGTH,
            "num_walks": N2V_NUM_WALKS,
            "window": 10,
            "p": 1, "q": 1,
        },
    }, f)
save_manifest_entry(emb_pickle_path, "node2vec drug embeddings + hyperparameters used")

# --- Patient-level features ---

def patient_embedding_feature(drugs):
    vecs = [drug_embeddings[d] for d in drugs if d in drug_embeddings]
    if not vecs:
        return np.zeros(N2V_DIMENSIONS)
    return np.mean(vecs, axis=0)

def patient_graph_theoretic_features(hadm_id, drugs):
    weights = []
    for u, v in combinations(drugs, 2):
        if G_weighted.has_edge(u, v):
            weights.append(G_weighted[u][v]["weight"])
    subG = per_stay_graphs.get(hadm_id, nx.Graph())
    density = nx.density(subG) if subG.number_of_nodes() > 1 else 0.0
    return {
        "sum_edge_weight": float(np.sum(weights)) if weights else 0.0,
        "max_edge_weight": float(np.max(weights)) if weights else 0.0,
        "subgraph_density": density,
        "n_drugs": len(drugs),
    }

emb_rows = []
graphstat_rows = []
for hadm_id, drugs in hadm_drugs.items():
    emb_rows.append({"hadm_id": hadm_id, **{f"emb_{i}": v for i, v in enumerate(patient_embedding_feature(drugs))}})
    graphstat_rows.append({"hadm_id": hadm_id, **patient_graph_theoretic_features(hadm_id, drugs)})

embedding_features_df = pd.DataFrame(emb_rows).set_index("hadm_id")
graphstat_features_df = pd.DataFrame(graphstat_rows).set_index("hadm_id")

print("Embedding feature matrix:", embedding_features_df.shape)
print("Graph-theoretic feature matrix:", graphstat_features_df.shape)

# --- Tabular baseline: one-hot drug indicators + age/sex/comorbidity count ---
# Comorbidity count proxy: number of distinct ICD diagnosis codes recorded for the hospital admission.

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
print("Tabular baseline feature matrix:", tabular_features_df.shape)

tabular_shape_note = (
    f"Tabular baseline has {tabular_features_df.shape[1]} columns "
    f"({drug_onehot.shape[1]} one-hot drug indicators + age/sex/n_diagnoses)."
)
print(tabular_shape_note)

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, brier_score_loss
import xgboost as xgb

OUTCOMES = ["aki_label", "delirium_label", "bleeding_label"]
FEATURE_SETS = {
    "embedding": embedding_features_df,
    "graph_theoretic": graphstat_features_df,
    "tabular": tabular_features_df,
}
N_FOLDS = 5
N_BOOTSTRAP = 1000
RANDOM_STATE = 42

def get_models():
    return {
        "logistic_regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "random_forest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
        "xgboost": xgb.XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1,
        ),
        "mlp": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=RANDOM_STATE),
    }

def bootstrap_ci(y_true, y_prob, metric_fn, n_boot=N_BOOTSTRAP, seed=RANDOM_STATE):
    rng = np.random.RandomState(seed)
    n = len(y_true)
    stats = []
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        try:
            stats.append(metric_fn(y_true[idx], y_prob[idx]))
        except Exception:
            continue
    if not stats:
        return (np.nan, np.nan)
    return (np.percentile(stats, 2.5), np.percentile(stats, 97.5))

results_rows = []

for outcome in OUTCOMES:
    y = cohort_df.set_index("hadm_id").loc[:, outcome]

    for fs_name, X_full in FEATURE_SETS.items():
        common_idx = X_full.index.intersection(y.index)
        X = X_full.loc[common_idx].values
        y_aligned = y.loc[common_idx].values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

        for model_name, _ in get_models().items():
            oof_pred = np.zeros(len(y_aligned))

            for train_idx, test_idx in skf.split(X_scaled, y_aligned):
                models = get_models()
                model = models[model_name]
                model.fit(X_scaled[train_idx], y_aligned[train_idx])
                if hasattr(model, "predict_proba"):
                    oof_pred[test_idx] = model.predict_proba(X_scaled[test_idx])[:, 1]
                else:
                    oof_pred[test_idx] = model.decision_function(X_scaled[test_idx])

            auroc = roc_auc_score(y_aligned, oof_pred)
            auprc = average_precision_score(y_aligned, oof_pred)
            f1 = f1_score(y_aligned, (oof_pred >= 0.5).astype(int))
            brier = brier_score_loss(y_aligned, oof_pred)

            auroc_lo, auroc_hi = bootstrap_ci(y_aligned, oof_pred, roc_auc_score)

            results_rows.append({
                "outcome": outcome,
                "feature_set": fs_name,
                "model": model_name,
                "n": len(y_aligned),
                "n_positive": int(y_aligned.sum()),
                "auroc": auroc,
                "auroc_ci_low": auroc_lo,
                "auroc_ci_high": auroc_hi,
                "auprc": auprc,
                "f1": f1,
                "brier": brier,
            })
            print(f"{outcome:16s} | {fs_name:16s} | {model_name:20s} | "
                  f"AUROC={auroc:.3f} [{auroc_lo:.3f},{auroc_hi:.3f}]  AUPRC={auprc:.3f}  F1={f1:.3f}  Brier={brier:.3f}")

model_results_df = pd.DataFrame(results_rows)
model_results_path = os.path.join(RESULTS_DIR, "model_results.csv")
model_results_df.to_csv(model_results_path, index=False)
save_manifest_entry(model_results_path, "Full model results grid: outcome x feature_set x model, with bootstrap AUROC CI")

def best_model_per_outcome_feature_set(results_df, outcome, feature_set):
    sub = results_df[(results_df.outcome == outcome) & (results_df.feature_set == feature_set)]
    return sub.sort_values("auroc", ascending=False).iloc[0]["model"]

def fit_full_model(model_name, X, y):
    models = get_models()
    model = models[model_name]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model.fit(X_scaled, y)
    return model, scaler

for outcome in OUTCOMES:
    y = cohort_df.set_index("hadm_id").loc[:, outcome]
    common_idx = embedding_features_df.index.intersection(y.index)
    X_emb = embedding_features_df.loc[common_idx]
    y_aligned = y.loc[common_idx]

    best_model_name = best_model_per_outcome_feature_set(model_results_df, outcome, "embedding")
    model, scaler = fit_full_model(best_model_name, X_emb.values, y_aligned.values)

    X_scaled_full = scaler.transform(X_emb.values)
    base_pred = model.predict_proba(X_scaled_full)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X_scaled_full)

    high_risk_thresh = np.percentile(base_pred, 90)
    high_risk_hadm = X_emb.index[base_pred >= high_risk_thresh]

    drug_deltas = defaultdict(list)
    for hadm_id in high_risk_hadm:
        drugs = hadm_drugs.get(hadm_id, [])
        if len(drugs) < 2:
            continue
        base_vec = patient_embedding_feature(drugs)
        base_score = model.predict_proba(scaler.transform([base_vec]))[:, 1][0]
        for d in drugs:
            remaining = [x for x in drugs if x != d]
            perturbed_vec = patient_embedding_feature(remaining)
            perturbed_score = model.predict_proba(scaler.transform([perturbed_vec]))[:, 1][0]
            drug_deltas[d].append(base_score - perturbed_score)

    delta_summary = pd.DataFrame([
        {"drug": d, "mean_risk_contribution": np.mean(v), "n_high_risk_patients": len(v)}
        for d, v in drug_deltas.items()
    ]).sort_values("mean_risk_contribution", ascending=False)

    top15 = delta_summary.head(15)
    out_path = os.path.join(RESULTS_DIR, f"top_risk_drugs_{outcome}.csv")
    top15.to_csv(out_path, index=False)
    save_manifest_entry(out_path, f"Top-15 perturbation-ranked risk-associated drugs for {outcome} (best model: {best_model_name})")
    print(f"\n=== {outcome} (best embedding model: {best_model_name}) ===")
    print(top15.to_string(index=False))

# --- SHAP on the tabular baseline (directly interpretable), best model per outcome ---

import shap

for outcome in OUTCOMES:
    y = cohort_df.set_index("hadm_id").loc[:, outcome]
    common_idx = tabular_features_df.index.intersection(y.index)
    X_tab = tabular_features_df.loc[common_idx]
    y_aligned = y.loc[common_idx]

    best_model_name = best_model_per_outcome_feature_set(model_results_df, outcome, "tabular")
    if best_model_name != "random_forest" and best_model_name != "xgboost":
        print(f"Skipping SHAP tree-explainer for {outcome}: best tabular model was {best_model_name} (not tree-based).")
        continue

    model, scaler = fit_full_model(best_model_name, X_tab.values, y_aligned.values)
    X_scaled_full = scaler.transform(X_tab.values)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_scaled_full[:2000])  # cap for runtime

    shap_summary_path = os.path.join(RESULTS_DIR, f"shap_summary_tabular_{outcome}.pkl")
    with open(shap_summary_path, "wb") as f:
        pickle.dump({
            "shap_values": shap_values,
            "feature_names": X_tab.columns.tolist(),
            "model_name": best_model_name,
        }, f)
    save_manifest_entry(shap_summary_path, f"SHAP values for best tabular model ({best_model_name}) on {outcome}")
    print(f"Saved SHAP summary for {outcome} ({best_model_name}).")

print("Top-20 highest-weight co-administered drug PAIRS (population graph):")
print(top20_edges.to_string(index=False))
print()
for outcome in OUTCOMES:
    path = os.path.join(RESULTS_DIR, f"top_risk_drugs_{outcome}.csv")
    if os.path.exists(path):
        print(f"--- Top risk-associated drugs for {outcome} (manually cross-check against DDI references) ---")
        print(pd.read_csv(path).to_string(index=False))
        print()

manifest_path = os.path.join(RESULTS_DIR, "manifest.json")
with open(manifest_path, "w") as f:
    json.dump(MANIFEST, f, indent=2)

zip_path = "results.zip"
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for root, _, files in os.walk(RESULTS_DIR):
        for fn in files:
            full_path = os.path.join(root, fn)
            zf.write(full_path, arcname=os.path.relpath(full_path, RESULTS_DIR))

print(f"Results zipped to {zip_path}\n")
print(f"{'FILE':45s} {'SIZE':>10s}  DESCRIPTION")
print("-" * 100)
for entry in MANIFEST:
    fp = entry["file"]
    size = os.path.getsize(fp) if os.path.exists(fp) else 0
    size_str = f"{size/1024:.1f} KB" if size < 1024*1024 else f"{size/1024/1024:.1f} MB"
    print(f"{os.path.basename(fp):45s} {size_str:>10s}  {entry['description']}")

print(f"\nGenerated: {datetime.now().isoformat()}")


# --- Local save confirmation (replaces Colab's files.download) ---
import os as _os
print(f"\nPipeline complete. Results are on disk at: {_os.path.abspath(RESULTS_DIR)}")
print(f"Zipped bundle at: {_os.path.abspath(zip_path)}")
