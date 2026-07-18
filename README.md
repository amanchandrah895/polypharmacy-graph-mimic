# Graph-Based Polypharmacy Risk Networks for Adverse Event Prediction in ICU Patients

A MIMIC-IV analysis comparing graph-embedding, graph-theoretic, and tabular
feature representations of ICU polypharmacy exposure for predicting three
adverse-event proxies: acute kidney injury (AKI), a delirium proxy, and a
bleeding proxy. Built for an INDICon submission.

## What this repo contains

This repo has **code only** — no patient data, no derived patient-level
extracts, no model artifacts trained on real admissions. See "Data Access"
below for why, and "Reproducing results" for how to regenerate everything
yourself with your own credentialed access.

```
notebooks/
  01_polypharmacy_pipeline.ipynb   Phase 1: cohort extraction, cohort/label
                                    construction, graph building, node2vec,
                                    model training (Colab, BigQuery-backed)
  02_resume_from_cache.ipynb       Phase 2: interpretability (perturbation +
                                    SHAP), delirium/tabular leakage fix,
                                    resumable from Phase 1's cached artifacts
                                    without re-pulling BigQuery data (Colab, T4)
  run_pipeline_local.py            Phase 1 as a standalone script, for running
                                    locally instead of in Colab
  replicate_delirium_subsample2.py Confirmatory replication: reruns the
                                    headline embedding-vs-tabular delirium
                                    comparison on a second, independent,
                                    zero-overlap subsample
  requirements.txt                 Python deps for the local script path

results_aggregate/
  model_results.csv                    Full outcome x feature-set x model AUROC/
                                        AUPRC/F1/Brier grid, with bootstrap 95% CIs
  cohort_attrition.csv                 Cohort filtering flow (Table 1)
  graph_stats.json                     Population co-administration graph stats
                                        (density, clustering, top-20 drugs/edges)
  degree_distribution_full.csv         Per-drug node degree (graph-level, not
                                        patient-level)
  top_risk_drugs_{outcome}.csv         Perturbation-ranked risk-associated drugs
                                        per outcome (reliability-filtered, min 10
                                        high-risk patients per drug)
  delirium_paired_significance_test.json
                                        Paired bootstrap significance test for the
                                        headline embedding-vs-tabular delirium result
```

Every file above is an **aggregate statistic** (counts, rates, per-drug or
per-model summaries) — none contain a row per patient or admission.

## Data Access

This project uses **MIMIC-IV** via Google BigQuery
(`physionet-data.mimiciv_3_1_hosp` / `mimiciv_3_1_icu` /
`mimiciv_3_1_derived`), which is **not publicly accessible**. To run any
notebook in this repo yourself, you need:

1. **PhysioNet credentialed access** to MIMIC-IV, with your PhysioNet
   account linked to the Google identity you'll authenticate with (see
   https://physionet.org/settings/cloud/).
2. **Your own GCP project with billing enabled.** `physionet-data` is
   read-only and shared; all query costs are billed to your project, not
   PhysioNet's. Cohort/prescription queries at the scale used here cost on
   the order of a few dollars, not more.

**Why no data is included in this repo:** MIMIC-IV is governed by the
PhysioNet Credentialed Health Data Use Agreement, which prohibits public
redistribution of the data or any derived patient/admission-level extract
— even de-identified. Concretely, this repo does **not** include: the
cohort table, normalized prescriptions, the co-administration graph, node2vec
embeddings, or SHAP value arrays, since all of these are built directly from
(and in some cases retain the shape of) real admission-level records. Only
the `results_aggregate/` files above are shared, since they contain no
patient rows — this mirrors what is reported in the paper itself.

## Reproducing results

1. **Phase 1** — run `notebooks/01_polypharmacy_pipeline.ipynb` in Colab
   (recommended for the BigQuery auth flow), or `notebooks/run_pipeline_local.py`
   locally after `gcloud auth application-default login` and
   `pip install -r notebooks/requirements.txt`. This builds the cohort,
   outcome labels, co-administration graph, node2vec embeddings, and the full
   4-model x 3-feature-set x 3-outcome CV grid, saving everything to a local
   `results/` folder (gitignored).
2. **Phase 2** — upload the 5 cached files it produces
   (`cohort.csv`, `prescriptions_normalized.csv`, `population_graphs.pkl`,
   `node2vec_embeddings.pkl`, `model_results.csv`) into
   `notebooks/02_resume_from_cache.ipynb` in Colab. This section does not
   re-pull the large BigQuery tables — it only reruns a small comorbidity-count
   query, the delirium/tabular leakage-corrected refit, and the
   interpretability analysis (batched perturbation + SHAP).

## Methodology notes (see the paper for full detail)

- **Cohort**: adult patients, first ICU stay per admission, ICU LOS ≥ 24h
  (68,013 eligible admissions), **randomly subsampled to 10,000** admissions
  (seed=42) for BigQuery cost/local compute tractability, then filtered to
  ≥3 distinct administered drugs (final n=9,974). This subsampling is a
  stated limitation, not a hidden one — see `delirium_paired_significance_test.json`
  and the replication note below for how this was stress-tested.
- **Outcome labels are proxies**, documented and reported as such: AKI via
  KDIGO stage ≥1 (derived table), delirium via antipsychotic/benzodiazepine
  administration **AND** CAM-ICU-positive chartevents (2 of the 4 CAM-ICU
  criteria, not the full clinical algorithm), bleeding via PRBC transfusion
  after ICU day 2.
- **Label-leakage correction**: the delirium label is partly *defined* by
  administration of specific drugs (haloperidol, quetiapine, etc.); the
  tabular one-hot baseline originally included those same drugs as features,
  inflating its AUROC to 0.944 via circularity. This was caught, the
  leakage-causing columns were excluded for the delirium outcome only, and
  the model was refit (corrected AUROC: 0.858). AKI/bleeding labels are not
  drug-defined and are unaffected.
- **Interpretability**: for graph-embedding models, drug-level risk
  attribution comes from a perturbation analysis (leave-one-drug-out,
  re-scored) rather than direct SHAP, since raw embedding dimensions aren't
  individually interpretable. Rankings are filtered to require ≥10 high-risk
  patients per drug, since unfiltered rankings were dominated by single-patient
  noise.
- **Confirmatory replication**: the headline result (node2vec embeddings
  beating a leak-free tabular baseline for delirium, paired bootstrap
  delta +0.0178, 95% CI [0.0112, 0.0249], p<0.0001) was re-tested on a
  second, independent, zero-overlap 10,000-admission subsample with the
  graph and embeddings rebuilt from scratch — see
  `replicate_delirium_subsample2.py` and its output for the replication delta.

## Manual verification still needed before submission

- Top risk-associated drug pairs (`top_risk_drugs_*.csv`) have **not** been
  cross-checked against a clinical drug-drug-interaction database
  (Lexicomp/Micromedex/FDA labels) — do this before claiming clinical
  validity in any Discussion section.
- No external validation cohort; single-center (BIDMC-derived MIMIC-IV) data.

## Requirements

See `notebooks/requirements.txt`. Python 3.12 recommended — `gensim`
(a `node2vec` dependency) does not currently build on Python 3.14 due to
removed CPython internals.
