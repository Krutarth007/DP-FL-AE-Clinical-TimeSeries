#!/usr/bin/env python3
"""
mimic_to_fhir_5000.py

Converts a sampled subset (5000 patients) of MIMIC-IV CSVs into per-patient FHIR Bundle JSONs,
using privacy-preserving relative offsets (days since patient's t0). Also creates a federated
client split manifest (default 50 clients).

This version increases the total sample size to 5000 patients and uses separate output 
directories to avoid mixing with the original 2000 patient subset.
"""

import os
import json
import pandas as pd
from pathlib import Path
import uuid
from datetime import datetime
import math
import logging
import random

# ---------------- USER CONFIG ----------------
BASE_DIR = r"C:\mimic-iv-2.2"   # <- CHANGE this to your local MIMIC folder root

# === MODIFICATIONS FOR 5000 PATIENTS ===
SAMPLE_SIZE = 5000              # *** CRITICAL CHANGE: Increased total sample size to 5000 ***
# Dynamic output directory names to avoid collision with previous run
OUT_DIR = os.path.join(BASE_DIR, f"mimic_fhir_{SAMPLE_SIZE}_output")
SUBSET_DIR = os.path.join(BASE_DIR, f"subset_{SAMPLE_SIZE}")
# =======================================

RANDOM_SEED = 42
N_CLIENTS = 50
CHUNK_SIZE = 200_000    # chunk size for streaming large CSVs
# Whether to produce synthetic absolute datetimes in addition to offsets.
# For privacy-preserving experiments we default to False (only offsets).
SYNTHETIC_ABSOLUTE = False
SYNTHETIC_EPOCH = datetime(2000, 1, 1)  # only used if SYNTHETIC_ABSOLUTE = True
# ---------------------------------------------

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def find_csvs_in_dir(d: str):
    """Return dict mapping stem -> fullpath for csv files under dir d."""
    out = {}
    for root, _, files in os.walk(d):
        for f in files:
            if f.lower().endswith(".csv"):
                out[Path(f).stem] = os.path.join(root, f)
    return out

def discover_mimic_csvs(base_dir: str):
    """Discover CSVs in hosp/ and icu/ under base_dir."""
    csv_map = {}
    for folder in ("hosp", "icu"):
        dirpath = os.path.join(base_dir, folder)
        if os.path.isdir(dirpath):
            csv_map.update(find_csvs_in_dir(dirpath))
    return csv_map

def safe_read_csv(path: str, usecols=None):
    if not path or not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False, usecols=usecols)

def to_iso(dt):
    """Convert pandas timestamp or string to ISO str; return None on failure."""
    if pd.isnull(dt):
        return None
    try:
        ts = pd.to_datetime(dt, errors='coerce')
        if pd.isnull(ts):
            return None
        return ts.isoformat()
    except Exception:
        return None

def compute_offset_days(ref_ts, ts):
    """Return integer days offset (ts - ref_ts). If ts invalid return None."""
    if ref_ts is None or ts is None:
        return None
    try:
        a = pd.to_datetime(ref_ts, errors='coerce')
        b = pd.to_datetime(ts, errors='coerce')
        if pd.isnull(a) or pd.isnull(b):
            return None
        delta = b - a
        # Round to nearest integer day (floor)
        return int(math.floor(delta.total_seconds() / 86400.0))
    except Exception:
        return None

# Main pipeline
def main():
    logging.info("Discovering CSV files...")
    csv_map = discover_mimic_csvs(BASE_DIR)
    if "patients" not in csv_map:
        logging.error("Could not find patients.csv under hosp/ or icu/ in BASE_DIR.")
        logging.error("CSV discovery result keys: %s", list(csv_map.keys())[:50])
        return

    logging.info("Found %d csv files (showing keys): %s", len(csv_map), list(csv_map.keys())[:40])

    ensure_dir(SUBSET_DIR)
    ensure_dir(OUT_DIR)
    logging.info("Output directory: %s", OUT_DIR)
    logging.info("Subset CSVs directory: %s", SUBSET_DIR)

    # Step 1: sample 5000 patients
    logging.info("Sampling %d patients from patients.csv (seed=%d)...", SAMPLE_SIZE, RANDOM_SEED)
    df_patients_all = pd.read_csv(csv_map["patients"], low_memory=False)
    if SAMPLE_SIZE > len(df_patients_all):
        logging.warning("Requested sample size %d > total patients %d. Reducing sample size.", SAMPLE_SIZE, len(df_patients_all))
    
    # *** CRITICAL: Sampling 5000 total patients ***
    sample = df_patients_all.sample(n=min(SAMPLE_SIZE, len(df_patients_all)), random_state=RANDOM_SEED).reset_index(drop=True)
    sampled_ids = set(sample["subject_id"].tolist())
    sample_path = os.path.join(SUBSET_DIR, "patients.csv")
    sample.to_csv(sample_path, index=False)
    logging.info("Wrote sampled patients to %s", sample_path)

    # Step 2: stream-filter all other CSVs by subject_id -> write to SUBSET_DIR
    logging.info("Filtering other CSVs by sampled subject_id (chunked read) to %s", SUBSET_DIR)
    for key, fullpath in csv_map.items():
        if key == "patients":
            continue
        outpath = os.path.join(SUBSET_DIR, f"{key}.csv")
        logging.info("Filtering %s -> %s", key, outpath)
        try:
            reader = pd.read_csv(fullpath, chunksize=CHUNK_SIZE, low_memory=False)
            first_write = True
            total_kept = 0
            for chunk in reader:
                if "subject_id" in chunk.columns:
                    filtered = chunk[chunk["subject_id"].isin(sampled_ids)]
                else:
                    # table doesn't have subject_id; skip writing rows
                    filtered = pd.DataFrame(columns=chunk.columns)
                if not filtered.empty:
                    if first_write:
                        filtered.to_csv(outpath, index=False, mode="w")
                        first_write = False
                    else:
                        filtered.to_csv(outpath, index=False, mode="a", header=False)
                    total_kept += len(filtered)
            if not os.path.exists(outpath) or total_kept == 0:
                # No rows matched or file doesn't exist; create empty CSV with correct header
                columns_to_use = pd.read_csv(fullpath, nrows=0, low_memory=False).columns.tolist()
                pd.DataFrame(columns=columns_to_use).to_csv(outpath, index=False)
            logging.info("  kept %d rows for %s", total_kept, key)
        except pd.errors.EmptyDataError:
            pd.DataFrame().to_csv(outpath, index=False)
            logging.warning("  %s is empty, created empty out file", key)
        except Exception as e:
            logging.exception("  error streaming %s: %s. Attempting fallback.", fullpath, e)
            # fallback: full read (may be heavy)
            try:
                df_full = pd.read_csv(fullpath, low_memory=False)
                if "subject_id" in df_full.columns:
                    df_f = df_full[df_full["subject_id"].isin(sampled_ids)]
                else:
                    df_f = pd.DataFrame(columns=df_full.columns)
                df_f.to_csv(outpath, index=False)
                logging.info("  fallback wrote %d rows", len(df_f))
            except Exception as e2:
                logging.exception("  fallback failed for %s: %s", fullpath, e2)
                pd.DataFrame().to_csv(outpath, index=False)

    # Step 3: load subset CSVs we will use (safe reads)
    logging.info("Loading subset CSVs (safe reads)...")
    def lr(k): return safe_read_csv(os.path.join(SUBSET_DIR, f"{k}.csv"))
    df_patients = lr("patients")
    df_admissions = lr("admissions")
    df_icustays = lr("icustays")
    df_diagnoses = lr("diagnoses_icd")
    df_procedures = lr("procedures_icd")
    df_prescriptions = lr("prescriptions")
    df_labevents = lr("labevents")
    df_chartevents = lr("chartevents")
    df_micro = lr("microbiologyevents")
    df_services = lr("services")

    logging.info("Loaded dataframes. Patients: %d, admissions rows: %d", len(df_patients), len(df_admissions))

    # Build a list of subject_ids in the sample (for iteration order)
    subject_list = df_patients["subject_id"].tolist()

    # Step 4: Prepare federated client split
    logging.info("Preparing federated split: %d clients", N_CLIENTS)
    random.Random(RANDOM_SEED).shuffle(subject_list)  # shuffle deterministically
    clients = {f"client_{i+1}": subject_list[i::N_CLIENTS] for i in range(N_CLIENTS)}
    manifest = {
        "sample_size": len(subject_list),
        "clients": {k: len(v) for k, v in clients.items()},
        "seed": RANDOM_SEED,
        "timeline_policy": "relative_offsets_only",
        "synthetic_absolute_enabled": SYNTHETIC_ABSOLUTE
    }
    with open(os.path.join(OUT_DIR, "manifest.json"), "w") as mf:
        json.dump(manifest, mf, indent=2)
    # Also write mapping file (patient -> client)
    pat2client = {}
    for c, pats in clients.items():
        for p in pats:
            pat2client[str(p)] = c
    with open(os.path.join(OUT_DIR, "patient_to_client_map.json"), "w") as f:
        json.dump(pat2client, f, indent=2)

    # Utility to build resource ids
    def pid(prefix, *parts):
        delim = "-"
        return f"{prefix}{delim}{delim.join(str(x) for x in parts if x is not None and x!='')}"

    # Step 5: convert each patient -> FHIR Bundle JSON
    logging.info("Converting each patient to FHIR Bundle JSONs (writing to %s)...", OUT_DIR)
    total_written = 0
    for idx, subj in enumerate(subject_list, 1):
        try:
            p_row = df_patients[df_patients["subject_id"] == subj].iloc[0].to_dict()
            # Determine t0 (patient-level reference time) = earliest available timestamp among admissions/icustays
            # Use admittime or icu intime
            t0_candidates = []
            if not df_admissions.empty:
                adm = df_admissions[df_admissions["subject_id"] == subj]
                if not adm.empty and "admittime" in adm.columns:
                    t0_candidates += adm["admittime"].dropna().astype(str).tolist()
            if not df_icustays.empty:
                icu = df_icustays[df_icustays["subject_id"] == subj]
                if not icu.empty:
                    # use intime if present
                    for col in ("intime", "in_time", "starttime"):
                        if col in icu.columns:
                            t0_candidates += icu[col].dropna().astype(str).tolist()
                            break
            # fallback: use patients table anchor_year (not as date) -> produce no t0 if nothing else
            t0 = None
            if t0_candidates:
                # pick earliest
                t0_parsed = pd.to_datetime(t0_candidates, errors='coerce')
                t0_valid = t0_parsed[~t0_parsed.isna()]
                if not t0_valid.empty:
                    t0 = t0_valid.min().isoformat()
            # Build patient resource (do NOT set birthDate from anchor_year)
            patient_resource = {
                "resourceType": "Patient",
                "id": pid("patient", subj),
                "identifier": [{"system": "http://mimic.mit.edu/subject", "value": str(subj)}],
                "gender": str(p_row.get("gender", "") or "").lower(),
                # Add anchor_year as an extension (deid provenance), do not use as birthDate
                "extension": [
                    {"url": "http://mimic.mit.edu/fhir/StructureDefinition/deid-anchor-year", "valueString": str(p_row.get("anchor_year", ""))},
                    {"url": "http://mimic.mit.edu/fhir/StructureDefinition/deid-anchor-age", "valueInteger": int(p_row.get("anchor_age", 0)) if pd.notnull(p_row.get("anchor_age", None)) else None}
                ]
            }

            entries = [{"resource": patient_resource}]

            # Admissions -> Encounter resources
            if not df_admissions.empty:
                adm_rows = df_admissions[df_admissions["subject_id"] == subj]
                for _, a in adm_rows.iterrows():
                    hadm = a.get("hadm_id", "")
                    enc_id = pid("enc", subj, hadm)
                    enc = {
                        "resourceType": "Encounter",
                        "id": enc_id,
                        "subject": {"reference": f"Patient/{patient_resource['id']}"},
                        "identifier": [{"system": "http://mimic.mit.edu/hadm", "value": str(hadm)}],
                        "period": {
                            "start": to_iso(a.get("admittime")),
                            "end": to_iso(a.get("dischtime"))
                        },
                        # store offsets relative to t0 as extensions
                    }
                    # compute offsets if t0 available
                    if t0:
                        off_start = compute_offset_days(t0, a.get("admittime"))
                        off_end = compute_offset_days(t0, a.get("dischtime"))
                        if off_start is not None:
                            enc.setdefault("extension", []).append({"url":"http://your-research.org/fhir/StructureDefinition/event-offset-days-start","valueInteger": off_start})
                        if off_end is not None:
                            enc.setdefault("extension", []).append({"url":"http://your-research.org/fhir/StructureDefinition/event-offset-days-end","valueInteger": off_end})
                        if SYNTHETIC_ABSOLUTE:
                            # create synthetic absolutes by adding offset to SYNTHETIC_EPOCH
                            import datetime as _dt
                            if off_start is not None:
                                synth_start = (SYNTHETIC_EPOCH + _dt.timedelta(days=off_start)).isoformat()
                                enc["period"]["start"] = synth_start
                            if off_end is not None:
                                synth_end = (SYNTHETIC_EPOCH + _dt.timedelta(days=off_end)).isoformat()
                                enc["period"]["end"] = synth_end
                    entries.append({"resource": enc})

            # ICU stays -> special Encounter entries
            if not df_icustays.empty:
                icu_rows = df_icustays[df_icustays["subject_id"] == subj]
                for _, r in icu_rows.iterrows():
                    stay_id = r.get("stay_id") or r.get("icustay_id") or uuid.uuid4().hex
                    enc_id = pid("encicu", subj, stay_id)
                    enc = {
                        "resourceType": "Encounter",
                        "id": enc_id,
                        "subject": {"reference": f"Patient/{patient_resource['id']}"},
                        "class": {"code": "ICU"},
                        "period": {"start": to_iso(r.get("intime")), "end": to_iso(r.get("outtime"))}
                    }
                    if t0:
                        off_start = compute_offset_days(t0, r.get("intime"))
                        off_end = compute_offset_days(t0, r.get("outtime"))
                        if off_start is not None:
                            enc.setdefault("extension", []).append({"url":"http://your-research.org/fhir/StructureDefinition/event-offset-days-start","valueInteger": off_start})
                        if off_end is not None:
                            enc.setdefault("extension", []).append({"url":"http://your-research.org/fhir/StructureDefinition/event-offset-days-end","valueInteger": off_end})
                        if SYNTHETIC_ABSOLUTE:
                            import datetime as _dt
                            if off_start is not None:
                                enc["period"]["start"] = (SYNTHETIC_EPOCH + _dt.timedelta(days=off_start)).isoformat()
                            if off_end is not None:
                                enc["period"]["end"] = (SYNTHETIC_EPOCH + _dt.timedelta(days=off_end)).isoformat()
                    entries.append({"resource": enc})

            # Diagnoses -> Condition
            if not df_diagnoses.empty:
                diag_rows = df_diagnoses[df_diagnoses["subject_id"] == subj]
                for _, d in diag_rows.iterrows():
                    cond_id = pid("cond", subj, d.get("hadm_id", ""), d.get("seq_num", ""))
                    condition = {
                        "resourceType": "Condition",
                        "id": cond_id,
                        "subject": {"reference": f"Patient/{patient_resource['id']}"},
                        "code": {
                            "text": d.get("icd_code", ""),
                            "coding": [{"system":"http://hl7.org/fhir/sid/icd-10-cm", "code": d.get("icd_code", "")}]
                        }
                    }
                    # attach onset offset if diag time exists (diagnoses_icd might not have timestamp)
                    # leave as extension if present
                    if "charttime" in d and pd.notnull(d.get("charttime")):
                        off = compute_offset_days(t0, d.get("charttime")) if t0 else None
                        if off is not None:
                            condition.setdefault("extension", []).append({"url":"http://your-research.org/fhir/StructureDefinition/event-offset-days","valueInteger": off})
                    entries.append({"resource": condition})

            # Procedures -> Procedure
            if not df_procedures.empty:
                proc_rows = df_procedures[df_procedures["subject_id"] == subj]
                for _, pr in proc_rows.iterrows():
                    pr_id = pid("proc", subj, pr.get("hadm_id", ""), pr.get("seq_num", ""))
                    proc = {
                        "resourceType": "Procedure",
                        "id": pr_id,
                        "subject": {"reference": f"Patient/{patient_resource['id']}"},
                        "code": {"text": pr.get("icd_code", "")}
                    }
                    entries.append({"resource": proc})

            # Prescriptions -> MedicationRequest
            if not df_prescriptions.empty:
                pres_rows = df_prescriptions[df_prescriptions["subject_id"] == subj]
                for i, row in enumerate(pres_rows.to_dict("records"), 1):
                    med_id = pid("medreq", subj, i)
                    med = {
                        "resourceType": "MedicationRequest",
                        "id": med_id,
                        "subject": {"reference": f"Patient/{patient_resource['id']}"},
                        "medicationCodeableConcept": {"text": str(row.get("drug", ""))}
                    }
                    # authoredOn offset
                    if "startdate" in row and pd.notnull(row.get("startdate")) and t0:
                        off = compute_offset_days(t0, row.get("startdate"))
                        if off is not None:
                            med.setdefault("extension", []).append({"url":"http://your-research.org/fhir/StructureDefinition/event-offset-days","valueInteger": off})
                            if SYNTHETIC_ABSOLUTE:
                                import datetime as _dt
                                med["authoredOn"] = (SYNTHETIC_EPOCH + _dt.timedelta(days=off)).isoformat()
                    entries.append({"resource": med})

            # Labs and Chart events -> Observations (valueQuantity when possible)
            def add_observations_from_df(df, prefix, time_cols=("charttime","chartdate","charttime"), value_col_candidates=("valuenum","value")):
                if df.empty:
                    return
                rows = df[df["subject_id"] == subj]
                if rows.empty:
                    return
                for i, row in enumerate(rows.to_dict("records"), 1):
                    itemid = row.get("itemid", "") or row.get("param_id", "") or row.get("test_id", "")
                    obs_id = pid(prefix, subj, i)
                    obs = {
                        "resourceType": "Observation",
                        "id": obs_id,
                        "subject": {"reference": f"Patient/{patient_resource['id']}"},
                        "code": {"text": str(itemid), "coding": [{"system":"http://mimic.mit.edu/itemid","code": str(itemid)}]}
                    }
                    # value
                    val = None
                    for vc in value_col_candidates:
                        if vc in row and pd.notnull(row.get(vc)):
                            val = row.get(vc)
                            break
                    if val is not None:
                        try:
                            valf = float(val)
                            obs["valueQuantity"] = {"value": valf}
                        except Exception:
                            obs["valueString"] = str(val)
                    # time -> compute offset
                    ts = None
                    for tc in time_cols:
                        if tc in row and pd.notnull(row.get(tc)):
                            ts = row.get(tc)
                            break
                    if ts and t0:
                        off = compute_offset_days(t0, ts)
                        if off is not None:
                            obs.setdefault("extension", []).append({"url":"http://your-research.org/fhir/StructureDefinition/event-offset-days","valueInteger": off})
                            if SYNTHETIC_ABSOLUTE:
                                import datetime as _dt
                                obs["effectiveDateTime"] = (SYNTHETIC_EPOCH + _dt.timedelta(days=off)).isoformat()
                    entries.append({"resource": obs})

            add_observations_from_df(df_labevents, "lab")
            add_observations_from_df(df_chartevents, "chart")
            add_observations_from_df(df_micro, "micro", time_cols=("charttime","chartdate"), value_col_candidates=("org_name","result"))

            # Services -> ServiceRequest (optional)
            if not df_services.empty:
                svc_rows = df_services[df_services["subject_id"] == subj]
                for i, row in enumerate(svc_rows.to_dict("records"), 1):
                    svc_id = pid("svc", subj, i)
                    svc = {
                        "resourceType": "ServiceRequest",
                        "id": svc_id,
                        "subject": {"reference": f"Patient/{patient_resource['id']}"},
                        "code": {"text": str(row.get("curr_service", ""))}
                    }
                    entries.append({"resource": svc})

            # Build final Bundle
            bundle = {
                "resourceType": "Bundle",
                "type": "collection",
                "entry": entries,
                "meta": {
                    "source": "MIMIC-IV (subset)",
                    "provenance": {"sample_seed": RANDOM_SEED}
                }
            }

            # write patient json
            outfile = os.path.join(OUT_DIR, f"patient_{subj}.json")
            with open(outfile, "w") as ofh:
                json.dump(bundle, ofh, indent=2, default=str)
            total_written += 1
            if total_written % 100 == 0:
                logging.info("  written %d patient JSONs so far...", total_written)
        except Exception as e:
            logging.exception("Error converting patient %s: %s", subj, e)
            # continue with next patient

    logging.info("Conversion done. Total patient JSONs written: %d", total_written)
    logging.info("Output directory: %s", OUT_DIR)
    logging.info("Subset CSVs directory: %s", SUBSET_DIR)
    logging.info("Manifest and patient-to-client map written to output directory.")

if __name__ == "__main__":
    main()
