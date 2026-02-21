# brains

`brains` is the canonical repository for **BD_Brain** content, machine-readable schemas, export contracts, and an executable Streamlit ingestion skeleton.

## What this repo includes

- **BD_Brain** baseline with authoritative master text, machine core JSONL initialization, and governance docs.
- **JSON Schemas (Draft 2020-12)** for records, packs, source metadata, and run metadata.
- **Export contract** describing `bd_brain_export.json` guarantees for downstream consumers.
- **Streamlit ingestion MVP** that discovers videos, creates mocked transcripts, extracts valid records, validates JSON, and writes a Brain Pack.

## What this repo does not include

- Any VacayRank application logic or code.

## Quick start

```bash
cd Brains_Ingestion_App
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt  # optional helper, see README inside app
streamlit run app.py
```

## Repository layout

```text
BD_Brain/
Governance/
Schemas/
Exports/
Brains_Ingestion_App/
```
