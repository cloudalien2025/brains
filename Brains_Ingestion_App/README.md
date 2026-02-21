# Brains Ingestion App (MVP)

Streamlit app for generating Brain Packs from a keyword using mocked ingestion components.

## Features

- Keyword input
- Max videos slider (default 25)
- Discovery-only toggle
- Generate Brain Pack button
- Expanders for discovery results, queue, run log, and outputs
- JSON schema validation gate before any pack output is written

## Run

```bash
cd Brains_Ingestion_App
python -m venv .venv
source .venv/bin/activate
pip install streamlit jsonschema
streamlit run app.py
```
