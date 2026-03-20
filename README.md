# Reltio Match Rule Generator — Streamlit on Databricks Apps

Uploads a data file → profiles it → calls Claude → outputs a ready-to-use Reltio `matchGroups` JSON.

## Project structure

```
reltio-match-agent/
├── app.py               # Streamlit UI
├── app.yaml             # Databricks Apps config
├── requirements.txt
└── utils/
    ├── profiler.py      # Column profiling (pandas)
    └── llm.py           # Claude API call + caching
```

## Deploy to Databricks Apps

1. Zip this folder or push to a Git repo.
2. In your Databricks workspace go to **Apps → Create App**.
3. Choose **Import from your computer** (or connect your Git repo).
4. Upload the folder.
5. Set your Anthropic API key as a secret:
   - Apps UI → Your App → Settings → Environment Variables
   - Name: `ANTHROPIC_API_KEY`, Value: `sk-ant-...`
6. Click **Deploy**.

Databricks will run `python -m streamlit run app.py` as defined in `app.yaml`.

## Run locally (for testing)

```bash
pip install -r requirements.txt
streamlit run app.py
```

## How it works

1. User uploads CSV / JSON / Parquet.
2. `profiler.py` analyses each column (null rate, cardinality, semantic type).
3. Columns above the null threshold are skipped.
4. `llm.py` builds a prompt with the profiling summary + few-shot example and calls
   `claude-sonnet-4-6` at `temperature=0`.
5. The response is validated (schema check) and cached by SHA-256 of the inputs.
6. The Streamlit UI renders each match group as a card and allows JSON download.

## Determinism guarantees

| Layer | Mechanism |
|-------|-----------|
| LLM randomness | `temperature=0` |
| Output structure | Strict system prompt + few-shot example |
| Schema | Validated on every call — errors surface immediately |
| Repeated inputs | SHA-256 cache → identical input always returns identical output |
