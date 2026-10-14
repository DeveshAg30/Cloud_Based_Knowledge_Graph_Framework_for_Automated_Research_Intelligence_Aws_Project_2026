# Dataset

`sample_papers.jsonl` holds 4 sample papers in the project's canonical document format (see `src/backend/kg/schema.py`).
Real data is fetched rather than stored in Git:
- `scripts/fetch_openalex.py`: OpenAlex metadata, abstracts and citations, plus optional open-access full text from arXiv.
- `scripts/fetch_s2orc.py`: Semantic Scholar S2ORC full text (needs a free API key).
- PDFs uploaded on the dashboard.

Downloaded data goes to `data/` (git-ignored) and is uploaded to S3 `raw/`.
