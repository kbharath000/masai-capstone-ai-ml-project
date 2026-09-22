
# Zepto Data & AI Platform

An end-to-end AI/ML engineer is expected to move comfortably across the full stack: pulling raw data from the wild, cleaning and storing it properly, understanding it visually, building and evaluating predictive models on it, and — increasingly — wrapping intelligence around it with a deployed GenAI service. In this capstone you are joining Zepto's analytics guild as an incoming AI/ML engineer, and your first assignment is to ship one connected platform made of three internally-linked capabilities: a data-engineering pipeline that turns raw scraped data into a clean relational store, an analytics pipeline that profiles and models a customer-style dataset end to end, and a GenAI support assistant that answers policy questions grounded in Zepto's own documents. All three live together in one repository — this is a single, coherent submission, not three unrelated exercises. Every technique required below is a standard, well-documented technique in professional AI/ML practice, with widely available tooling and documentation support.

Build the three modules in any order, but they are meant to read as one story: a data pipeline feeds Zepto's analysts clean structured data (/data_pipeline), an analytics pipeline shows how Zepto would profile and predict customer/passenger-style outcomes end to end (/analytics), and a support assistant shows how Zepto would put a grounded GenAI service in front of its own policies (/support_assistant). Each module is graded independently against its own criteria, but they all live in, and are submitted as, one single public GitHub repository.

## Setup

Requires Python >= 3.9. All commands below are run from the repo root.

```bash
# 1. clone and enter the repo
git clone <this-repo-url>
cd masai-capstone-ai-ml-project

# 2. create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 3. install dependencies (shared across all three modules)
pip install -r requirements.txt
```

`support_assistant` also ships its own `requirements.txt` (used only for its standalone Docker
build — see below); for local development the root `requirements.txt` already covers it.

## Running each module

### data_pipeline — scrape, clean, store, query

```bash
python3 data_pipeline/pipeline.py
```
or

```bash
python data_pipeline/pipeline.py
```

Scrapes books.toscrape.com, builds `data_pipeline/database/books.db`, and runs the 5 queries in
`data_pipeline/resources/queries.json`, logging their output to
`data_pipeline/resources/query_results.json`. See [data_pipeline/README.md](data_pipeline/README.md) for details.

### analytics — EDA + predictive modelling

```bash
python -m analytics.eda_analysis
python -m analytics.predictive_modelling
```

The first regenerates the EDA charts (PNGs) in `analytics/`; the second trains/compares/tunes
the classifiers and regression model and saves the winning pipeline to
`analytics/models/best_classifier_pipeline.joblib`. See [analytics/README.md](analytics/README.md) for details.

### support_assistant — RAG + LangGraph + FastAPI

```bash
uvicorn support_assistant.support_assistant:app --reload --port 8000
```

Runs with `MOCK_LLM=1` by default (deterministic, no API key needed), configurable via
`support_assistant/.env`. Once running, query it with:

```bash
curl -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" \
  -d '{"query": "What is your delivery policy for orders under INR 149?"}'
```

Alternatively, run it standalone in Docker:

```bash
cd support_assistant
docker build -t support-assistant .
docker run -p 7860:7860 support-assistant
```

See [support_assistant/README.md](support_assistant/README.md) for the full architecture and API details.
