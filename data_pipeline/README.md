# Data Pipeline — Scrape → Clean → Convert → Store → Query

A self-contained data-engineering pipeline: scrape live catalog data from a public
scraping-practice site, clean and enrich it, load it into a normalized SQLite database, and
query it with both SQL and pandas.

Data source: [books.toscrape.com](https://books.toscrape.com/) — built specifically for
scraping practice, no login/API key/paid tier required. The catalog is books rather than
groceries; the exercise is about pipeline mechanics (identical regardless of product
category), not the specific domain.

## Folder structure

    /data_pipeline
        pipeline_utils.py    ConvertHtmlToData: scrape, clean, currency-convert
        database.py          DataBase: creates the SQLite schema (categories, books)
        pipeline.py          DataPipeline: orchestrates scrape -> clean -> store -> query;
                              __main__ block is the runnable end-to-end script
        database/books.db    the SQLite database file (recreated by running pipeline.py)
        resources/
            queries.json         the 5 SQL queries run against the database, by name
            query_results.json   each query's exact text + row output from the last run

## Architecture

    Scrape -> Clean -> Convert -> Store -> Query

- **Scrape** (`ConvertHtmlToData.get_scraped_books_data`) — fetches 4 category pages from
  books.toscrape.com and parses each book's title, price, star rating, and availability with
  BeautifulSoup.
- **Clean** (`ConvertHtmlToData.clean_scraped_data`) — converts price from GBP to INR at a
  fixed rate (`_convert_gbp_inr`, 1 GBP = 105.50 INR), maps the star-rating word ("One".."Five")
  to a numeric 1-5 (`_convert_rating_to_numeric`), and converts "In stock" text to a boolean
  `availability` flag.
- **Convert** — happens as part of cleaning above (currency + categorical-to-numeric).
- **Store** (`DataBase.create_conn_tables_database` + `DataPipeline.__main__`) — creates a
  normalized two-table SQLite schema (`categories`, `books`) with a foreign key from `books` to
  `categories`, then loads the cleaned data via `pandas.DataFrame.to_sql`.
- **Query** (`DataPipeline.run_queries`) — runs 5 named SQL queries from `resources/queries.json`
  against the database and logs each query's text + output to `resources/query_results.json`.

## Design decisions

- **Normalized schema over a single flat table.** `categories` and `books` are split with a
  foreign key (`ON DELETE CASCADE`) instead of repeating the category name on every book row,
  matching how a real catalog/competitive-intelligence store would be modeled.
- **`INSERT OR IGNORE` for categories, `replace` for books.** Categories are looked up/inserted
  idempotently so re-running the pipeline doesn't create duplicate category rows; `books` is
  fully replaced each run since it's re-scraped from scratch every time, so keeping stale rows
  around would be incorrect.
- **Fixed-rate currency conversion.** A single constant (105.50 INR/GBP) is used rather than a
  live FX API call, since the assignment is about the pipeline mechanics, not FX accuracy, and
  a fixed rate keeps the pipeline deterministic and network-independent for that step.
- **SQL queries stored as data, not hardcoded strings**, in `resources/queries.json` — keeps
  the 5 required query types (WHERE, ORDER BY + LIMIT, DISTINCT, BETWEEN, and a window-function
  JOIN ranking top books per category) declarative and easy to inspect/re-run independently of
  the pipeline code, and lets `run_queries` log each query's exact text next to its output.
- **Cross-check the JOIN in pandas, with no SQL.** `pipeline.py`'s `__main__` block reproduces
  `join_top_rated_per_category` purely with `DataFrame.merge` + `groupby().cumcount()` and
  asserts it equals the SQL window-function result (`results_match`), which is the same
  computation done two ways as a correctness check on the SQL query.
- **Everything module-local.** The database file and query resources live inside
  `data_pipeline/` (not scattered at the repo root) so the module is self-contained and the
  whole thing can be deleted/recreated as a unit.

## Install / run

From the repo root, with the project's dependencies installed (`beautifulsoup4`, `requests`,
`pandas` — already in the root `requirements.txt`):

    pip install -r requirements.txt
    python3 data_pipeline/pipeline.py

This is also the exact recreation script for the database: it scrapes fresh data, calls
`DataBase.create_conn_tables_database()` (creates `data_pipeline/database/books.db` and its
schema if missing, `CREATE TABLE IF NOT EXISTS`), loads it, runs the 5 queries in
`resources/queries.json`, and overwrites `resources/query_results.json` with the fresh
query text + output. Deleting `data_pipeline/database/` entirely and re-running the command
above recreates it from scratch — this was verified while writing this module.

## Executed SQL queries and output

The 5 queries actually run (see `resources/queries.json` for the exact SQL, and
`resources/query_results.json` for the full logged output of the run above):

| Query | Technique |
|---|---|
| `select_where` | `SELECT ... WHERE rating >= 4` |
| `order_by_limit` | `SELECT ... ORDER BY price_inr DESC LIMIT 5` |
| `distinct` | `SELECT DISTINCT category_name FROM categories` |
| `between` | `SELECT ... WHERE price_inr BETWEEN 500 AND 1500` |
| `join_top_rated_per_category` | `JOIN` + `ROW_NUMBER() OVER (PARTITION BY ... ORDER BY ...)` to rank each category's top-10 books by rating/price |

The pipeline's `__main__` block also reads two of these back with `pd.read_sql` and reproduces
the JOIN query purely in pandas to confirm it matches the SQL output (`results_match`,
printed as `True` on every run).
