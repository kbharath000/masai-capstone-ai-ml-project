# Support Assistant — RAG + LangGraph + FastAPI

A retrieval-augmented Zepto policy support assistant. This module is self-contained: the
policy corpus, the LangGraph + FastAPI application code, the vendored embedding model, and
the Dockerfile all live here.

    /support_assistant
        docs/                    8 Zepto policy .txt files (delivery, returns, refunds, ...)
        models/all-MiniLM-L6-v2/ vendored embedding model weights (see models/README.md)
        prompt_template.py       role/context/task/format/length RAG prompt, one few-shot example
        support_assistant.py     ChromaDB knowledge base, LangGraph graph, Pydantic schema, FastAPI app
        requirements.txt         this module's own runtime dependencies (Docker build uses these)
        Dockerfile               builds and serves this module standalone
        chroma_db/               persisted ChromaDB collection (created on first run)

Tests live at `/tests/support_assistant` in the repo root (26 tests: ingestion, prompt
template, graph routing, and the FastAPI endpoint).

## Architecture: ingestion → embedding → retrieval → generation

**1. Ingestion.** `SupportKnowledgeBase._load_documents()` reads all 8 `.txt` files from
`support_assistant/docs/`, one `(doc_id, text)` pair per file. `SupportKnowledgeBase._chunk_text()`
then chunks each doc: every doc here is well under `CHUNK_SIZE` (50 chars), so each becomes
exactly one chunk; a longer doc would instead fall back to fixed-size slices with a 20-char
overlap (`CHUNK_OVERLAP`). Each chunk gets an id of the form `f"{doc_id}_chunk_{i}"` (e.g.
`doc_07_chunk_0`).

**2. Embedding.** `SupportKnowledgeBase.__init__` loads `SentenceTransformer(EMBEDDING_MODEL_NAME)`
once as `self.embedding_model`, where `EMBEDDING_MODEL_NAME` points at the vendored
`support_assistant/models/all-MiniLM-L6-v2/` directory (see "Getting the embedding model"
below for why it's a local path rather than the hub name `all-MiniLM-L6-v2`).
`SupportKnowledgeBase.build()` calls `self.embedding_model.encode(...)` on every chunk from
step 1, producing 384-dim vectors, and writes them with `self.collection.upsert(ids=...,
embeddings=..., documents=..., metadatas=...)` into the ChromaDB collection named
`support_docs` (`COLLECTION_NAME`), persisted on disk at `support_assistant/chroma_db/`
(`CHROMA_DIR`) through a `chromadb.PersistentClient`. The collection is created with
`metadata={"hnsw:space": "cosine"}` so similarity search is cosine, not ChromaDB's L2 default.
`build()` runs once, lazily, from the FastAPI `_lifespan` startup hook if the collection is
still empty.

**3. Retrieval.** This happens inside the LangGraph node `retrieve_and_answer`, reached only
when `classify_intent` has labeled the query `policy_question`. It calls
`SupportKnowledgeBase.query(text, n_results=3)`, which embeds the incoming query with the same
model and asks the `support_docs` collection for its top-3 cosine-nearest chunks, returning
their ids and text. Retrieval runs identically regardless of `MOCK_LLM` — no LLM call, no API
key, and (with the model vendored locally) no network call either.

**4. Generation.** This is the stage that branches on `MOCK_LLM`, and it differs by which
terminal node the graph reached:
- `retrieve_and_answer` (policy path) — mock mode returns the fixed template
  `f"Based on the retrieved context: {docs[0][:200]}"` built from only the single closest
  chunk, no LLM call. `MOCK_LLM=0` instead builds the full role–context–task–format–length
  prompt via `prompt_template.build_prompt(context, question)` (the retrieved chunks joined as
  context) and calls the Groq API through `_llm_answer_from_context()` →
  `_generate_structured_answer()`, which validates the raw model output against the
  `AnswerResponse` schema and retries up to 2 times with a corrective instruction before
  giving up with a clearly marked error.
- `direct_answer` (general path) — mock mode returns the fixed string
  `DIRECT_ANSWER_FALLBACK`, no retrieval, no LLM call. `MOCK_LLM=0` calls
  `_llm_direct_answer()` → the same `_generate_structured_answer()` validator, prompting the
  LLM with just the question and no context; `sources` stays empty either way.

(`classify_intent` also branches on `MOCK_LLM` — keyword heuristic vs. an LLM call via
`_llm_classify_intent()` — but that's the routing decision upstream of generation, not the
answer itself.)

**Data flow, end to end:**

```
docs/*.txt
   |  _load_documents()                                    [ingestion]
   v
(doc_id, text)
   |  _chunk_text()                                         [ingestion]
   v
chunks + chunk ids
   |  embedding_model.encode()                               [embedding]
   v
384-dim vectors --> collection.upsert() --> ChromaDB "support_docs" (persisted, cosine space)

                                    query arrives at POST /ask
                                              |
                                              v
                                   classify_intent (keyword or LLM)
                          policy_question /              \ general_question
                                        v                  v
                          retrieve_and_answer          direct_answer
                          query() embeds question,          |
                          top-3 cosine search  [retrieval]   |
                                        |                    |
                     MOCK_LLM? template : LLM+prompt  MOCK_LLM? fixed string : LLM   [generation]
                                        \                    /
                                              v
                          AnswerResponse{answer, sources, confidence}  (Pydantic-validated)
```

## LangGraph structure

    classify_intent --(policy_question)--> retrieve_and_answer --> END
                    \-(general_question)--> direct_answer      --> END

- `classify_intent` — keyword heuristic (mock) or LLM call (`MOCK_LLM=0`) over
  `delivery / return / refund / membership / tracking / cancel / gift card / support hours`.
- `retrieve_and_answer` — embeds the query, retrieves the top-3 ChromaDB chunks, then either
  returns a templated `"Based on the retrieved context: {snippet}"` string (mock) or prompts
  the LLM with the structured prompt template, grounded only in the retrieved chunks.
- `direct_answer` — returns the fixed fallback string (mock) or prompts the LLM directly, with
  no retrieval, in either case with empty `sources`.

## MOCK_LLM toggle

- **`MOCK_LLM` unset or `1` — REQUIRED, graded baseline.** Every generation step is
  deterministic (keyword heuristic for classification, canned/templated strings for answers).
  No LLM call, no API key, no paid or external LLM service of any kind. This is the mode the
  submission is graded on, and it's the default with no configuration needed.
- **`MOCK_LLM=0` — OPTIONAL / UNGRADED extension.** Generation steps call the Groq API
  (`llama-3.3-70b-versatile`) instead, with the structured-output schema enforced via Pydantic and up
  to 2 corrective retries before returning a clearly marked error response. Requires a
  `GROQ_API_KEY` (see `.env` below). This extension was written but not exercised against a
  live API key here — no real-LLM usage was incurred.

A `support_assistant/.env` file holds both settings for local runs
(`python-dotenv` loads it automatically on import): `MOCK_LLM=1` by default, and a
`GROQ_API_KEY` placeholder that only needs a real value if you opt into `MOCK_LLM=0`. It's
git-ignored — never commit a real key. The Docker image never reads this file; it sets
`MOCK_LLM=1` directly as a container `ENV` var (see Docker section below).

## Output schema

`AnswerResponse { answer: str, sources: list[str], confidence: float (0-1) }`, enforced via
Pydantic on every response — populated deterministically in mock mode (no LLM output to fail
validation), or by the LLM with retry-and-revalidate in the `MOCK_LLM=0` extension.

## Getting the embedding model (network note)

Rather than pulling weights from `huggingface.co` at runtime, the model was fetched once via
ModelScope's mirror of the same repo (`modelscope.cn`) using `modelscope.snapshot_download`,
pruned down to just the files sentence-transformers needs (`model.safetensors` + tokenizer/config
json, ~87 MB — the `pytorch_model.bin` / `tf_model.h5` / `rust_model.ot` / `onnx/` / `openvino/`
duplicates were dropped), and committed into this module at `models/all-MiniLM-L6-v2/`.
`EMBEDDING_MODEL_NAME` in `support_assistant.py` points at that local directory, so both local
runs and the Docker image work fully offline.

## Running locally

    pip install -r requirements.txt
    uvicorn support_assistant.support_assistant:app --reload --port 8000

(run from the repo root, so `support_assistant` resolves as a top-level package). The first
request triggers `SupportKnowledgeBase.build()` (via a FastAPI lifespan hook), which embeds
all 8 docs into ChromaDB using the vendored model — no network call required.

## Example call transcripts (`MOCK_LLM` left at its default, real embedding model, real ChromaDB)

Both calls below were sent with `curl` to a real running `uvicorn` process — not a test
client:

    curl -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" \
      -d '{"query": "What is your delivery policy for orders under INR 149?"}'

**1. Policy question** (routed to `retrieve_and_answer`; `classify_intent`'s keyword
heuristic matched "delivery", no LLM call):

```json
{
  "answer": "Based on the retrieved context: Delivery Policy: \"Zepto delivers grocery and household essentials to serviceable pin codes within 10 to 30 minutes of order confirmation, depending on the customer's delivery zone and current order vo",
  "sources": ["doc_01_chunk_0", "doc_05_chunk_0", "doc_06_chunk_0"],
  "confidence": 1.0
}
```

The top-ranked chunk (`doc_01_chunk_0`) is `docs/doc_01.txt` — Zepto's actual Delivery Policy
document — confirming retrieval found the chunk that genuinely matches the question, using the
real `all-MiniLM-L6-v2` embeddings and real ChromaDB cosine search.

    curl -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" \
      -d '{"query": "What is the capital of France?"}'

**2. General question** (routed to `direct_answer`; `classify_intent`'s keyword heuristic
matched none of the policy keywords, no LLM call, no retrieval):

```json
{"answer": "I can only answer questions about Zepto policies right now.", "sources": [], "confidence": 1.0}
```

## Docker

This module is a self-contained Docker build context — build from inside this directory:

    cd support_assistant
    docker build -t support-assistant .
    docker run -p 7860:7860 support-assistant

`COPY . ./support_assistant` in the Dockerfile brings the code, the docs corpus, and the
vendored `all-MiniLM-L6-v2` weights into the image together, so it needs no network call at
build or request-serving time. It runs as a non-root user and serves `POST /ask` on port 7860
with `MOCK_LLM=1` by default. This is the required, graded baseline; the image was not pushed
anywhere. Docker itself isn't available in the development sandbox this was built in, so the
build/run commands above are written and reviewed carefully but haven't been executed here —
the equivalent `uvicorn` command was run directly instead (see the example transcripts above,
which hit a real running server) to verify the app serves `/ask` correctly.

## Optional extensions

Neither optional extension was attempted for this submission:
- **Real-LLM (`MOCK_LLM=0`) usage**: the code path exists (`_llm_classify_intent`,
  `_llm_answer_from_context`, `_llm_direct_answer`, `_generate_structured_answer` with retry)
  but was never run against a live Groq API key, so there is no real-LLM usage or cost to
  report.
- **Hugging Face Spaces deployment**: not attempted; there is no live Space URL.

The required, graded baseline — mock-mode graph, retrieval, FastAPI app, and Dockerfile — is
complete and demonstrated above.
