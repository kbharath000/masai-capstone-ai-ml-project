"""Zepto support assistant: ChromaDB knowledge base, LangGraph intent router, and FastAPI app.

MOCK_LLM controls whether generation steps call a real LLM:
- unset or "1" (default, graded baseline): every generation step is deterministic —
  keyword heuristics and canned/templated strings. No LLM call, no API key needed.
- "0" (optional, ungraded extension): generation steps call the Groq API instead.
Retrieval (embedding + ChromaDB) always runs for real in both modes.
"""

import chromadb
import os
import groq

from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI

from langgraph.graph import END, StateGraph
from pathlib import Path
from pydantic import BaseModel, Field, ValidationError
from sentence_transformers import SentenceTransformer
from support_assistant.prompt_template import build_prompt
from typing import Optional, TypedDict


load_dotenv(Path(__file__).parent / ".env")

DOCS_DIR = Path(__file__).parent / "docs"
CHROMA_DIR = Path(__file__).parent / "chroma_db"
COLLECTION_NAME = "support_docs"
EMBEDDING_MODEL_NAME = str(Path(__file__).parent / "models" / "all-MiniLM-L6-v2")
CHUNK_SIZE = 50
CHUNK_OVERLAP = 20

MOCK_LLM = os.environ.get("MOCK_LLM", "1") != "0"
LLM_MODEL_NAME = "llama-3.3-70b-versatile"

POLICY_KEYWORDS = (
    "delivery",
    "return",
    "refund",
    "membership",
    "tracking",
    "cancel",
    "gift card",
    "support hours",
)
DIRECT_ANSWER_FALLBACK = "I can only answer questions about Zepto policies right now."


class SupportKnowledgeBase:
    """Loads support docs, embeds them, and stores them in a ChromaDB collection."""

    def __init__(self, docs_dir: Path = DOCS_DIR, chroma_dir: Path = CHROMA_DIR):
        self.docs_dir = docs_dir
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        self.client = chromadb.PersistentClient(path=str(chroma_dir))
        self.collection = self.client.get_or_create_collection(
            COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
        )

    def _load_documents(self) -> list[tuple[str, str]]:
        """Read every .txt file in docs_dir, returning (doc_id, text) pairs."""
        doc_paths = sorted(self.docs_dir.glob("*.txt"))
        return [(path.stem, path.read_text().strip()) for path in doc_paths]

    def _chunk_text(self, text: str) -> list[str]:
        """Split text into overlapping fixed-size chunks (a no-op for short docs)."""
        if len(text) <= CHUNK_SIZE:
            return [text]

        chunks = []
        start = 0
        while start < len(text):
            end = start + CHUNK_SIZE
            chunks.append(text[start:end])
            start = end - CHUNK_OVERLAP
        return chunks

    def build(self) -> int:
        """Embed all doc chunks and upsert them into the ChromaDB collection.

        Returns the number of chunks stored.
        """
        documents, ids, metadatas = [], [], []
        for doc_id, text in self._load_documents():
            for i, chunk in enumerate(self._chunk_text(text)):
                documents.append(chunk)
                ids.append(f"{doc_id}_chunk_{i}")
                metadatas.append({"source": doc_id, "chunk_index": i})

        embeddings = self.embedding_model.encode(documents).tolist()

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        return len(documents)

    def query(self, text: str, n_results: int = 3):
        """Return the n_results chunks most similar to the given text (cosine similarity)."""
        query_embedding = self.embedding_model.encode([text]).tolist()
        return self.collection.query(query_embeddings=query_embedding, n_results=n_results)

    def build_answer_prompt(self, question: str, n_results: int = 3) -> str:
        """Retrieve relevant chunks for question and fill the RAG prompt template."""
        results = self.query(question, n_results=n_results)
        context = "\n\n".join(results["documents"][0])
        return build_prompt(context=context, question=question)


_knowledge_base: Optional[SupportKnowledgeBase] = None


def get_knowledge_base() -> SupportKnowledgeBase:
    """Lazily construct (and cache) the module-level knowledge base singleton."""
    global _knowledge_base
    if _knowledge_base is None:
        _knowledge_base = SupportKnowledgeBase()
    return _knowledge_base


def set_knowledge_base(kb: SupportKnowledgeBase) -> None:
    """Override the module-level knowledge base singleton (used by callers and tests)."""
    global _knowledge_base
    _knowledge_base = kb


# --------------------------------------------------------------------------
# Pydantic request/response schema
# --------------------------------------------------------------------------


class AskRequest(BaseModel):
    query: str


class AnswerResponse(BaseModel):
    answer: str
    sources: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


# --------------------------------------------------------------------------
# Optional MOCK_LLM=0 extension: real LLM calls via the Groq API
# --------------------------------------------------------------------------


def _get_groq_client():
    return groq.Groq()


def _llm_classify_intent(query: str) -> str:
    client = _get_groq_client()
    response = client.chat.completions.create(
        model=LLM_MODEL_NAME,
        max_tokens=10,
        messages=[
            {
                "role": "user",
                "content": (
                    "Classify the customer query below as exactly one word: "
                    "'policy_question' if answering it needs Zepto policy information "
                    "(delivery, returns, refunds, membership, tracking, cancellations, "
                    "gift cards, support hours), otherwise 'general_question'.\n\n"
                    f"Query: {query!r}\n\nRespond with only the single classification word."
                ),
            }
        ],
    )
    text = response.choices[0].message.content.strip().lower()
    return "policy_question" if "policy_question" in text else "general_question"


def _generate_structured_answer(
    prompt: str, allowed_sources: list[str], max_retries: int = 2
) -> tuple[str, list[str], float]:
    """Call the LLM requiring strict AnswerResponse-shaped JSON; retry on invalid output.

    Returns (answer, sources, confidence). On repeated validation failure, returns a
    clearly marked error answer with empty sources and zero confidence.
    """
    client = _get_groq_client()
    schema_instruction = (
        "\n\nRespond with ONLY a JSON object matching this schema, no other text: "
        '{"answer": "<string>", "sources": ["<string>", ...], "confidence": <0-1 float>}.'
    )
    current_prompt = prompt + schema_instruction
    last_error: Optional[Exception] = None
    last_raw_text = ""

    for _ in range(max_retries + 1):
        response = client.chat.completions.create(
            model=LLM_MODEL_NAME,
            max_tokens=512,
            messages=[{"role": "user", "content": current_prompt}],
        )
        last_raw_text = response.choices[0].message.content.strip()
        try:
            parsed = AnswerResponse.model_validate_json(last_raw_text)
        except ValidationError as exc:
            last_error = exc
            current_prompt = (
                prompt
                + schema_instruction
                + f"\n\nYour previous response was invalid: {exc}. "
                f"Previous response: {last_raw_text!r}. Return ONLY valid JSON matching the schema."
            )
            continue

        sources = [s for s in parsed.sources if s in allowed_sources] if allowed_sources else []
        return parsed.answer, sources, parsed.confidence

    return (
        f"ERROR: LLM failed to produce a schema-valid response after {max_retries + 1} "
        f"attempts. Last error: {last_error}",
        [],
        0.0,
    )


def _llm_answer_from_context(
    question: str, context: str, retrieved_ids: list[str]
) -> tuple[str, list[str], float]:
    prompt = build_prompt(context=context, question=question)
    answer, sources, confidence = _generate_structured_answer(prompt, allowed_sources=retrieved_ids)
    return answer, (sources or retrieved_ids), confidence


def _llm_direct_answer(question: str) -> tuple[str, float]:
    prompt = f"Answer the following general question directly and concisely:\n\n{question}"
    answer, _, confidence = _generate_structured_answer(prompt, allowed_sources=[])
    return answer, confidence


# --------------------------------------------------------------------------
# LangGraph state and nodes
# --------------------------------------------------------------------------


class GraphState(TypedDict, total=False):
    query: str
    intent: str
    retrieved_ids: list[str]
    retrieved_docs: list[str]
    answer: str
    confidence: float


def classify_intent(state: GraphState) -> dict:
    """Classify the query as policy_question (needs retrieval) or general_question."""
    query = state["query"]

    if MOCK_LLM:
        lowered = query.lower()
        intent = "policy_question" if any(kw in lowered for kw in POLICY_KEYWORDS) else "general_question"
    else:
        intent = _llm_classify_intent(query)

    return {"intent": intent}


def retrieve_and_answer(state: GraphState) -> dict:
    """Retrieve top-3 chunks for policy_question queries and generate the answer."""
    query = state["query"]
    results = get_knowledge_base().query(query, n_results=3)
    ids = results["ids"][0]
    docs = results["documents"][0]

    if MOCK_LLM:
        top_chunk_snippet = docs[0][:200] if docs else ""
        answer = f"Based on the retrieved context: {top_chunk_snippet}"
        confidence = 1.0
    else:
        context = "\n\n".join(docs)
        answer, ids, confidence = _llm_answer_from_context(query, context, ids)

    return {"retrieved_ids": ids, "retrieved_docs": docs, "answer": answer, "confidence": confidence}


def direct_answer(state: GraphState) -> dict:
    """Answer general_question queries with no retrieval."""
    if MOCK_LLM:
        answer = DIRECT_ANSWER_FALLBACK
        confidence = 1.0
    else:
        answer, confidence = _llm_direct_answer(state["query"])

    return {"retrieved_ids": [], "retrieved_docs": [], "answer": answer, "confidence": confidence}


def route_from_classification(state: GraphState) -> str:
    """Route to retrieval for policy_question, otherwise straight to a direct answer."""
    return "retrieve_and_answer" if state["intent"] == "policy_question" else "direct_answer"


def build_graph():
    """Assemble the classify -> (retrieve_and_answer | direct_answer) StateGraph."""
    graph = StateGraph(GraphState)
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("retrieve_and_answer", retrieve_and_answer)
    graph.add_node("direct_answer", direct_answer)

    graph.set_entry_point("classify_intent")
    graph.add_conditional_edges(
        "classify_intent",
        route_from_classification,
        {"retrieve_and_answer": "retrieve_and_answer", "direct_answer": "direct_answer"},
    )
    graph.add_edge("retrieve_and_answer", END)
    graph.add_edge("direct_answer", END)
    return graph.compile()


_graph = None


def get_graph():
    """Lazily build (and cache) the compiled LangGraph."""
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def ask(query: str) -> AnswerResponse:
    """Run the graph for query and return the validated AnswerResponse."""
    final_state = get_graph().invoke({"query": query})
    return AnswerResponse(
        answer=final_state["answer"],
        sources=final_state.get("retrieved_ids", []),
        confidence=final_state.get("confidence", 1.0),
    )


# --------------------------------------------------------------------------
# FastAPI app
# --------------------------------------------------------------------------

@asynccontextmanager
async def _lifespan(app: FastAPI):
    kb = get_knowledge_base()
    if kb.collection.count() == 0:
        kb.build()
    yield


app = FastAPI(title="Zepto Support Assistant", lifespan=_lifespan)


@app.post("/ask")
def ask_endpoint(request: AskRequest) -> AnswerResponse:
    return ask(request.query)


if __name__ == "__main__":
    kb = SupportKnowledgeBase()
    stored = kb.build()
    print(f"Stored {stored} chunks from {len(list(DOCS_DIR.glob('*.txt')))} documents "
          f"in collection '{COLLECTION_NAME}' at {CHROMA_DIR}")
