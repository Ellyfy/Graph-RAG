"""
run_autoq_queries.py

Batch-runs 30 AutoQ-style evaluation queries against the GraphRAG pipeline
built in graph.py. Saves results to graphrag_autoq_results.json for later
RAGAs + LLM-as-Judge evaluation.

Usage (from Graph_RAG project root, with .venv activated):
    python run_autoq_queries.py

Before running:
  1. Neo4j Desktop instance must be RUNNING
  2. The Honeywell knowledge graph must already be built (domain=honeywell)
  3. .env must have OPENAI_API_KEY and NEO4J credentials set
  4. Streamlit does NOT need to be running - this script calls graph.py directly
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------

QUERIES_FILE = "honeywell_autoq_queries.json"
OUTPUT_FILE = "graphrag_autoq_results.json"

# Matches the domain name you used in the Streamlit sidebar when you
# clicked "Build Knowledge Graph" for the Honeywell data.
# If you used something else, change this value.
DOMAIN = "honeywell_products"

# Cheaper, faster model for querying. Change to "gpt-4o" if you want higher quality.
GEN_MODEL = "gpt-4o-mini"

# Pause between queries to stay well under any rate limits (seconds)
SLEEP_BETWEEN_QUERIES = 1.0

# ----------------------------------------------------------------------------
# IMPORTS FROM PROJECT
# ----------------------------------------------------------------------------

# These imports assume the script lives in the Graph_RAG project root,
# right next to graph.py and config.py.

try:
    from graph import Neo4jClient, query_graph_rag
except ImportError as e:
    print("ERROR: Could not import from graph.py.")
    print("Make sure this script is in the Graph_RAG project root folder.")
    print(f"Detail: {e}")
    sys.exit(1)


# ----------------------------------------------------------------------------
# HELPERS
# ----------------------------------------------------------------------------

def load_queries(path: str) -> list[dict[str, Any]]:
    """Load the 30 AutoQ queries from JSON."""
    qpath = Path(path)
    if not qpath.exists():
        print(f"ERROR: Queries file not found: {qpath.resolve()}")
        print("Download honeywell_autoq_queries.json into this folder.")
        sys.exit(1)
    data = json.loads(qpath.read_text(encoding="utf-8"))
    return data["queries"]


def run_single_query(driver, question: str) -> dict[str, Any]:
    """
    Call query_graph_rag and normalize its output.
    Returns a dict with keys: answer, contexts, error.
    """
    try:
        result = query_graph_rag(driver, question, GEN_MODEL, DOMAIN)

        # query_graph_rag typically returns an object/dict with .answer and
        # retrieved context records. We handle both common shapes.
        answer = None
        contexts: list[str] = []

        if isinstance(result, dict):
            answer = result.get("answer") or result.get("response") or str(result)
            raw_ctx = (
                result.get("retriever_result")
                or result.get("contexts")
                or result.get("context")
                or []
            )
        else:
            # neo4j-graphrag GraphRAG.search returns a RagResultModel
            answer = getattr(result, "answer", None) or str(result)
            raw_ctx = getattr(result, "retriever_result", None) or []
            if hasattr(raw_ctx, "items"):
                raw_ctx = raw_ctx.items

        # Normalize contexts to a list of strings
        if raw_ctx:
            for item in raw_ctx:
                if isinstance(item, str):
                    contexts.append(item)
                elif hasattr(item, "content"):
                    contexts.append(str(item.content))
                else:
                    contexts.append(str(item))

        return {"answer": answer, "contexts": contexts, "error": None}

    except Exception as e:
        return {
            "answer": None,
            "contexts": [],
            "error": f"{type(e).__name__}: {e}",
        }


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------

def main():
    print("=" * 72)
    print(" Batch runner: 30 AutoQ queries against GraphRAG")
    print("=" * 72)

    queries = load_queries(QUERIES_FILE)
    print(f"Loaded {len(queries)} queries from {QUERIES_FILE}")
    print(f"Domain: {DOMAIN}  |  Model: {GEN_MODEL}")
    print()

    # Connect to Neo4j once, reuse for all queries
    print("Connecting to Neo4j...")
    try:
        client = Neo4jClient()
        driver = client()
    except Exception as e:
        print(f"ERROR connecting to Neo4j: {e}")
        print("Make sure your Neo4j Desktop instance is RUNNING and .env is correct.")
        sys.exit(1)
    print("Connected.\n")

    results = []
    t_start = time.time()

    for i, q in enumerate(queries, start=1):
        qid = q["id"]
        qclass = q["class"]
        question = q["question"]

        print(f"[{i:2d}/{len(queries)}] {qid} ({qclass})")
        print(f"         Q: {question[:80]}{'...' if len(question) > 80 else ''}")

        t0 = time.time()
        out = run_single_query(driver, question)
        elapsed = time.time() - t0

        entry = {
            "id": qid,
            "class": qclass,
            "question": question,
            "graphrag_answer": out["answer"],
            "retrieved_contexts": out["contexts"],
            "error": out["error"],
            "latency_seconds": round(elapsed, 2),
        }
        results.append(entry)

        if out["error"]:
            print(f"         ERROR: {out['error']}")
        else:
            ans = (out["answer"] or "").strip().replace("\n", " ")
            print(f"         A: {ans[:120]}{'...' if len(ans) > 120 else ''}")
            print(f"         ({len(out['contexts'])} chunks, {elapsed:.1f}s)")
        print()

        # Checkpoint save every 5 queries so we don't lose progress
        if i % 5 == 0:
            Path(OUTPUT_FILE).write_text(
                json.dumps(results, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            print(f"    [checkpoint saved: {i} queries done]\n")

        time.sleep(SLEEP_BETWEEN_QUERIES)

    # Final save
    Path(OUTPUT_FILE).write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    total_elapsed = time.time() - t_start
    n_ok = sum(1 for r in results if not r["error"])
    n_err = len(results) - n_ok

    print("=" * 72)
    print(f" DONE in {total_elapsed/60:.1f} min")
    print(f"   Success: {n_ok}/{len(results)}")
    print(f"   Errors:  {n_err}/{len(results)}")
    print(f"   Saved to: {OUTPUT_FILE}")
    print("=" * 72)

    # Close driver
    try:
        driver.close()
    except Exception:
        pass


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user. Partial results saved.")
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        traceback.print_exc()
        sys.exit(1)
