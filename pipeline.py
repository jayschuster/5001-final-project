#!/usr/bin/env python3
"""
Chat Log AI Pipeline
--------------------
Ingests a JSON message log, builds a basic RAG index using Ollama embeddings,
prints a catch-up summary, then enters an interactive chat loop.

Usage:
    python pipeline.py --file logs.json --user Alice
    python pipeline.py --file logs.json --user Alice --model llama3.2 --embed-model nomic-embed-text
"""

import json
import argparse
import sys
import numpy as np
import ollama


# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_CHAT_MODEL  = "devstral-small-2:24b-cloud"
DEFAULT_EMBED_MODEL = "nomic-embed-text"
CHUNK_SIZE          = 10   # messages per chunk
TOP_K               = 3    # chunks to retrieve per query
MAX_HISTORY         = 10   # max chat turns kept in memory


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_logs(path: str) -> list[dict]:
    """Load and lightly validate the JSON log file."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        sys.exit(f"[ERROR] File not found: {path}")
    except json.JSONDecodeError as e:
        sys.exit(f"[ERROR] Invalid JSON: {e}")

    if not isinstance(data, list) or not data:
        sys.exit("[ERROR] Log file must be a non-empty JSON array of message objects.")

    required = {"sender", "timestamp", "message"}
    missing = required - data[0].keys()
    if missing:
        sys.exit(f"[ERROR] Message objects are missing required fields: {missing}")

    return data


def format_message(m: dict) -> str:
    return f"[{m['timestamp']}] {m['sender']}: {m['message']}"


def chunk_messages(messages: list[dict], chunk_size: int = CHUNK_SIZE) -> list[str]:
    """Split messages into fixed-size text chunks."""
    chunks = []
    for i in range(0, len(messages), chunk_size):
        block = messages[i : i + chunk_size]
        chunks.append("\n".join(format_message(m) for m in block))
    return chunks


def get_embedding(text: str, model: str) -> np.ndarray:
    response = ollama.embeddings(model=model, prompt=text)
    return np.array(response["embedding"], dtype=np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else 0.0


def build_index(chunks: list[str], embed_model: str) -> list[np.ndarray]:
    """Embed every chunk and return the embedding list."""
    print(f"  Embedding {len(chunks)} chunk(s) with '{embed_model}'...")
    embeddings = []
    for i, chunk in enumerate(chunks, 1):
        print(f"  [{i}/{len(chunks)}]", end="\r", flush=True)
        embeddings.append(get_embedding(chunk, embed_model))
    print()  # newline after progress
    return embeddings


def retrieve(
    query: str,
    chunks: list[str],
    embeddings: list[np.ndarray],
    embed_model: str,
    top_k: int = TOP_K,
) -> list[str]:
    """Return the top-k most relevant chunks for a query."""
    q_emb = get_embedding(query, embed_model)
    scores = [cosine_similarity(q_emb, e) for e in embeddings]
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [chunks[i] for i in top_indices]


# ── Agents ────────────────────────────────────────────────────────────────────

def run_summarizer(chunks: list[str], user: str, chat_model: str) -> str:
    """Summarize the full log and produce a structured catch-up report."""
    all_logs = "\n\n---\n\n".join(chunks)

    prompt = f"""You are processing team chat logs for {user}, who has been away and needs to catch up.

CHAT LOGS:
{all_logs}

Write a structured catch-up report in plain text. Use these exact section headers:

OVERVIEW
A 2-4 sentence description of what the team was working on.

KEY DECISIONS
A bullet list (use "- ") of any conclusions or agreements reached.

OPEN QUESTIONS / BLOCKERS
A bullet list of anything unresolved, unclear, or blocking progress.

ACTION ITEMS FOR {user.upper()}
A bullet list of tasks directed at or relevant to {user}. Include why each task matters. Make sure to use "you" and "yours" rather than "{user}" and the user's pronouns.
If {user} is not mentioned by name, list general next steps visible in the logs.

Rules:
- Plain text only, no markdown symbols like ** or ##.
- Do not invent any details not present in the logs.
- If a section has nothing to report, write "None identified."
"""

    response = ollama.chat(
        model=chat_model,
        messages=[{"role": "user", "content": prompt}],
    )
    return response["message"]["content"]


def run_chat(
    query: str,
    chunks: list[str],
    embeddings: list[np.ndarray],
    user: str,
    history: list[dict],
    chat_model: str,
    embed_model: str,
) -> str:
    """Answer a user question grounded in retrieved log chunks."""
    context_chunks = retrieve(query, chunks, embeddings, embed_model)
    context = "\n\n---\n\n".join(context_chunks)

    system_prompt = f"""You are a chat assistant helping {user} understand their team's chat logs.
Answer questions using ONLY the context extracted from the logs provided in each message.
If the answer is not clearly present in the context, say: "I don't see that in the logs."
Be explanative and direct. Plain text only — no bullet symbols, no markdown, no ** for bolding.
Make sure to ask followup questions to narrow down specifics.
Address the user directly, replacing their name ({user}) with "you"."""

    messages = [{"role": "system", "content": system_prompt}]
    messages += history
    messages.append({
        "role": "user",
        "content": f"Relevant log excerpts:\n{context}\n\nQuestion: {query}",
    })

    response = ollama.chat(model=chat_model, messages=messages)
    return response["message"]["content"]


# ── Entry Point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="AI agent pipeline for team chat logs (Ollama-backed)."
    )
    parser.add_argument("--file",        required=True,                    help="Path to the JSON log file.")
    parser.add_argument("--user",        required=True,                    help="Your name, for task attribution.")
    parser.add_argument("--model",       default=DEFAULT_CHAT_MODEL,       help=f"Ollama chat model.  (default: {DEFAULT_CHAT_MODEL})")
    parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL,      help=f"Ollama embedding model. (default: {DEFAULT_EMBED_MODEL})")
    args = parser.parse_args()

    chat_model  = args.model
    embed_model = args.embed_model

    # ── Stage 1: Ingestion ────────────────────────────────────────────────────
    print(f"\n[1/3] Loading logs from '{args.file}'...")
    messages = load_logs(args.file)
    print(f"  Loaded {len(messages)} message(s).")

    chunks = chunk_messages(messages)
    print(f"  Split into {len(chunks)} chunk(s) of up to {CHUNK_SIZE} messages each.")

    # ── Stage 2: RAG Encoding ─────────────────────────────────────────────────
    print(f"\n[2/3] Building RAG index...")
    embeddings = build_index(chunks, embed_model)
    print(f"  Index ready.")

    # ── Stage 3: Summarization ────────────────────────────────────────────────
    print(f"\n[3/3] Generating catch-up summary with '{chat_model}'...")
    summary = run_summarizer(chunks, args.user, chat_model)

    print("\n" + "=" * 60)
    print(" CATCH-UP SUMMARY")
    print("=" * 60)
    print(summary)
    print("=" * 60)

    # ── Stage 4: Interactive Chat ─────────────────────────────────────────────
    print(f"\nChat mode active. Ask anything about the logs.")
    print("Type 'exit' or press Ctrl+C to quit.\n")

    history: list[dict] = []

    while True:
        try:
            query = input(f"{args.user}> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not query:
            continue

        if query.lower() in ("exit", "quit", "q"):
            print("Goodbye!")
            break

        answer = run_chat(
            query, chunks, embeddings,
            args.user, history,
            chat_model, embed_model,
        )
        print(f"\nAgent: {answer}\n")

        # Rolling history window to avoid context bloat
        history.append({"role": "user",      "content": query})
        history.append({"role": "assistant", "content": answer})
        if len(history) > MAX_HISTORY * 2:
            history = history[-(MAX_HISTORY * 2):]


if __name__ == "__main__":
    main()
