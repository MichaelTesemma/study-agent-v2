"""Simple retrieval module for the 'show source in book' feature.

Provides two functions:
- get_chunk: fetch a single chunk by its ID from the database
- get_similar_chunks: find similar chunks in Chroma using the same document's collection
"""

import chromadb

from models import Chunk
from config import CHROMA_PERSIST_DIR


def get_chunk(session, chunk_id: int) -> Chunk | None:
    """Fetch a chunk from the database by its ID.

    Args:
        session: SQLAlchemy session.
        chunk_id: The primary key of the chunk.

    Returns:
        The Chunk ORM object (with .text and .page_number), or None if not found.
    """
    return session.query(Chunk).filter_by(id=chunk_id).first()


def get_similar_chunks(session, chunk_id: int, n_results: int = 2) -> list[dict]:
    """Query Chroma for chunks similar to the given chunk.

    Uses the chunk's document_id to locate the correct Chroma collection,
    then performs a simple similarity query.

    Args:
        session: SQLAlchemy session.
        chunk_id: The primary key of the source chunk.
        n_results: Number of similar chunks to return (default 2).

    Returns:
        A list of dicts, each with keys: text, page_number, distance.
        The source chunk itself is excluded from results.
    """
    chunk = get_chunk(session, chunk_id)
    if chunk is None:
        return []

    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    collection_name = f"document_{chunk.document_id}"

    try:
        collection = client.get_collection(name=collection_name)
    except ValueError:
        # Collection does not exist -- no embeddings stored for this document
        return []

    results = collection.query(
        query_texts=[chunk.text],
        n_results=n_results + 1,  # +1 because the source chunk itself will be returned
    )

    similar = []
    ids = results.get("ids", [[]])[0]
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for i, doc_id in enumerate(ids):
        # Skip the source chunk itself
        if doc_id == str(chunk_id):
            continue
        similar.append({
            "text": documents[i],
            "page_number": metadatas[i].get("page_number"),
            "distance": distances[i],
        })
        # Stop once we have enough results (after excluding the source)
        if len(similar) >= n_results:
            break

    return similar
