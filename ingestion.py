"""PDF processing pipeline for the study app.

Pipeline steps (linear, one-pass):
1. Extract text from PDF using pypdf
2. Split text into chunks (paragraph-based, max ~300-500 tokens)
3. Store chunks in SQLite
4. Generate embeddings and store in Chroma
5. Generate questions from chunks using Ollama (local LLM)
6. Store questions in SQLite
"""

import json
import os
import re
import time

import chromadb
import requests
import tiktoken
from pypdf import PdfReader

from config import OLLAMA_BASE_URL, OLLAMA_MODEL, CHROMA_PERSIST_DIR
from models import Document, Chunk, Question, ConceptStat
from embeddings import generate_embedding

# Tokenizer for counting tokens (cl100k_base = GPT-4 / GPT-3.5 encoding)
TOKENIZER = tiktoken.get_encoding("cl100k_base")

# Maximum tokens per chunk
MAX_CHUNK_TOKENS = 400

# Ollama API endpoint
OLLAMA_GENERATE_URL = f"{OLLAMA_BASE_URL}/api/generate"


# ---------------------------------------------------------------------------
# Step 1: Extract text from PDF
# ---------------------------------------------------------------------------
def extract_text_from_pdf(pdf_path: str) -> list[tuple[str, int]]:
    """Extract text from each page of a PDF.

    Returns a list of (text, page_number) tuples, one per page.
    """
    reader = PdfReader(pdf_path)
    pages = []
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages.append((text, page_num))
    return pages


# ---------------------------------------------------------------------------
# Step 2: Split text into chunks
# ---------------------------------------------------------------------------
def count_tokens(text: str) -> int:
    """Count the number of tokens in a string."""
    return len(TOKENIZER.encode(text))


def split_into_chunks(pages: list[tuple[str, int]]) -> list[dict]:
    """Split page text into paragraph-based chunks of max ~MAX_CHUNK_TOKENS tokens.

    Strategy:
    - Split each page into paragraphs (double-newline boundaries)
    - Accumulate paragraphs into a chunk until the token limit is reached
    - When the limit would be exceeded, flush the current chunk and start a new one

    Returns a list of dicts with keys: text, page_number
    """
    chunks = []

    for page_text, page_number in pages:
        # Split page into paragraphs
        paragraphs = re.split(r"\n\s*\n", page_text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        current_chunk_parts = []
        current_chunk_tokens = 0

        for paragraph in paragraphs:
            paragraph_tokens = count_tokens(paragraph)

            # If a single paragraph exceeds the limit, split it by sentences
            if paragraph_tokens > MAX_CHUNK_TOKENS:
                # Flush current chunk first if it has content
                if current_chunk_parts:
                    chunks.append({
                        "text": "\n\n".join(current_chunk_parts),
                        "page_number": page_number,
                    })
                    current_chunk_parts = []
                    current_chunk_tokens = 0

                # Split oversized paragraph by sentences
                sentences = re.split(r"(?<=[.!?])\s+", paragraph)
                sentence_buffer = ""
                sentence_buffer_tokens = 0

                for sentence in sentences:
                    sentence_tokens = count_tokens(sentence)

                    if sentence_buffer_tokens + sentence_tokens > MAX_CHUNK_TOKENS:
                        if sentence_buffer:
                            chunks.append({
                                "text": sentence_buffer,
                                "page_number": page_number,
                            })
                        sentence_buffer = sentence
                        sentence_buffer_tokens = sentence_tokens
                    else:
                        sentence_buffer += " " + sentence if sentence_buffer else sentence
                        sentence_buffer_tokens += sentence_tokens

                if sentence_buffer:
                    chunks.append({
                        "text": sentence_buffer,
                        "page_number": page_number,
                    })

            # Normal paragraph: try to fit into current chunk
            elif current_chunk_tokens + paragraph_tokens > MAX_CHUNK_TOKENS:
                # Flush current chunk
                if current_chunk_parts:
                    chunks.append({
                        "text": "\n\n".join(current_chunk_parts),
                        "page_number": page_number,
                    })
                # Start new chunk with this paragraph
                current_chunk_parts = [paragraph]
                current_chunk_tokens = paragraph_tokens
            else:
                current_chunk_parts.append(paragraph)
                current_chunk_tokens += paragraph_tokens

        # Flush remaining content on this page
        if current_chunk_parts:
            chunks.append({
                "text": "\n\n".join(current_chunk_parts),
                "page_number": page_number,
            })

    return chunks


# ---------------------------------------------------------------------------
# Step 3: Store chunks in SQLite
# ---------------------------------------------------------------------------
def store_chunks(session, chunks: list[dict], document_id: int) -> list[Chunk]:
    """Insert all chunks into the database and return the ORM objects."""
    db_chunks = []
    for chunk_data in chunks:
        chunk = Chunk(
            document_id=document_id,
            page_number=chunk_data["page_number"],
            text=chunk_data["text"],
        )
        session.add(chunk)
        db_chunks.append(chunk)

    session.flush()  # Assign IDs without committing
    return db_chunks


# ---------------------------------------------------------------------------
# Step 4: Generate embeddings and store in Chroma
# ---------------------------------------------------------------------------
def store_embeddings(chunks: list[Chunk], document_id: int):
    """Generate embeddings for each chunk and store them in Chroma.

    Uses sentence-transformers for consistent high-quality embeddings.
    """
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)

    # Get or create a collection per document
    collection_name = f"document_{document_id}"
    collection = client.get_or_create_collection(name=collection_name)

    ids = []
    documents = []
    metadatas = []
    texts = []

    for chunk in chunks:
        chunk_id = str(chunk.id)
        ids.append(chunk_id)
        documents.append(chunk.text)
        texts.append(chunk.text)
        metadatas.append({
            "document_id": document_id,
            "page_number": chunk.page_number,
        })

    # Generate embeddings using sentence-transformers
    print(f"Generating embeddings for {len(texts)} chunks...")
    embeddings = generate_embedding(texts)

    # Upsert embeddings (add or update)
    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings,
    )


# ---------------------------------------------------------------------------
# Step 5: Generate questions from chunks using Ollama
# ---------------------------------------------------------------------------
def generate_questions_for_chunk(chunk: Chunk, num_questions: int) -> list[dict]:
    """Send a chunk to Ollama and get back study questions as JSON.

    Returns a list of dicts with keys:
        question_text, answer_text, concept, difficulty (1-5), source_chunk_id
    """
    prompt = (
        f"Generate {num_questions} multiple-choice style study questions from this text. "
        f"Each question should have: question_text, answer_text, concept, difficulty (1-5). "
        f"Return as a JSON array only, no extra text.\n\n"
        f"Text:\n{chunk.text}"
    )

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "format": "json",
    }

    try:
        response = requests.post(
            OLLAMA_GENERATE_URL,
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
        result = response.json()
        raw_response = result.get("response", "")

        # Parse the JSON response
        data = json.loads(raw_response)

        # Ollama may return a single object or a list of objects.
        # Normalize to a list either way.
        if isinstance(data, dict):
            questions = [data]
        elif isinstance(data, list):
            questions = data
        else:
            return []

        # Validate and attach source_chunk_id
        validated = []
        for q in questions:
            if all(k in q for k in ("question_text", "answer_text", "concept", "difficulty")):
                validated.append({
                    "question_text": str(q["question_text"]),
                    "answer_text": str(q["answer_text"]),
                    "concept": str(q["concept"]),
                    "difficulty": max(1, min(5, int(q["difficulty"]))),
                    "source_chunk_id": chunk.id,
                })

        return validated

    except (requests.RequestException, json.JSONDecodeError, KeyError) as e:
        print(f"Warning: Failed to generate questions for chunk {chunk.id}: {e}")
        return []


# ---------------------------------------------------------------------------
# Step 6: Store questions in SQLite
# ---------------------------------------------------------------------------
def store_questions(session, questions: list[dict]) -> list[Question]:
    """Insert all questions into the database and return the ORM objects."""
    db_questions = []
    for q_data in questions:
        question = Question(
            source_chunk_id=q_data["source_chunk_id"],
            concept=q_data["concept"],
            difficulty=q_data["difficulty"],
            text=q_data["question_text"],
            answer=q_data["answer_text"],
            times_asked=0,
            times_correct=0,
        )
        session.add(question)
        db_questions.append(question)

    session.flush()
    return db_questions


# ---------------------------------------------------------------------------
# Update ConceptStats
# ---------------------------------------------------------------------------
def update_concept_stats(session, questions: list[Question], category_id: int):
    """Create or update ConceptStat entries for each concept found in questions."""
    # Group questions by concept
    concept_counts = {}
    for q in questions:
        concept = q.concept
        if concept not in concept_counts:
            concept_counts[concept] = 0
        concept_counts[concept] += 1

    # Upsert ConceptStat rows
    for concept, count in concept_counts.items():
        existing = (
            session.query(ConceptStat)
            .filter_by(category_id=category_id, concept=concept)
            .first()
        )
        if existing is None:
            stat = ConceptStat(
                category_id=category_id,
                concept=concept,
                correct_count=0,
                wrong_count=0,
            )
            session.add(stat)


# ---------------------------------------------------------------------------
# Main pipeline entry point
# ---------------------------------------------------------------------------
def process_pdf(
    session,
    pdf_path: str,
    category_id: int,
    number_of_questions: int,
    max_chunks: int | None = None,
):
    """Process a PDF file through the full ingestion pipeline.

    Args:
        session: SQLAlchemy session (must not be committed yet).
        pdf_path: Path to the PDF file on disk.
        category_id: The category this document belongs to.
        number_of_questions: How many questions to generate per chunk.
        max_chunks: Optional limit on how many chunks to process from the PDF.

    Returns:
        dict with counts: document_id, num_chunks, num_questions_generated
    """
    # --- Validate input ---
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    filename = os.path.basename(pdf_path)
    abs_path = os.path.abspath(pdf_path)

    # --- Step 1: Extract text ---
    print(f"[1/6] Extracting text from {filename}...")
    pages = extract_text_from_pdf(pdf_path)
    total_pages = len(pages)
    print(f"      Found {total_pages} pages.")

    # --- Step 2: Split into chunks ---
    print("[2/6] Splitting text into chunks...")
    chunk_data = split_into_chunks(pages)
    total_chunks = len(chunk_data)
    if max_chunks is not None:
        chunk_data = chunk_data[:max_chunks]
        print(f"      Created {total_chunks} chunks, processing {len(chunk_data)}.")
    else:
        print(f"      Created {total_chunks} chunks.")

    # --- Step 3: Store document and chunks in SQLite ---
    print("[3/6] Storing document and chunks in SQLite...")
    document = Document(
        category_id=category_id,
        filename=filename,
        file_path=abs_path,
    )
    session.add(document)
    session.flush()  # Get document.id

    db_chunks = store_chunks(session, chunk_data, document.id)
    print(f"      Stored {len(db_chunks)} chunks (document_id={document.id}).")

    # --- Step 4: Generate embeddings in Chroma ---
    print("[4/6] Generating embeddings and storing in Chroma...")
    store_embeddings(db_chunks, document.id)
    print("      Embeddings stored.")

    # --- Step 5: Generate questions via Ollama ---
    print(f"[5/6] Generating questions using Ollama ({OLLAMA_MODEL})...")
    all_questions = []
    for i, chunk in enumerate(db_chunks, start=1):
        print(f"      Processing chunk {i}/{len(db_chunks)}...")
        questions = generate_questions_for_chunk(chunk, number_of_questions)
        all_questions.extend(questions)
        # Small delay to avoid overwhelming the local LLM
        time.sleep(0.5)

    print(f"      Generated {len(all_questions)} questions total.")

    # --- Step 6: Store questions in SQLite ---
    print("[6/6] Storing questions in SQLite...")
    db_questions = store_questions(session, all_questions)
    update_concept_stats(session, db_questions, category_id)
    print(f"      Stored {len(db_questions)} questions.")

    return {
        "document_id": document.id,
        "num_chunks": len(db_chunks),
        "total_chunks": total_chunks,
        "num_questions_generated": len(db_questions),
    }
