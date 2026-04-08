"""Local web server for the Study Agent app."""

import os
import time
from datetime import datetime, timezone

import requests as http_requests
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from models import init_db, Category, Document, Chunk, Question, QuestionSet, QuestionSetItem, Attempt
from ingestion import process_pdf
from selection import select_questions, save_as_question_set
from evaluation import evaluate_answer
from retrieval import get_chunk
from source_trace import format_source
from concept_tracking import update_concept_stat, get_weak_concepts, get_all_concept_stats
from config import OLLAMA_BASE_URL

app = FastAPI(title="Study Agent")
Session = init_db()


# ---------------------------------------------------------------------------
# Health / status
# ---------------------------------------------------------------------------
@app.get("/api/health")
def health():
    ollama_ok = False
    try:
        r = http_requests.get(OLLAMA_BASE_URL, timeout=3)
        ollama_ok = r.status_code == 200
    except Exception:
        pass
    return {"status": "ok", "ollama": ollama_ok}


# ---------------------------------------------------------------------------
# Categories
# ---------------------------------------------------------------------------
@app.get("/api/categories")
def list_categories():
    session = Session()
    try:
        cats = session.query(Category).order_by(Category.name).all()
        return [{"id": c.id, "name": c.name} for c in cats]
    finally:
        session.close()


@app.post("/api/categories")
def create_category(name: str = Form(...)):
    session = Session()
    try:
        existing = session.query(Category).filter_by(name=name.strip()).first()
        if existing:
            return {"id": existing.id, "name": existing.name}
        cat = Category(name=name.strip())
        session.add(cat)
        session.commit()
        return {"id": cat.id, "name": cat.name}
    finally:
        session.close()


# ---------------------------------------------------------------------------
# PDF upload
# ---------------------------------------------------------------------------
@app.post("/api/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    category_id: int = Form(...),
    questions_per_chunk: int = Form(3),
):
    session = Session()
    try:
        # Save uploaded file
        upload_dir = "uploads"
        os.makedirs(upload_dir, exist_ok=True)
        pdf_path = os.path.join(upload_dir, file.filename)
        with open(pdf_path, "wb") as f:
            content = await file.read()
            f.write(content)

        result = process_pdf(session, pdf_path, category_id, questions_per_chunk)
        session.commit()
        return {
            "message": "PDF processed successfully",
            "document_id": result["document_id"],
            "num_chunks": result["num_chunks"],
            "num_questions": result["num_questions_generated"],
        }
    except Exception as e:
        session.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Study session
# ---------------------------------------------------------------------------
@app.post("/api/session/start")
def start_session(category_id: int = Form(...)):
    """Select 10 questions and return them as a session."""
    session = Session()
    try:
        questions = select_questions(session, category_id, count=10)
        if not questions:
            raise HTTPException(status_code=404, detail="No questions found for this category")

        qset = save_as_question_set(session, category_id, questions)
        session.commit()

        return {
            "session_id": qset.id,
            "questions": [
                {
                    "id": q.id,
                    "text": q.text,
                    "concept": q.concept,
                    "difficulty": q.difficulty,
                }
                for q in questions
            ],
        }
    except HTTPException:
        raise
    except Exception as e:
        session.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        session.close()


@app.post("/api/session/{session_id}/answer")
def submit_answer(session_id: int, question_id: int, answer: str = Form(...)):
    """Evaluate a single answer and update stats."""
    session = Session()
    try:
        question = session.query(Question).filter_by(id=question_id).first()
        if not question:
            raise HTTPException(status_code=404, detail="Question not found")

        # Evaluate
        result = evaluate_answer(question.text, question.answer, answer)
        is_correct = result["is_correct"]

        # Update question stats
        question.times_asked += 1
        if is_correct:
            question.times_correct += 1

        # Update concept stats
        qset = session.query(QuestionSet).filter_by(id=session_id).first()
        category_id = qset.category_id if qset else None
        if category_id:
            update_concept_stat(session, category_id, question.concept, is_correct)

        # Log attempt
        attempt = Attempt(
            question_id=question_id,
            is_correct=1 if is_correct else 0,
            mistake_type=result.get("mistake_type"),
            explanation=result.get("explanation"),
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        session.add(attempt)
        session.commit()

        # Build response
        resp = {
            "is_correct": is_correct,
            "mistake_type": result.get("mistake_type"),
            "explanation": result.get("explanation"),
            "correct_answer": question.answer,
        }

        # Add source info if wrong
        if not is_correct:
            chunk = get_chunk(session, question.source_chunk_id)
            if chunk:
                resp["source"] = {
                    "page_number": chunk.page_number,
                    "excerpt": format_source(chunk),
                }

        return resp
    except HTTPException:
        raise
    except Exception as e:
        session.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        session.close()


@app.post("/api/session/{session_id}/end")
def end_session(session_id: int):
    """Return session summary."""
    session = Session()
    try:
        qset = session.query(QuestionSet).filter_by(id=session_id).first()
        if not qset:
            raise HTTPException(status_code=404, detail="Session not found")

        attempts = session.query(Attempt).join(QuestionSetItem).filter(
            QuestionSetItem.question_set_id == session_id
        ).all()

        correct = sum(1 for a in attempts if a.is_correct == 1)
        total = len(attempts)

        # Weak concepts
        weak = get_weak_concepts(session, qset.category_id)
        weak_concepts = [
            {"concept": s.concept, "correct": s.correct_count, "wrong": s.wrong_count}
            for s in weak
        ]

        return {
            "score": f"{correct}/{total}",
            "weak_concepts": weak_concepts,
        }
    except HTTPException:
        raise
    except Exception as e:
        session.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Weak concepts
# ---------------------------------------------------------------------------
@app.get("/api/categories/{category_id}/weak-concepts")
def weak_concepts(category_id: int):
    session = Session()
    try:
        weak = get_weak_concepts(session, category_id)
        return [
            {"concept": s.concept, "correct": s.correct_count, "wrong": s.wrong_count}
            for s in weak
        ]
    finally:
        session.close()


@app.get("/api/categories/{category_id}/all-stats")
def all_stats(category_id: int):
    session = Session()
    try:
        stats = get_all_concept_stats(session, category_id)
        return [
            {"concept": s.concept, "correct": s.correct_count, "wrong": s.wrong_count}
            for s in stats
        ]
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Serve frontend
# ---------------------------------------------------------------------------
app.mount("/", StaticFiles(directory="static", html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
