"""Session orchestrator -- controls the complete study session flow.

Provides three functions:
- list_categories: print and return all available categories
- select_category: interactive prompt for the user to pick a category
- run_study_session: linear flow through 10 questions with evaluation and stats
"""

from datetime import datetime, timezone

from models import Category, Question, QuestionSet, Attempt
from selection import select_questions, save_as_question_set
from evaluation import evaluate_answer
from retrieval import get_chunk
from source_trace import format_source
from concept_tracking import update_concept_stat, get_weak_concepts


def list_categories(session):
    """Print all available categories and return them as a list.

    Args:
        session: SQLAlchemy session.

    Returns:
        List of Category objects.
    """
    categories = session.query(Category).order_by(Category.name).all()

    if not categories:
        print("No categories found. Ingest some documents first.")
        return []

    print("\n=== Available Categories ===")
    for cat in categories:
        print(f"  [{cat.id}] {cat.name}")
    print()

    return categories


def select_category(session):
    """Interactively ask the user to pick a category.

    Args:
        session: SQLAlchemy session.

    Returns:
        The selected category_id (int), or None if no categories exist.
    """
    categories = list_categories(session)
    if not categories:
        return None

    valid_ids = {cat.id for cat in categories}

    while True:
        raw = input("Select a category (enter number): ").strip()
        if not raw:
            print("Please enter a number.")
            continue

        try:
            category_id = int(raw)
        except ValueError:
            print(f"Invalid input '{raw}'. Please enter a number.")
            continue

        if category_id not in valid_ids:
            print(f"Category {category_id} does not exist. Choose from the list above.")
            continue

        selected = session.query(Category).filter_by(id=category_id).first()
        print(f"\nCategory: {selected.name}\n")
        return category_id


def run_study_session(session, category_id):
    """Run a complete study session for the given category.

    Flow:
      1. Select 10 questions from the database.
      2. Show each question one by one, collect the user's answer.
      3. Evaluate the answer and show feedback (with source if wrong).
      4. Update question stats, concept stats, and attempt logs.
      5. Print a summary at the end.

    Args:
        session: SQLAlchemy session.
        category_id: The category to study.

    Returns:
        A dict with session results: total, correct, wrong, weak_concepts.
    """
    category = session.query(Category).filter_by(id=category_id).first()
    if category is None:
        print(f"Category {category_id} not found.")
        return {"total": 0, "correct": 0, "wrong": 0, "weak_concepts": []}

    # Step 1: Select questions
    questions = select_questions(session, category_id, count=10)
    if not questions:
        print(f"No questions available for category '{category.name}'.")
        return {"total": 0, "correct": 0, "wrong": 0, "weak_concepts": []}

    # Persist this selection as a question set for traceability
    qset = save_as_question_set(session, category_id, questions)

    total = len(questions)
    correct_count = 0
    wrong_count = 0

    print(f"=== Study Session ===")
    print(f"Category: {category.name}")
    print(f"Questions: {total}")
    print()

    # Step 2-4: Iterate through questions
    for idx, question in enumerate(questions, start=1):
        print(f"Question {idx}/{total}:")
        print(question.text)
        print()

        user_answer = input("Your answer: ").strip()

        # Evaluate
        result = evaluate_answer(question.text, question.answer, user_answer)
        is_correct = result["is_correct"]

        if is_correct:
            correct_count += 1
            print("\u2713 Correct!")
        else:
            wrong_count += 1
            print(f"\u2717 Incorrect.")
            print(f"  Correct answer: {question.answer}")
            print(f"  Mistake type: {result['mistake_type']}")
            print(f"  Explanation: {result['explanation']}")

            # Show source reference for wrong answers
            chunk = get_chunk(session, question.source_chunk_id)
            if chunk is not None:
                print()
                print(format_source(chunk))

        print()

        # Step 4a: Update question stats
        question.times_asked += 1
        if is_correct:
            question.times_correct += 1

        # Step 4b: Update concept stat
        update_concept_stat(session, category_id, question.concept, is_correct)

        # Step 4c: Log the attempt
        now = datetime.now(timezone.utc).isoformat()
        attempt = Attempt(
            question_id=question.id,
            is_correct=1 if is_correct else 0,
            mistake_type=None if is_correct else result["mistake_type"],
            explanation=result["explanation"],
            created_at=now,
        )
        session.add(attempt)

        # Flush after each question so stats are persisted incrementally
        session.flush()

    # Step 5: Print summary
    weak_concepts = get_weak_concepts(session, category_id)

    print("=== Session Complete ===")
    print(f"Score: {correct_count}/{total}")

    if weak_concepts:
        weak_names = [wc.concept for wc in weak_concepts]
        print(f"Weak concepts: {', '.join(weak_names)}")
    else:
        print("No weak concepts detected. Great job!")

    print()

    return {
        "total": total,
        "correct": correct_count,
        "wrong": wrong_count,
        "weak_concepts": [wc.concept for wc in weak_concepts],
    }
