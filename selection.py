"""Simple selection engine for study sessions.

Selects exactly 10 questions per session, prioritizing concepts the user
struggles with and questions they have seen least.
"""

from datetime import datetime, timezone

from models import Question, QuestionSet, QuestionSetItem, ConceptStat, Chunk, Document


def _get_category_questions(session, category_id):
    """Return all Question objects belonging to the given category.

    Traversal: category -> documents -> chunks -> questions.
    """
    return (
        session.query(Question)
        .join(Chunk, Question.source_chunk_id == Chunk.id)
        .join(Document, Chunk.document_id == Document.id)
        .filter(Document.category_id == category_id)
        .all()
    )


def _get_concept_stats(session, category_id):
    """Return a dict mapping concept name -> ConceptStat for the category."""
    stats = (
        session.query(ConceptStat)
        .filter(ConceptStat.category_id == category_id)
        .all()
    )
    return {stat.concept: stat for stat in stats}


def _score_question(question, concept_stats):
    """Compute a priority score for a single question.

    Higher score means the question should be picked sooner.

    Components:
      - concept_wrong_count: high wrong count on the concept => higher score
      - times_asked: low exposure => higher score
      - times_correct: low mastery => higher score

    The wrong_count dominates so that struggling concepts surface first.
    Within the same concept, less-asked questions rise to the top.
    """
    stat = concept_stats.get(question.concept)
    wrong_count = stat.wrong_count if stat else 0

    # Negate asked/correct so that lower values produce higher scores.
    # Add 1 to avoid division by zero and give unseen questions a boost.
    asked_factor = 1.0 / (question.times_asked + 1)
    correct_factor = 1.0 / (question.times_correct + 1)

    # Weight: wrong_count is the primary driver; asked/correct are tiebreakers.
    score = wrong_count * 10.0 + asked_factor + correct_factor
    return score


def _ensure_difficulty_mix(selected, remaining):
    """Try to include a mix of difficulty levels in the selected questions.

    If all selected questions share the same difficulty, attempt to swap
    some out for questions at other difficulty levels from the remaining
    pool (which is already sorted by score descending).

    Parameters:
      selected: list of Question objects currently chosen (top by score).
      remaining: list of Question objects not yet chosen (also sorted by score).

    Returns a list of Question objects.
    """
    if not selected:
        return selected

    difficulties = {q.difficulty for q in selected}
    if len(difficulties) > 1:
        # Already a mix -- nothing to do.
        return selected

    # All same difficulty; try to swap in questions of other difficulties.
    single_difficulty = difficulties.pop()
    swapped = list(selected)

    for candidate in remaining:
        if candidate.difficulty != single_difficulty:
            # Replace the lowest-scored selected item with this candidate.
            swapped[-1] = candidate
            # Re-check if we now have a mix.
            new_difficulties = {q.difficulty for q in swapped}
            if len(new_difficulties) > 1:
                break

    return swapped


def select_questions(session, category_id, count=10):
    """Select exactly *count* questions for a study session.

    Algorithm:
      1. Fetch all questions for the category (via chunks -> documents).
      2. Fetch concept stats for the category.
      3. Score each question (higher wrong_count + lower asked/correct = higher score).
      4. Sort by score descending, pick top *count*.
      5. Attempt to include a mix of difficulty levels.

    Returns a list of Question objects (length may be < count if fewer
    questions exist in the database).
    """
    questions = _get_category_questions(session, category_id)
    if not questions:
        return []

    concept_stats = _get_concept_stats(session, category_id)

    # Score and sort.
    scored = [(q, _score_question(q, concept_stats)) for q in questions]
    scored.sort(key=lambda pair: pair[1], reverse=True)

    # Deduplicate by question id (safety net -- should not happen with
    # a clean schema, but guards against data issues).
    seen_ids = set()
    unique_scored = []
    for q, score in scored:
        if q.id not in seen_ids:
            seen_ids.add(q.id)
            unique_scored.append((q, score))

    # Pick top N and try to ensure difficulty mix.
    selected = [q for q, _ in unique_scored[:count]]
    remaining = [q for q, _ in unique_scored[count:]]
    result = _ensure_difficulty_mix(selected, remaining)

    return result


def save_as_question_set(session, category_id, questions):
    """Persist a list of Question objects as a QuestionSet with QuestionSetItems.

    Returns the newly created QuestionSet object.
    """
    now = datetime.now(timezone.utc).isoformat()

    qset = QuestionSet(category_id=category_id, created_at=now)
    session.add(qset)
    session.flush()  # populate qset.id

    for question in questions:
        item = QuestionSetItem(question_set_id=qset.id, question_id=question.id)
        session.add(item)

    session.flush()
    return qset
