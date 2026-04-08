"""Simple concept statistics tracking for the study app.

Tracks correct/wrong counts per concept per category and identifies weak concepts.
"""

from sqlalchemy import desc

from models import ConceptStat


def update_concept_stat(session, category_id, concept, is_correct):
    """Find or create a ConceptStat row and increment the appropriate counter.

    Args:
        session: SQLAlchemy session.
        category_id: The category this concept belongs to.
        concept: The concept name (string).
        is_correct: Boolean -- True increments correct_count, False increments wrong_count.

    Returns:
        The updated (or newly created) ConceptStat instance.
    """
    stat = (
        session.query(ConceptStat)
        .filter_by(category_id=category_id, concept=concept)
        .first()
    )

    if stat is None:
        stat = ConceptStat(
            category_id=category_id,
            concept=concept,
            correct_count=0,
            wrong_count=0,
        )
        session.add(stat)
        session.flush()

    if is_correct:
        stat.correct_count += 1
    else:
        stat.wrong_count += 1

    return stat


def get_weak_concepts(session, category_id):
    """Return concepts where wrong_count > correct_count.

    Args:
        session: SQLAlchemy session.
        category_id: The category to filter by.

    Returns:
        List of ConceptStat instances that are considered weak.
    """
    return (
        session.query(ConceptStat)
        .filter_by(category_id=category_id)
        .filter(ConceptStat.wrong_count > ConceptStat.correct_count)
        .all()
    )


def get_all_concept_stats(session, category_id):
    """Return all concept stats for a category, sorted by wrong_count descending.

    Args:
        session: SQLAlchemy session.
        category_id: The category to filter by.

    Returns:
        List of all ConceptStat instances for the category.
    """
    return (
        session.query(ConceptStat)
        .filter_by(category_id=category_id)
        .order_by(desc(ConceptStat.wrong_count))
        .all()
    )
