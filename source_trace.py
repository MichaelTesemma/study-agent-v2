"""Source tracing module -- tells users where to study based on a chunk."""

from models import Chunk


def get_important_lines(chunk_text: str, max_lines: int = 5) -> str:
    """Extract the most important lines from chunk text.

    Simply takes the first *max_lines* non-empty lines from the text.
    """
    lines = [line.strip() for line in chunk_text.splitlines() if line.strip()]
    selected = lines[:max_lines]
    return "\n".join(selected)


def format_source(chunk: Chunk, max_lines: int = 5) -> str:
    """Return a formatted study-reference string for a single Chunk.

    Output includes the page number, a short excerpt, and a brief
    explanation directing the user to study that section.
    """
    excerpt = get_important_lines(chunk.text, max_lines=max_lines)

    # Build a one-line explanation from the first meaningful sentence
    first_sentence = excerpt.split(".")[0].strip()
    if first_sentence:
        explanation = f"Study this section to understand {first_sentence.lower()}."
    else:
        explanation = "Study this section for key concepts."

    return (
        f"\U0001f4d6 Page {chunk.page_number}\n"
        "---\n"
        "Important excerpt:\n"
        f'"{excerpt}"\n'
        "---\n"
        f"{explanation}"
    )
