"""Minimal SQLAlchemy schema for a single-user study app."""

from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    Float,
    ForeignKey,
    create_engine,
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

from config import DATABASE_URL

Base = declarative_base()


# ---------------------------------------------------------------------------
# Category
# ---------------------------------------------------------------------------
class Category(Base):
    __tablename__ = "categories"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False, unique=True)

    documents = relationship("Document", back_populates="category", cascade="all, delete-orphan")
    concept_stats = relationship("ConceptStat", back_populates="category", cascade="all, delete-orphan")


# ---------------------------------------------------------------------------
# Document
# ---------------------------------------------------------------------------
class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, autoincrement=True)
    category_id = Column(Integer, ForeignKey("categories.id"), nullable=False)
    filename = Column(String, nullable=False)
    file_path = Column(String, nullable=False)

    category = relationship("Category", back_populates="documents")
    chunks = relationship("Chunk", back_populates="document", cascade="all, delete-orphan")


# ---------------------------------------------------------------------------
# Chunk
# ---------------------------------------------------------------------------
class Chunk(Base):
    __tablename__ = "chunks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)
    page_number = Column(Integer, nullable=False)
    text = Column(Text, nullable=False)

    document = relationship("Document", back_populates="chunks")
    questions = relationship("Question", back_populates="source_chunk")


# ---------------------------------------------------------------------------
# Question
# ---------------------------------------------------------------------------
class Question(Base):
    __tablename__ = "questions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    source_chunk_id = Column(Integer, ForeignKey("chunks.id"), nullable=False)
    concept = Column(String, nullable=False)
    difficulty = Column(Integer, nullable=False)  # 1-5 scale
    text = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    times_asked = Column(Integer, default=0)
    times_correct = Column(Integer, default=0)

    source_chunk = relationship("Chunk", back_populates="questions")
    set_items = relationship("QuestionSetItem", back_populates="question", cascade="all, delete-orphan")
    attempts = relationship("Attempt", back_populates="question", cascade="all, delete-orphan")


# ---------------------------------------------------------------------------
# QuestionSet
# ---------------------------------------------------------------------------
class QuestionSet(Base):
    __tablename__ = "question_sets"

    id = Column(Integer, primary_key=True, autoincrement=True)
    category_id = Column(Integer, ForeignKey("categories.id"), nullable=False)
    created_at = Column(String, nullable=False)  # ISO timestamp string

    category = relationship("Category")
    items = relationship("QuestionSetItem", back_populates="question_set", cascade="all, delete-orphan")


# ---------------------------------------------------------------------------
# QuestionSetItem
# ---------------------------------------------------------------------------
class QuestionSetItem(Base):
    __tablename__ = "question_set_items"

    id = Column(Integer, primary_key=True, autoincrement=True)
    question_set_id = Column(Integer, ForeignKey("question_sets.id"), nullable=False)
    question_id = Column(Integer, ForeignKey("questions.id"), nullable=False)

    question_set = relationship("QuestionSet", back_populates="items")
    question = relationship("Question", back_populates="set_items")


# ---------------------------------------------------------------------------
# Attempt
# ---------------------------------------------------------------------------
class Attempt(Base):
    __tablename__ = "attempts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    question_id = Column(Integer, ForeignKey("questions.id"), nullable=False)
    is_correct = Column(Integer, nullable=False)  # 0 or 1 (SQLite compatible)
    mistake_type = Column(String)  # null when correct
    explanation = Column(Text)
    created_at = Column(String, nullable=False)  # ISO timestamp string

    question = relationship("Question", back_populates="attempts")


# ---------------------------------------------------------------------------
# ConceptStat
# ---------------------------------------------------------------------------
class ConceptStat(Base):
    __tablename__ = "concept_stats"

    id = Column(Integer, primary_key=True, autoincrement=True)
    category_id = Column(Integer, ForeignKey("categories.id"), nullable=False)
    concept = Column(String, nullable=False)
    correct_count = Column(Integer, default=0)
    wrong_count = Column(Integer, default=0)

    category = relationship("Category", back_populates="concept_stats")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def init_db(database_url: str = DATABASE_URL):
    """Create all tables and return a configured Session factory."""
    engine = create_engine(database_url, echo=False)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)
