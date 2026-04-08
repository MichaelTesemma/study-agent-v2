"""Main CLI entry point for the Study Agent app."""

import os
import sys

import requests

from models import init_db, Category
from ingestion import process_pdf
from session import list_categories, select_category, run_study_session
from concept_tracking import get_weak_concepts
from config import OLLAMA_BASE_URL


def check_ollama():
    """Try to reach Ollama at the configured base URL. Print a warning if unreachable."""
    try:
        response = requests.get(OLLAMA_BASE_URL, timeout=3)
        if response.status_code == 200:
            print("Ollama is reachable.")
        else:
            print(f"WARNING: Ollama responded with status {response.status_code}. "
                  f"Some features (question generation, answer evaluation) may not work.")
    except requests.RequestException:
        print(f"WARNING: Cannot reach Ollama at {OLLAMA_BASE_URL}. "
              f"Some features (question generation, answer evaluation) will not work.")


def show_menu():
    """Print the main menu and return the user's choice as a string."""
    print("\n=== Study Agent ===")
    print("1. Upload PDF")
    print("2. Start Study Session")
    print("3. View Weak Concepts")
    print("4. Exit")
    return input("Choose an option (1-4): ").strip()


def upload_pdf(session):
    """Interactively upload a PDF and process it through the ingestion pipeline."""
    print("\n--- Upload PDF ---")

    # Get PDF path
    pdf_path = input("Enter the path to the PDF file: ").strip()
    if not pdf_path:
        print("No path provided. Aborting.")
        return

    if not os.path.isfile(pdf_path):
        print(f"File not found: {pdf_path}")
        return

    # Show existing categories or allow creating a new one
    categories = list_categories(session)

    if categories:
        choice = input("Enter an existing category number, or type 'new' to create one: ").strip()
        if choice.lower() == "new":
            category_name = input("Enter the new category name: ").strip()
            if not category_name:
                print("No category name provided. Aborting.")
                return
            # Check if category already exists
            existing = session.query(Category).filter_by(name=category_name).first()
            if existing:
                print(f"Category '{category_name}' already exists (id={existing.id}). Using it.")
                category_id = existing.id
            else:
                new_cat = Category(name=category_name)
                session.add(new_cat)
                session.flush()
                category_id = new_cat.id
                print(f"Created new category '{category_name}' (id={category_id}).")
        else:
            try:
                category_id = int(choice)
            except ValueError:
                print(f"Invalid input '{choice}'. Aborting.")
                return
            cat = session.query(Category).filter_by(id=category_id).first()
            if cat is None:
                print(f"Category {category_id} does not exist. Aborting.")
                return
            category_id = cat.id
    else:
        category_name = input("No categories exist. Enter a name for the first category: ").strip()
        if not category_name:
            print("No category name provided. Aborting.")
            return
        new_cat = Category(name=category_name)
        session.add(new_cat)
        session.flush()
        category_id = new_cat.id
        print(f"Created new category '{category_name}' (id={category_id}).")

    # Get number of questions per chunk
    raw = input("Number of questions to generate per chunk (default 3): ").strip()
    try:
        num_questions = int(raw) if raw else 3
    except ValueError:
        print(f"Invalid number '{raw}'. Using default of 3.")
        num_questions = 3

    # Process the PDF
    print()
    try:
        result = process_pdf(session, pdf_path, category_id, num_questions)
        session.commit()
        print(f"\nDone! Document processed:")
        print(f"  Document ID: {result['document_id']}")
        print(f"  Chunks: {result['num_chunks']}")
        print(f"  Questions generated: {result['num_questions_generated']}")
    except Exception as e:
        session.rollback()
        print(f"\nError processing PDF: {e}")


def view_weak_concepts(session):
    """Show weak concepts for a selected category."""
    print("\n--- View Weak Concepts ---")

    categories = list_categories(session)
    if not categories:
        print("No categories found. Upload a PDF first.")
        return

    raw = input("Enter the category number: ").strip()
    try:
        category_id = int(raw)
    except ValueError:
        print(f"Invalid input '{raw}'.")
        return

    cat = session.query(Category).filter_by(id=category_id).first()
    if cat is None:
        print(f"Category {category_id} does not exist.")
        return

    weak = get_weak_concepts(session, category_id)

    if not weak:
        print(f"\nNo weak concepts found for '{cat.name}'. Great job!")
        return

    print(f"\nWeak concepts for '{cat.name}' (wrong > correct):")
    for stat in weak:
        print(f"  - {stat.concept}: {stat.correct_count} correct, {stat.wrong_count} wrong")


def main():
    """Main entry point: initialize the database and run the menu loop."""
    print("Starting Study Agent...")

    # Check Ollama availability
    check_ollama()

    # Initialize the database
    print("Initializing database...")
    Session = init_db()
    session = Session()

    try:
        while True:
            choice = show_menu()

            if choice == "1":
                upload_pdf(session)
            elif choice == "2":
                print("\n--- Start Study Session ---")
                category_id = select_category(session)
                if category_id is not None:
                    run_study_session(session, category_id)
                    session.commit()
            elif choice == "3":
                view_weak_concepts(session)
            elif choice == "4":
                print("Goodbye!")
                break
            else:
                print(f"Invalid option '{choice}'. Please choose 1-4.")
    except KeyboardInterrupt:
        print("\nInterrupted. Goodbye!")
    except EOFError:
        print("\nEnd of input. Goodbye!")
    finally:
        session.close()


if __name__ == "__main__":
    main()
