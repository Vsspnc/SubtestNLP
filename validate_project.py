"""Check the learning corpus and evaluation file before deployment."""

from __future__ import annotations

import csv
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DOCUMENTS_DIR = PROJECT_ROOT / "data" / "documents"
EXPECTED_DOCUMENTS = {
    "network_basics.txt",
    "osi_model.txt",
    "tcp_ip.txt",
    "ip_address.txt",
    "routing.txt",
    "vlan.txt",
    "cybersecurity.txt",
}
EXPECTED_COLUMNS = {
    "Group",
    "Question",
    "Expected Answer",
    "Retrieved Source",
    "Result",
    "Pass/Fail",
}


def validate_project() -> None:
    """Fail fast if required documents or grouped evaluation rows are missing."""
    actual_documents = {path.name for path in DOCUMENTS_DIR.glob("*.txt")}
    missing_documents = EXPECTED_DOCUMENTS - actual_documents
    if missing_documents:
        raise SystemExit(f"Missing documents: {', '.join(sorted(missing_documents))}")

    character_count = sum(
        len(path.read_text(encoding="utf-8"))
        for path in DOCUMENTS_DIR.glob("*.txt")
    )
    if character_count < 15_000:
        raise SystemExit(f"The corpus has {character_count:,} characters; at least 15,000 are required.")

    with (PROJECT_ROOT / "test_questions.csv").open(encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        missing_columns = EXPECTED_COLUMNS - set(reader.fieldnames or [])
        rows = list(reader)
    if missing_columns:
        raise SystemExit(f"Missing CSV columns: {', '.join(sorted(missing_columns))}")
    if len(rows) != 15:
        raise SystemExit(f"Expected 15 evaluation questions, found {len(rows)}.")

    expected_groups = {"A": 5, "B": 3, "C": 2, "D": 2, "E": 3}
    actual_groups = {group: sum(row["Group"] == group for row in rows) for group in expected_groups}
    if actual_groups != expected_groups:
        raise SystemExit(f"Unexpected question groups: {actual_groups}")

    print(f"OK: {len(actual_documents)} documents, {character_count:,} characters, {len(rows)} grouped questions.")


if __name__ == "__main__":
    validate_project()