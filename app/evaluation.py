
"""Evaluation utilities for the Public FAQ Answer Bot."""

import csv
import re
from collections import Counter
from pathlib import Path


def tokenize_words(text):
    """Extract lowercase words for comparison."""
    return re.findall(r"\b\w+\b", text.lower())


def calculate_metrics(reference, prediction):
    """
    Calculate token-overlap precision, recall and F1.

    These are similarity metrics, not factual accuracy.
    """
    reference_tokens = tokenize_words(reference)
    predicted_tokens = tokenize_words(prediction)

    reference_counts = Counter(reference_tokens)
    predicted_counts = Counter(predicted_tokens)

    overlap = sum(
        (reference_counts & predicted_counts).values()
    )

    precision = (
        overlap / len(predicted_tokens)
        if predicted_tokens else 0.0
    )

    recall = (
        overlap / len(reference_tokens)
        if reference_tokens else 0.0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4)
    }


def save_evaluation_csv(results, output_path):
    """Save generated answers and review scores to CSV."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "id",
        "category",
        "question",
        "reference_answer",
        "generated_answer",
        "precision",
        "recall",
        "f1",
        "accuracy_rating",
        "relevance_rating",
        "review_notes"
    ]

    with path.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fields,
            extrasaction="ignore"
        )

        writer.writeheader()
        writer.writerows(results)

    return str(path)
