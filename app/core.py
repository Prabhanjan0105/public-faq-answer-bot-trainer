"""Your code. One stub per milestone; build them in order and keep each one small."""


def milestone_1():
    """
    Load and validate the public service FAQ dataset.
    Prepare question-answer pairs for LoRA training.
    """
    import json
    from pathlib import Path

    dataset_path = (
        Path(__file__).resolve().parent
        / "faq_dataset.json"
    )

    with dataset_path.open(
        "r", encoding="utf-8"
    ) as file:
        faqs = json.load(file)

    if not isinstance(faqs, list) or not faqs:
        raise ValueError("FAQ dataset must be a nonempty list")

    cleaned_faqs = []
    training_examples = []
    seen_questions = set()

    for faq in faqs:
        required_fields = (
            "id", "category", "question", "answer"
        )

        if not isinstance(faq, dict):
            raise ValueError("Each FAQ must be an object")

        if any(field not in faq for field in required_fields):
            raise ValueError("FAQ is missing required fields")

        category = faq["category"].strip()
        question = faq["question"].strip()
        answer = faq["answer"].strip()

        if not category or not question or not answer:
            raise ValueError("FAQ fields cannot be empty")

        normalized_question = question.casefold()

        if normalized_question in seen_questions:
            raise ValueError("Duplicate FAQ question found")

        seen_questions.add(normalized_question)

        cleaned_faq = {
            "id": faq["id"],
            "category": category,
            "question": question,
            "answer": answer
        }

        cleaned_faqs.append(cleaned_faq)

        # Format examples for language-model fine-tuning
        training_text = (
            "### Question:\n"
            f"{question}\n\n"
            "### Answer:\n"
            f"{answer}"
        )

        training_examples.append({
            "text": training_text
        })

    return {
        "faqs": cleaned_faqs,
        "training_examples": training_examples,
        "total_faqs": len(cleaned_faqs),
        "categories": sorted({
            faq["category"]
            for faq in cleaned_faqs
        })
    }



# --- Milestone 2: Implement LoRA adapter injection and training on a pre-trained LLM (e.g., GPT-2).
# Hint: Use the LoRA adapter injection and training technique from the `Build LoRA from Scratch` drill.
def milestone_2():  # TODO(milestone 2)
    raise NotImplementedError("TODO(milestone 2)")


# --- Milestone 3: Build a Streamlit interface where users can ask questions and get answers.
# Hint: Leverage Streamlit for building an interactive web application where users can input questions.
def milestone_3():  # TODO(milestone 3)
    raise NotImplementedError("TODO(milestone 3)")


# --- Milestone 4: Verify the accuracy and relevance of the generated answers.
# Hint: Manually review the generated answers against your dataset to verify their accuracy and relevance.
def milestone_4():  # TODO(milestone 4)
    raise NotImplementedError("TODO(milestone 4)")


# --- Milestone 5: Implement adapter merging for inference.
# Hint: Apply the adapter merging technique for efficient and standalone inference of your fine-tuned model.
def milestone_5():  # TODO(milestone 5)
    raise NotImplementedError("TODO(milestone 5)")
