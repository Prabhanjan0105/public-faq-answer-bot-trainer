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




def milestone_2(
    model_name="sshleifer/tiny-gpt2",
    epochs=1,
    learning_rate=0.0005,
    rank=4,
    alpha=8,
    max_length=128,
    output_dir="models/faq_lora"
):
    """
    Fine-tune GPT-2 using custom LoRA adapters
    on the public service FAQ dataset.
    """
    import json
    from pathlib import Path

    import torch
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM
    )

    from app.lora import (
        inject_lora,
        count_parameters,
        save_lora_adapter
    )

    if epochs < 1:
        raise ValueError("epochs must be positive")

    if learning_rate <= 0:
        raise ValueError(
            "learning_rate must be positive"
        )

    if max_length < 8:
        raise ValueError(
            "max_length must be at least 8"
        )

    torch.manual_seed(42)

    # Milestone 1 dataset
    dataset = milestone_1()
    faqs = dataset["faqs"]

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name
    )

    tokenizer.pad_token = tokenizer.eos_token

    print("Loading pretrained model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name
    )

    # Inject custom LoRA adapters
    replaced = inject_lora(
        model,
        rank=rank,
        alpha=alpha
    )

    stats = count_parameters(model)

    print("LoRA layers injected:", replaced)
    print("Total parameters:", stats["total"])
    print("Trainable parameters:", stats["trainable"])
    print(
        "Trainable percentage:",
        round(stats["percentage"], 4)
    )

    optimizer = torch.optim.AdamW(
        (
            p for p in model.parameters()
            if p.requires_grad
        ),
        lr=learning_rate
    )

    loss_history = []

    model.train()

    for epoch in range(epochs):
        epoch_loss = 0.0
        trained_examples = 0

        for faq in faqs:
            prompt = (
                "### Question:\n"
                f"{faq['question']}\n\n"
                "### Answer:\n"
            )

            answer = (
                faq["answer"] +
                tokenizer.eos_token
            )

            # Tokenize prompt and answer separately
            prompt_ids = tokenizer.encode(
                prompt, add_special_tokens=False
            )

            answer_ids = tokenizer.encode(
                answer, add_special_tokens=False
            )

            # Keep space for at least one answer token
            prompt_ids = prompt_ids[
                :max_length - 2
            ]

            remaining = (
                max_length - len(prompt_ids)
            )

            answer_ids = answer_ids[:remaining]

            input_ids = prompt_ids + answer_ids

            # Compute loss on answer tokens only
            labels = (
                [-100] * len(prompt_ids)
                + answer_ids
            )

            inputs = torch.tensor(
                [input_ids],
                dtype=torch.long
            )

            targets = torch.tensor(
                [labels],
                dtype=torch.long
            )

            optimizer.zero_grad()

            outputs = model(
                input_ids=inputs,
                labels=targets
            )

            loss = outputs.loss

            if not torch.isfinite(loss):
                raise RuntimeError(
                    "Training produced non-finite loss"
                )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                (
                    p for p in model.parameters()
                    if p.requires_grad
                ),
                max_norm=1.0
            )

            optimizer.step()

            epoch_loss += loss.item()
            trained_examples += 1

        average_loss = (
            epoch_loss / trained_examples
        )

        loss_history.append(average_loss)

        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"| Loss: {average_loss:.4f}"
        )

    model.eval()

    # Save adapter and metadata
    output_path = Path(output_dir)
    output_path.mkdir(
        parents=True,
        exist_ok=True
    )

    adapter_path = (
        output_path / "adapter.safetensors"
    )

    save_lora_adapter(
        model, adapter_path
    )

    config = {
        "model_name": model_name,
        "rank": rank,
        "alpha": alpha,
        "max_length": max_length,
        "epochs": epochs
    }

    config_path = output_path / "config.json"

    config_path.write_text(
        json.dumps(config, indent=2),
        encoding="utf-8"
    )

    print("LoRA adapter saved:", adapter_path)

    return {
        "model": model,
        "tokenizer": tokenizer,
        "loss_history": loss_history,
        "adapter_path": str(adapter_path),
        "config_path": str(config_path),
        "injected_layers": replaced,
        "parameter_stats": stats,
        "total_faqs": len(faqs)
    }




def milestone_3(
    question,
    model,
    tokenizer,
    max_new_tokens=80,
    temperature=0.7
):
    """
    Connect the Streamlit application
    to the LoRA FAQ inference pipeline.
    """
    from app.inference import generate_faq_answer

    answer = generate_faq_answer(
        question=question,
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        temperature=temperature
    )

    return {
        "question": question,
        "answer": answer,
        "model_type": "GPT-2 with LoRA"
    }




def milestone_4(
    answer_generator=None,
    output_path="data/faq_evaluation.csv",
    max_questions=None
):
    """
    Evaluate generated answers against reference FAQs.

    answer_generator: callable(question) -> answer
    """
    from app.evaluation import (
        calculate_metrics,
        save_evaluation_csv
    )

    dataset = milestone_1()
    faqs = dataset["faqs"]

    if max_questions is not None:
        if not isinstance(max_questions, int) or max_questions < 1:
            raise ValueError(
                "max_questions must be a positive integer"
            )
        faqs = faqs[:max_questions]

    # Load actual model only when no generator is provided.
    if answer_generator is None:
        from app.inference import (
            load_faq_model,
            generate_faq_answer
        )

        model, tokenizer, _ = load_faq_model()

        def answer_generator(question):
            return generate_faq_answer(
                question=question,
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=80,
                temperature=0.7
            )

    results = []

    for faq in faqs:
        question = faq["question"]
        reference = faq["answer"]

        generated = answer_generator(question)

        if not isinstance(generated, str):
            raise TypeError(
                "Answer generator must return a string"
            )

        metrics = calculate_metrics(
            reference,
            generated
        )

        results.append({
            "id": faq["id"],
            "category": faq["category"],
            "question": question,
            "reference_answer": reference,
            "generated_answer": generated,
            **metrics,
            "accuracy_rating": "",
            "relevance_rating": "",
            "review_notes": ""
        })

    average_f1 = (
        sum(row["f1"] for row in results) / len(results)
        if results else 0.0
    )

    saved_path = save_evaluation_csv(
        results,
        output_path
    )

    return {
        "results": results,
        "total_evaluated": len(results),
        "average_f1": round(average_f1, 4),
        "output_path": saved_path
    }


# --- Milestone 5: Implement adapter merging for inference.
# Hint: Apply the adapter merging technique for efficient and standalone inference of your fine-tuned model.
def milestone_5():  # TODO(milestone 5)
    raise NotImplementedError("TODO(milestone 5)")
