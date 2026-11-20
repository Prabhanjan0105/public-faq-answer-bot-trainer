"""Your tests. Each milestone in README.md gets one here: write the test first, then make it pass.

They start skipped. Remove the `skip` line as you work on each milestone.
"""
import pytest

# Milestone 1: Create public service FAQ dataset
def test_milestone_1():
    from app.core import milestone_1

    result = milestone_1()

    # Verify returned structure
    assert isinstance(result, dict)

    assert "faqs" in result
    assert "training_examples" in result
    assert "total_faqs" in result
    assert "categories" in result

    # Verify dataset size
    assert result["total_faqs"] == 12
    assert len(result["faqs"]) == 12

    # Verify each FAQ
    for faq in result["faqs"]:
        assert "id" in faq
        assert "category" in faq
        assert "question" in faq
        assert "answer" in faq

        assert faq["question"].strip()
        assert faq["answer"].strip()

    # Verify no duplicate questions
    questions = [
        faq["question"].casefold()
        for faq in result["faqs"]
    ]

    assert len(questions) == len(set(questions))

    # Verify training examples
    examples = result["training_examples"]

    assert len(examples) == 12

    for example in examples:
        assert "### Question:" in example["text"]
        assert "### Answer:" in example["text"]

    # Verify expected categories
    assert "General Services" in result["categories"]
    assert "Documentation" in result["categories"]
    assert "Payments" in result["categories"]
    assert "Support" in result["categories"]




# Milestone 2: LoRA injection and fine-tuning
def test_milestone_2(tmp_path):
    import torch
    from app.core import milestone_2
    from app.lora import (
        LoRAConv1D,
        count_parameters,
        load_lora_adapter
    )
    from transformers import AutoModelForCausalLM

    result = milestone_2(
        model_name="sshleifer/tiny-gpt2",
        epochs=1,
        rank=4,
        alpha=8,
        max_length=96,
        output_dir=str(tmp_path / "faq_lora")
    )

    assert result["total_faqs"] == 12
    assert result["injected_layers"] > 0

    stats = result["parameter_stats"]

    assert stats["trainable"] > 0
    assert stats["trainable"] < stats["total"]

    assert len(result["loss_history"]) == 1
    assert result["loss_history"][0] > 0

    assert (
        tmp_path / "faq_lora" / "adapter.safetensors"
    ).exists()

    assert (
        tmp_path / "faq_lora" / "config.json"
    ).exists()

    model = result["model"]

    assert isinstance(
        model.transformer.h[0].attn.c_attn,
        LoRAConv1D
    )

    # Verify only LoRA parameters are trainable
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            assert (
                "lora_A" in name
                or "lora_B" in name
            )

    # Verify the saved adapter can be reloaded
    from app.lora import inject_lora

    fresh_model = (
        AutoModelForCausalLM.from_pretrained(
            "sshleifer/tiny-gpt2"
        )
    )

    inject_lora(
        fresh_model,
        rank=4,
        alpha=8
    )

    load_lora_adapter(
        fresh_model,
        result["adapter_path"]
    )

    fresh_stats = count_parameters(fresh_model)
    assert fresh_stats["trainable"] > 0




# Milestone 3: Streamlit FAQ inference interface
def test_milestone_3(monkeypatch):
    import pytest

    from app import inference
    from app.core import milestone_3

    # Use a mock to avoid downloading GPT-2
    def fake_generate_faq_answer(
        question,
        model,
        tokenizer,
        max_new_tokens=80,
        temperature=0.7
    ):
        if not question.strip():
            raise ValueError("Question cannot be empty")

        return (
            "Use the official department portal "
            "to check your application status."
        )

    monkeypatch.setattr(
        inference,
        "generate_faq_answer",
        fake_generate_faq_answer
    )

    result = milestone_3(
        question="How can I check my application status?",
        model=object(),
        tokenizer=object()
    )

    assert isinstance(result, dict)

    assert "question" in result
    assert "answer" in result
    assert "model_type" in result

    assert result["question"] == (
        "How can I check my application status?"
    )

    assert "official department portal" in result["answer"]
    assert result["model_type"] == "GPT-2 with LoRA"

    with pytest.raises(ValueError):
        milestone_3(
            question="",
            model=object(),
            tokenizer=object()
        )




# Milestone 4: Verify answer accuracy and relevance
def test_milestone_4(tmp_path):
    import csv
    import pytest

    from app.core import milestone_1, milestone_4
    from app.evaluation import calculate_metrics

    dataset = milestone_1()

    reference_lookup = {
        faq["question"]: faq["answer"]
        for faq in dataset["faqs"]
    }

    # Perfect-answer generator for metric verification
    def perfect_generator(question):
        return reference_lookup[question]

    output_file = tmp_path / "evaluation.csv"

    result = milestone_4(
        answer_generator=perfect_generator,
        output_path=str(output_file)
    )

    assert result["total_evaluated"] == 12
    assert result["average_f1"] == 1.0

    assert output_file.exists()

    for row in result["results"]:
        assert row["precision"] == 1.0
        assert row["recall"] == 1.0
        assert row["f1"] == 1.0

        # Manual ratings start empty
        assert row["accuracy_rating"] == ""
        assert row["relevance_rating"] == ""

    # Verify saved CSV
    with output_file.open(
        newline="", encoding="utf-8"
    ) as file:
        rows = list(csv.DictReader(file))

    assert len(rows) == 12
    assert "generated_answer" in rows[0]
    assert "review_notes" in rows[0]

    # Completely unrelated answer
    metrics = calculate_metrics(
        "application status",
        "banana orange"
    )

    assert metrics["f1"] == 0.0

    # Empty prediction
    metrics = calculate_metrics(
        "application status",
        ""
    )

    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0

    # Invalid question limit
    with pytest.raises(ValueError):
        milestone_4(
            answer_generator=perfect_generator,
            max_questions=0
        )




# Milestone 5: Merge LoRA adapter for inference
def test_milestone_5(tmp_path, monkeypatch):
    import torch

    from transformers import (
        GPT2Config,
        GPT2LMHeadModel
    )

    from app import inference
    from app.core import milestone_5
    from app.lora import (
        LoRAConv1D,
        inject_lora
    )

    torch.manual_seed(42)

    # Create a small GPT-2 model locally.
    # No Hugging Face download is required.
    config = GPT2Config(
        vocab_size=100,
        n_positions=32,
        n_ctx=32,
        n_embd=32,
        n_layer=2,
        n_head=4
    )

    model = GPT2LMHeadModel(config)

    injected = inject_lora(
        model,
        rank=4,
        alpha=8
    )

    assert injected == 4

    # Simulate trained, nonzero LoRA weights
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, LoRAConv1D):
                module.lora_B.normal_(
                    mean=0.0,
                    std=0.01
                )

    class FakeTokenizer:
        eos_token_id = 0

        def __call__(self, text, return_tensors="pt"):
            return {
                "input_ids": torch.tensor(
                    [[1, 2, 3, 4]],
                    dtype=torch.long
                )
            }

        def save_pretrained(self, path):
            from pathlib import Path

            Path(path, "tokenizer_test.txt").write_text(
                "test tokenizer",
                encoding="utf-8"
            )

    tokenizer = FakeTokenizer()

    # Substitute a locally constructed model
    # for the saved LoRA model.
    def fake_load_model(output_dir):
        return model, tokenizer, {
            "model_name": "local-test-gpt2"
        }

    monkeypatch.setattr(
        inference,
        "load_faq_model",
        fake_load_model
    )

    output_dir = tmp_path / "merged"

    result = milestone_5(
        adapter_dir=str(tmp_path / "adapter"),
        output_dir=str(output_dir)
    )

    assert result["merged_layers"] == 4
    assert result["remaining_lora_layers"] == 0
    assert result["max_logit_difference"] < 1e-4

    # Confirm the merged model can be reloaded
    restored = GPT2LMHeadModel.from_pretrained(
        str(output_dir)
    )

    restored.eval()

    sample = torch.tensor([[1, 2, 3, 4]])

    with torch.no_grad():
        expected = result["model"](sample).logits
        actual = restored(sample).logits

    assert torch.allclose(
        expected,
        actual,
        atol=1e-5,
        rtol=1e-5
    )


