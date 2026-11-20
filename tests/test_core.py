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



# Milestone 3: Build a Streamlit interface where users can ask questions and get answers.
@pytest.mark.skip(reason="remove this line when you start milestone 3")
def test_milestone_3():
    raise NotImplementedError("write the test for milestone 3")


# Milestone 4: Verify the accuracy and relevance of the generated answers.
@pytest.mark.skip(reason="remove this line when you start milestone 4")
def test_milestone_4():
    raise NotImplementedError("write the test for milestone 4")


# Milestone 5: Implement adapter merging for inference.
@pytest.mark.skip(reason="remove this line when you start milestone 5")
def test_milestone_5():
    raise NotImplementedError("write the test for milestone 5")

