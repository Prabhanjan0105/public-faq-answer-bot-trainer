
"""Inference using the LoRA fine-tuned GPT-2 model."""

import json
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)

from app.lora import inject_lora, load_lora_adapter


def load_faq_model(output_dir="models/faq_lora"):
    """Load pretrained GPT-2 and its trained LoRA adapter."""

    output_path = Path(output_dir)
    config_path = output_path / "config.json"
    adapter_path = output_path / "adapter.safetensors"

    if not config_path.exists():
        raise FileNotFoundError(
            "Model configuration not found. "
            "Run python train_lora.py first."
        )

    if not adapter_path.exists():
        raise FileNotFoundError(
            "LoRA adapter not found. "
            "Run python train_lora.py first."
        )

    config = json.loads(
        config_path.read_text(encoding="utf-8")
    )

    model_name = config["model_name"]

    tokenizer = AutoTokenizer.from_pretrained(
        model_name
    )
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name
    )

    inject_lora(
        model,
        rank=config["rank"],
        alpha=config["alpha"]
    )

    load_lora_adapter(
        model,
        adapter_path
    )

    model.eval()

    return model, tokenizer, config


def generate_faq_answer(
    question,
    model,
    tokenizer,
    max_new_tokens=80,
    temperature=0.7
):
    """Generate an answer from a student/user question."""

    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question cannot be empty")

    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")

    if temperature <= 0:
        raise ValueError("temperature must be positive")

    prompt = (
        "### Question:\n"
        f"{question.strip()}\n\n"
        "### Answer:\n"
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=256
    )

    device = next(model.parameters()).device

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id
        )

    prompt_length = inputs["input_ids"].shape[1]

    new_tokens = output[0][prompt_length:]

    answer = tokenizer.decode(
        new_tokens,
        skip_special_tokens=True
    ).strip()

    # Stop if the model begins another FAQ example
    for marker in ("### Question:", "### Answer:"):
        if marker in answer:
            answer = answer.split(marker)[0].strip()

    if not answer:
        return "The model could not generate an answer."

    return answer
