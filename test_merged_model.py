
import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)

MODEL_PATH = "models/faq_merged"

print("=" * 55)
print("TESTING STANDALONE MERGED FAQ MODEL")
print("=" * 55)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH
)

model.eval()

question = "How can I check my application status?"

prompt = (
    "### Question:\n"
    f"{question}\n\n"
    "### Answer:\n"
)

inputs = tokenizer(
    prompt,
    return_tensors="pt"
)

with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=60,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

new_tokens = output[0][
    inputs["input_ids"].shape[1]:
]

answer = tokenizer.decode(
    new_tokens,
    skip_special_tokens=True
).strip()

print("\nQUESTION:")
print(question)

print("\nGENERATED ANSWER:")
print(answer or "(No answer generated)")

print("\nStandalone inference successful!")

