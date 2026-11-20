
from app.core import milestone_5

print("=" * 55)
print("PUBLIC FAQ ANSWER BOT - FINAL MODEL MERGING")
print("=" * 55)

result = milestone_5(
    adapter_dir="models/faq_lora",
    output_dir="models/faq_merged"
)

print("\nMERGING SUMMARY")
print("-" * 55)

print("Base model:", result["base_model"])
print("Merged layers:", result["merged_layers"])
print(
    "Remaining LoRA layers:",
    result["remaining_lora_layers"]
)
print(
    "Maximum logit difference:",
    result["max_logit_difference"]
)
print("Saved directory:", result["output_dir"])

print("\nStandalone model is ready!")
