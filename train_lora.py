
from app.core import milestone_2

print("=" * 55)
print("PUBLIC FAQ ANSWER BOT - LORA TRAINING")
print("=" * 55)

result = milestone_2(
    model_name="sshleifer/tiny-gpt2",
    epochs=3,
    learning_rate=0.0005,
    rank=4,
    alpha=8,
    max_length=128,
    output_dir="models/faq_lora"
)

print("\nTRAINING SUMMARY")
print("-" * 55)

print("FAQ examples:", result["total_faqs"])
print("Injected LoRA layers:", result["injected_layers"])

stats = result["parameter_stats"]

print("Total parameters:", stats["total"])
print("Trainable parameters:", stats["trainable"])
print(
    "Trainable percentage:",
    round(stats["percentage"], 4), "%"
)

print("Loss history:", result["loss_history"])
print("Adapter saved:", result["adapter_path"])
print("=" * 55)
