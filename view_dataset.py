
from app.core import milestone_1

data = milestone_1()

print("=" * 55)
print("PUBLIC FAQ ANSWER BOT - DATASET")
print("=" * 55)

print("Total FAQs:", data["total_faqs"])
print("Categories:", ", ".join(data["categories"]))

for faq in data["faqs"]:
    print("\n" + "-" * 55)
    print("ID:", faq["id"])
    print("Category:", faq["category"])
    print("Question:", faq["question"])
    print("Answer:", faq["answer"])

print("\n" + "=" * 55)
print("TRAINING EXAMPLE")
print("=" * 55)

print(data["training_examples"][0]["text"])
