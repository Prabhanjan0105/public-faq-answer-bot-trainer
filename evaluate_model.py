
from app.core import milestone_4

print("=" * 55)
print("PUBLIC FAQ ANSWER BOT - MODEL EVALUATION")
print("=" * 55)

result = milestone_4(
    output_path="data/faq_evaluation.csv"
)

print("\nEVALUATION SUMMARY")
print("-" * 55)

print("Questions evaluated:", result["total_evaluated"])
print("Average word-overlap F1:", result["average_f1"])

for row in result["results"]:
    print("\nQuestion:", row["question"])
    print("Reference:", row["reference_answer"])
    print("Generated:", row["generated_answer"])
    print("F1:", row["f1"])

print("\nEvaluation CSV:", result["output_path"])
print("Manual review is required.")
