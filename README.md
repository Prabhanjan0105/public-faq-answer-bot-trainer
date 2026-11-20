# Public FAQ Answer Bot Trainer

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/7k997nxnr6/badge.svg)](https://ask.thili.ai/b/7k997nxnr6)

This project, 'Public FAQ Answer Bot Trainer', enables public service administrators to train a language model using LoRA. The goal is to provide accurate answers to frequently asked questions based on specific departmental guidelines, thereby reducing inquiry load. It adapts a pre-trained LLM to generate correct FAQ responses, demonstrating the effectiveness of LoRA fine-tuning in a public services domain.

## What you are building

**Goal.** You will have built a LoRA fine-tuned language model capable of generating accurate answers to public service FAQs, integrated into a Streamlit interface for easy interaction and training by administrators.

**Done when.** The project is complete when you have successfully trained a LoRA adapter on a pre-trained LLM, merged it for inference, and built a Streamlit interface that provides accurate and relevant answers to public service FAQs, verified against your dataset.

## Milestones

Tick these off as you go, and commit after each one.

- [ ] **1. Create a small dataset of public service FAQs and their answers.**  
      Hint: For creating the dataset, focus on clear question-answer pairs from public service guidelines.
- [ ] **2. Implement LoRA adapter injection and training on a pre-trained LLM (e.g., GPT-2).**  
      Hint: Use the LoRA adapter injection and training technique from the `Build LoRA from Scratch` drill.
- [ ] **3. Build a Streamlit interface where users can ask questions and get answers.**  
      Hint: Leverage Streamlit for building an interactive web application where users can input questions.
- [ ] **4. Verify the accuracy and relevance of the generated answers.**  
      Hint: Manually review the generated answers against your dataset to verify their accuracy and relevance.
- [ ] **5. Implement adapter merging for inference.**  
      Hint: Apply the adapter merging technique for efficient and standalone inference of your fine-tuned model.

## Run it

You need Python 3.12 and git.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then choose a model; see the comments inside
streamlit run app/main.py
pytest
```

Your work goes in `app/core.py` (one stub per milestone) and `tests/test_core.py`. The first tests pass already; they
check the plumbing.

## Settings and secrets

Model settings live in `.env`, which is ignored by git. **Never commit `.env` or paste a key into code.** If a key ever
reaches GitHub, treat it as leaked and make a new one.

## Stretch ideas

- Expand the public service FAQ dataset with a wider range of questions and answers to improve the model's generalization.
- Enhance the Streamlit interface with additional features like a history of asked questions or a feedback mechanism for administrators.
- Implement more detailed quantitative evaluation metrics to assess the accuracy and relevance of generated answers beyond manual verification.
