
"""Streamlit interface for Public FAQ Answer Bot."""

import streamlit as st

from app.core import milestone_1, milestone_3
from app.inference import load_faq_model


st.set_page_config(
    page_title="Public FAQ Answer Bot",
    page_icon="📚",
    layout="wide"
)


@st.cache_resource
def get_model():
    """Load the LoRA model once per Streamlit session."""
    return load_faq_model("models/faq_lora")


def main():
    st.title("📚 Public FAQ Answer Bot Trainer")

    st.write(
        "Ask questions about public services and "
        "departmental procedures using our "
        "LoRA fine-tuned language model."
    )

    st.info(
        "Educational demonstration only. "
        "Generated answers may be inaccurate. "
        "Verify requirements with official sources."
    )

    st.sidebar.title("⚙️ Model Settings")

    max_tokens = st.sidebar.slider(
        "Maximum generated tokens",
        min_value=10,
        max_value=150,
        value=80,
        step=10
    )

    temperature = st.sidebar.slider(
        "Sampling temperature",
        min_value=0.1,
        max_value=1.5,
        value=0.7,
        step=0.1
    )

    st.sidebar.divider()

    st.sidebar.subheader("Model Information")
    st.sidebar.write("Architecture: GPT-2")
    st.sidebar.write("Fine-tuning: LoRA")
    st.sidebar.write("Framework: PyTorch")

    try:
        model, tokenizer, config = get_model()

        st.sidebar.success("LoRA model loaded")
        st.sidebar.caption(
            f"Base model: {config['model_name']}"
        )

    except (FileNotFoundError, OSError, ValueError, KeyError) as error:
        st.error(f"Unable to load the model: {error}")
        st.code("python train_lora.py")
        st.stop()

    tab1, tab2 = st.tabs([
        "💬 Ask a Question",
        "📋 FAQ Dataset"
    ])

    with tab1:
        st.subheader("Ask the FAQ Answer Bot")

        question = st.text_area(
            "Enter your public service question",
            placeholder=(
                "Example: How can I check "
                "my application status?"
            ),
            height=120
        )

        if st.button(
            "Generate Answer",
            type="primary"
        ):
            if not question.strip():
                st.warning(
                    "Please enter a question first."
                )
            else:
                with st.spinner(
                    "Generating answer..."
                ):
                    try:
                        result = milestone_3(
                            question=question,
                            model=model,
                            tokenizer=tokenizer,
                            max_new_tokens=max_tokens,
                            temperature=temperature
                        )

                        st.session_state["last_result"] = result

                    except (ValueError, RuntimeError) as error:
                        st.error(
                            f"Generation failed: {error}"
                        )

        if "last_result" in st.session_state:
            result = st.session_state["last_result"]

            st.divider()
            st.subheader("🤖 Generated Answer")
            st.write(result["answer"])

            st.caption(
                "AI-generated response. "
                "Not verified against official guidelines."
            )

    with tab2:
        st.subheader("Public Service FAQ Dataset")

        dataset = milestone_1()

        st.metric(
            "Total FAQ Examples",
            dataset["total_faqs"]
        )

        category = st.selectbox(
            "Filter by category",
            ["All"] + dataset["categories"]
        )

        for faq in dataset["faqs"]:
            if (
                category != "All"
                and faq["category"] != category
            ):
                continue

            with st.expander(faq["question"]):
                st.write(faq["answer"])
                st.caption(
                    f"Category: {faq['category']}"
                )

    st.divider()

    st.caption(
        "Public FAQ Answer Bot Trainer | "
        "LoRA Fine-Tuning Project"
    )


if __name__ == "__main__":
    main()
