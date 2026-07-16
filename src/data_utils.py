"""
Shared prompt formatting and tokenization utilities.
Kept in one place so train.py, evaluate.py, and inference.py never drift
out of sync on prompt format (a common source of silent bugs).
"""

SYSTEM_PROMPT = (
    "You are a SQL assistant. Given a table schema and a question, "
    "reply with ONLY the SQL query, nothing else."
)


def build_prompt(context: str, question: str) -> str:
    """Builds the inference-time prompt (no answer, ready for generation)."""
    return (
        "### Instruction:\n"
        f"{SYSTEM_PROMPT}\n\n"
        f"### Schema:\n{context}\n\n"
        f"### Question:\n{question}\n\n"
        "### SQL:\n"
    )


def build_training_example(context: str, question: str, answer: str, eos_token: str) -> str:
    """Builds the full training string, including the gold answer + eos token."""
    return build_prompt(context, question) + answer + eos_token


def make_format_fn(tokenizer):
    """Returns a `.map()`-compatible function that formats one dataset row."""

    def format_example(row):
        text = build_training_example(
            row["context"], row["question"], row["answer"], tokenizer.eos_token
        )
        return {"text": text}

    return format_example


def make_tokenize_fn(tokenizer, max_length: int = 512):
    """Returns a `.map()`-compatible function that tokenizes + masks labels
    so loss is only computed on the SQL answer, not the prompt or padding."""

    def tokenize(example):
        full_text = example["text"]
        tokenized = tokenizer(
            full_text, truncation=True, max_length=max_length, padding="max_length"
        )

        prompt_only = full_text.split("### SQL:")[0] + "### SQL:\n"
        prompt_len = len(tokenizer(prompt_only, truncation=True)["input_ids"])

        input_ids = tokenized["input_ids"]
        labels = input_ids.copy()

        # mask the prompt (instruction + schema + question)
        labels[:prompt_len] = [-100] * prompt_len

        # find the first eos token after the prompt -- that's the *real*
        # end-of-answer marker. Everything after it is padding (since
        # pad_token == eos_token here) and should also be masked out,
        # so the model isn't trained to endlessly predict eos as filler.
        real_eos_pos = None
        for idx in range(prompt_len, len(input_ids)):
            if input_ids[idx] == tokenizer.eos_token_id:
                real_eos_pos = idx
                break

        if real_eos_pos is not None:
            for idx in range(real_eos_pos + 1, len(labels)):
                labels[idx] = -100

        tokenized["labels"] = labels
        return tokenized

    return tokenize
