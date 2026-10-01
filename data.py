"""MedMCQA loading + prompt formatting, shared by train.py and eval.py.

Keeping the prompt format in one place matters: if training and eval format
questions differently, eval numbers are meaningless.
"""

from datasets import load_dataset

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DATASET_NAME = "openlifescienceai/medmcqa"
LETTERS = ["A", "B", "C", "D"]


def load_examples(split, n, seed=0):
    """Return n shuffled examples from a MedMCQA split ('train' or 'validation').

    The official test split has no answers (cop == -1), so we evaluate on
    'validation' instead.
    """
    ds = load_dataset(DATASET_NAME, split=split)
    return ds.shuffle(seed=seed).select(range(min(n, len(ds))))


def build_messages(ex):
    """Turn one MedMCQA row into a chat conversation (without the answer)."""
    question = (
        f"{ex['question']}\n\n"
        f"A. {ex['opa']}\n"
        f"B. {ex['opb']}\n"
        f"C. {ex['opc']}\n"
        f"D. {ex['opd']}\n\n"
        "Answer with a single letter."
    )
    return [{"role": "user", "content": question}]


def prompt_text(tokenizer, ex):
    """The prompt as the model sees it, ending right where the answer should start."""
    return tokenizer.apply_chat_template(
        build_messages(ex), tokenize=False, add_generation_prompt=True
    )


def answer_letter(ex):
    return LETTERS[ex["cop"]]  # cop = index of the correct option (0-3)
