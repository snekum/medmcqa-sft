"""Measure MedMCQA accuracy of the base model, or base + a LoRA adapter.

    python eval.py                               # baseline
    python eval.py --adapter outputs/lora        # after SFT

Scoring: run the prompt once, look at the model's next-token logits, and pick
whichever of "A" / "B" / "C" / "D" scores highest. This tests what the model
knows without penalising it for chatty formatting.
"""

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from data import LETTERS, MODEL_NAME, answer_letter, load_examples, prompt_text


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", default=None, help="path to a saved LoRA adapter")
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--out", default=None, help="where to write results json")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    letter_ids = [tokenizer(l, add_special_tokens=False)["input_ids"][0] for l in LETTERS]
    # Fixed seed, separate split from training: same 500 questions every run.
    examples = load_examples("validation", args.n, seed=1234)

    correct, preds = 0, []
    for i in range(0, len(examples), args.batch_size):
        chunk = examples.select(range(i, min(i + args.batch_size, len(examples))))
        enc = tokenizer([prompt_text(tokenizer, ex) for ex in chunk], return_tensors="pt",
                        padding=True, add_special_tokens=False).to(device)
        logits = model(**enc).logits
        # With right padding, the last real token of row b is at length-1.
        last = enc["attention_mask"].sum(dim=1) - 1
        next_token = logits[torch.arange(len(chunk)), last]
        choice = next_token[:, letter_ids].argmax(dim=-1).tolist()

        for ex, c in zip(chunk, choice):
            pred, gold = LETTERS[c], answer_letter(ex)
            correct += pred == gold
            preds.append({"id": ex["id"], "subject": ex["subject_name"], "pred": pred, "gold": gold})

    acc = correct / len(examples)
    label = args.adapter or "base"
    print(f"{label}: {correct}/{len(examples)} = {acc:.1%}  (random guessing = 25%)")

    out = Path(args.out or f"results/{'lora' if args.adapter else 'base'}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"model": MODEL_NAME, "adapter": args.adapter, "n": len(examples),
                               "accuracy": acc, "predictions": preds}, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
