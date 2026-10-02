"""Supervised fine-tuning (SFT) of Qwen2.5-0.5B-Instruct on MedMCQA with LoRA.

SFT in one sentence: show the model (prompt, answer) pairs and train it with
ordinary next-token cross-entropy -- but only on the answer tokens.

This is a deliberately plain PyTorch loop (no Trainer / SFTTrainer) so every
step is visible.

    python train.py                     # 1000 examples, 2 epochs
    python train.py --n-train 2000      # more data
"""

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

from data import MODEL_NAME, answer_letter, load_examples, prompt_text


def tokenize_example(tokenizer, ex):
    prompt_ids = tokenizer(prompt_text(tokenizer, ex), add_special_tokens=False)["input_ids"]
    # Target = the letter, then <|im_end|> so the model also learns to stop.
    answer_ids = tokenizer(answer_letter(ex), add_special_tokens=False)["input_ids"]
    answer_ids = answer_ids + [tokenizer.eos_token_id]

    input_ids = prompt_ids + answer_ids
    # -100 means "ignore this position in the loss". The model reads the
    # question but is only graded on predicting the answer.
    labels = [-100] * len(prompt_ids) + answer_ids
    return input_ids, labels


def collate(batch, pad_id):
    """Right-pad a list of (input_ids, labels) to the same length."""
    max_len = max(len(ids) for ids, _ in batch)
    input_ids, labels, attention_mask = [], [], []
    for ids, labs in batch:
        pad = max_len - len(ids)
        input_ids.append(ids + [pad_id] * pad)
        labels.append(labs + [-100] * pad)
        attention_mask.append([1] * len(ids) + [0] * pad)
    return {
        "input_ids": torch.tensor(input_ids),
        "labels": torch.tensor(labels),
        "attention_mask": torch.tensor(attention_mask),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-train", type=int, default=1000)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=2)  # small: fits a 4 GB GPU
    p.add_argument("--grad-accum", type=int, default=8)  # effective batch = 2 x 8 = 16
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--max-len", type=int, default=384)
    p.add_argument("--out", default="outputs/lora")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"
    if use_amp:
        # On Windows the NVIDIA driver silently spills GPU memory into system
        # RAM when the card is full. Capping PyTorch makes it reuse its cache
        # instead (or fail with a clear OOM error) rather than eat your RAM.
        torch.cuda.set_per_process_memory_fraction(0.85)

    # --- Model ---------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # fp16 base weights (~1 GB) so it fits on a small GPU. The base stays frozen;
    # only the LoRA adapters (kept in fp32 by peft) get gradients.
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16 if use_amp else torch.float32, device_map=device
    )
    model.config.use_cache = False

    # LoRA: instead of updating a weight matrix W (d x d), learn a low-rank
    # update B @ A (d x r, r x d) with r << d, and use W + (alpha/r) * B @ A.
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=2 * args.lora_r,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # --- Data ----------------------------------------------------------------
    raw = load_examples("train", args.n_train, seed=args.seed)
    examples = [tokenize_example(tokenizer, ex) for ex in raw]
    kept = [e for e in examples if len(e[0]) <= args.max_len]
    print(f"{len(kept)} training examples ({len(examples) - len(kept)} dropped as > {args.max_len} tokens)")

    # --- Optimizer -----------------------------------------------------------
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)
    micro_per_epoch = math.ceil(len(kept) / args.batch_size)
    total_steps = math.ceil(micro_per_epoch / args.grad_accum) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.05 * total_steps), total_steps)
    # fp16 gradients can underflow to zero; GradScaler multiplies the loss up
    # before backward and divides the gradients back down before the step.
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # --- Train ---------------------------------------------------------------
    model.train()
    log, step, running, t0 = [], 0, 0.0, time.time()
    for epoch in range(args.epochs):
        random.shuffle(kept)
        for i in range(micro_per_epoch):
            batch = collate(kept[i * args.batch_size:(i + 1) * args.batch_size], tokenizer.pad_token_id)
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                # Passing labels makes the model compute shifted cross-entropy
                # over every position whose label isn't -100.
                loss = model(**batch).loss / args.grad_accum
            scaler.scale(loss).backward()
            running += loss.item()

            if (i + 1) % args.grad_accum == 0 or i == micro_per_epoch - 1:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                step += 1

                log.append({"step": step, "epoch": epoch, "loss": running, "lr": scheduler.get_last_lr()[0]})
                if step % 10 == 0 or step == total_steps:
                    print(f"step {step}/{total_steps}  epoch {epoch}  loss {running:.4f}  "
                          f"lr {scheduler.get_last_lr()[0]:.2e}  {time.time() - t0:.0f}s")
                running = 0.0

    # --- Save ----------------------------------------------------------------
    # Only the adapter weights are saved (a few MB), not the 0.5B base model.
    out = Path(args.out)
    model.save_pretrained(out)
    (out / "train_log.json").write_text(json.dumps({"args": vars(args), "log": log}, indent=2))
    print(f"Saved LoRA adapter to {out}")


if __name__ == "__main__":
    main()
