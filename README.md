# Fine-tuning Qwen2.5-0.5B on MedMCQA with LoRA

A minimal, from-scratch-ish supervised fine-tuning (SFT) experiment. I'm using it to learn how SFT works end to end.

- **Model:** [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
- **Data:** 1,000 questions from [MedMCQA](https://huggingface.co/datasets/openlifescienceai/medmcqa), a dataset of 4-option medical entrance-exam questions
- **Method:** LoRA adapters trained with a plain PyTorch loop (no `Trainer` / `SFTTrainer`, so every step is visible)
- **Hardware:** a single 4 GB GTX 1650

## Results

Accuracy on 500 held-out MedMCQA validation questions (random guessing = 25%).

| Model | Accuracy |
|---|---|
| Qwen2.5-0.5B-Instruct (base) | _TBD_ |
| + LoRA SFT, 1,000 examples | _TBD_ |

## How it works

**SFT** means showing the model (prompt, answer) pairs and training it with the usual next-token cross-entropy loss, computed **only on the answer tokens**. Each example becomes:

```
<|im_start|>user
Which nerve supplies ...?

A. ...
B. ...
C. ...
D. ...

Answer with a single letter.<|im_end|>
<|im_start|>assistant
B<|im_end|>
```

Every prompt token gets label `-100`, so the loss ignores it. Only `B` and `<|im_end|>` are trained ([train.py](train.py), `tokenize_example`).

**LoRA** freezes the 0.5B base weights. For each attention and MLP projection `W`, it learns a small low-rank update `B @ A` (rank 16) and uses `W + (α/r)·B@A`. About 1.8% of the parameters are trainable, and the saved adapter is only a few MB.

**Evaluation** ([eval.py](eval.py)) runs each question through the model once. It then takes the next-token logits for `A` / `B` / `C` / `D` and counts the highest-scoring letter as the prediction. This measures what the model knows, without penalising the base model for answering in full sentences.

## Run it

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows; use `source .venv/bin/activate` on Linux/macOS
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt

python eval.py                          # baseline  -> results/base.json
python train.py                         # SFT       -> outputs/lora/
python eval.py --adapter outputs/lora   # after SFT -> results/lora.json
```

Useful knobs: `--n-train` (500–2000), `--epochs`, `--lr`, `--lora-r`.

## Files

| File | What it does |
|---|---|
| [data.py](data.py) | Loads MedMCQA and formats the prompt (shared by train and eval) |
| [train.py](train.py) | LoRA SFT training loop |
| [eval.py](eval.py) | Before/after accuracy |

## Notes / what I learned

_To be filled in as I go._

## Next steps

- Vary training-set size (500 / 1,000 / 2,000) and plot accuracy against it
- Train on the explanations (`exp` field) as well as the letter
- Compare LoRA rank 4 / 16 / 64
- Full fine-tuning compared with LoRA
