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
| Always answer "A" (most common answer) | 30.8% |
| Qwen2.5-0.5B-Instruct (base) | 32.4% |
| + LoRA SFT, 1,000 examples, 2 epochs | 34.6% |

**The +2.2 points is not significant.** SFT changed the answer on 267 of 500 questions, gaining 83 and losing 72 (McNemar exact test, p = 0.42).

**SFT mostly removed a letter bias rather than adding medical knowledge.** The base model picks "A" 61% of the time. After SFT, its picks are spread across the letters much like the real answers:

| | A | B | C | D |
|---|---|---|---|---|
| Correct answers | 154 | 134 | 120 | 92 |
| Base model picks | 304 | 84 | 72 | 40 |
| After SFT | 141 | 122 | 118 | 119 |

**The training loss tells the same story.** Only the answer letter and `<|im_end|>` are trained, and `<|im_end|>` becomes trivial to predict. Random guessing over 4 letters therefore gives a mean loss of about ln(4)/2 ≈ 0.69.
- **Epoch 1** (all-new questions): mean loss 0.694, exactly the guessing level.
- **Epoch 2** (repeat questions): mean loss dropped to 0.418. The drop starts on the first repeated batch, which suggests memorisation rather than learning.

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
python compare.py                       # significance test + letter distribution
```

Useful knobs: `--n-train` (500–2000), `--epochs`, `--lr`, `--lora-r`.

## Files

| File | What it does |
|---|---|
| [data.py](data.py) | Loads MedMCQA and formats the prompt (shared by train and eval) |
| [train.py](train.py) | LoRA SFT training loop |
| [eval.py](eval.py) | Before/after accuracy |
| [compare.py](compare.py) | Paired significance test and letter distribution for two eval runs |
| [results/](results/) | Per-question predictions from each eval run |

## Notes / what I learned

_To be filled in as I go._

## Next steps

- Train for 1 epoch only, to test whether the second epoch helps or just memorises
- Vary training-set size (500 / 1,000 / 2,000) and plot accuracy against it
- Train on the explanations (`exp` field) as well as the letter
- Compare LoRA rank 4 / 16 / 64
- Full fine-tuning compared with LoRA
