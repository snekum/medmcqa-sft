"""Compare two eval runs on the same questions.

    python compare.py                                  # results/base.json vs results/lora.json
    python compare.py results/base.json results/x.json

Two accuracies a few points apart on 500 questions can easily be noise, so
this also runs McNemar's exact test: of the questions where exactly one model
is right, is the split further from 50/50 than chance would give?
"""

import json
import sys
from collections import Counter
from math import comb

from data import LETTERS


def main():
    a_path, b_path = sys.argv[1:3] if len(sys.argv) >= 3 else ("results/base.json", "results/lora.json")
    a = json.load(open(a_path))["predictions"]
    b = json.load(open(b_path))["predictions"]
    assert [x["id"] for x in a] == [x["id"] for x in b], "runs must use the same questions"

    a_right = [x["pred"] == x["gold"] for x in a]
    b_right = [x["pred"] == x["gold"] for x in b]
    only_a = sum(ra and not rb for ra, rb in zip(a_right, b_right))
    only_b = sum(rb and not ra for ra, rb in zip(a_right, b_right))
    n, k = only_a + only_b, min(only_a, only_b)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n) if n else 1.0

    print(f"{a_path}: {sum(a_right)}/{len(a)} = {sum(a_right) / len(a):.1%}")
    print(f"{b_path}: {sum(b_right)}/{len(b)} = {sum(b_right) / len(b):.1%}")
    print(f"only first right: {only_a}, only second right: {only_b}, McNemar exact p = {p:.3f}")
    print(f"answers changed: {sum(x['pred'] != y['pred'] for x, y in zip(a, b))}/{len(a)}")

    print("\nletter   gold  first  second")
    gold, pa, pb = (Counter(x["gold"] for x in a), Counter(x["pred"] for x in a),
                    Counter(x["pred"] for x in b))
    for letter in LETTERS:
        print(f"  {letter}    {gold[letter]:5d}  {pa[letter]:5d}  {pb[letter]:6d}")
    top = gold.most_common(1)[0]
    print(f"\nalways answering '{top[0]}' would score {top[1] / len(a):.1%}")


if __name__ == "__main__":
    main()
