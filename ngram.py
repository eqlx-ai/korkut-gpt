"""Episode 1: a letter-by-letter model with no neural network, only counting.

The model reads the book and counts which letter follows each run of n-1 letters. To write, it picks
the next letter by those counts, again and again. Deli Dumrul's story is held out: the model never
sees it, so it tells us how well the model guesses text it has not memorised.

    uv run python ngram.py                  # every order 1..8: score on Deli Dumrul and a sample
    uv run python ngram.py --n 5 --len 400  # one order, a longer sample
"""

import argparse
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

DATA = Path(__file__).parent / "data"
HELD_OUT = "05"  # Deli Dumrul


def load() -> tuple[str, str]:
    boys = sorted((DATA / "boys").glob("*.txt"))
    train = "\n".join(b.read_text() for b in boys if b.stem != HELD_OUT)
    held_out = (DATA / "boys" / f"{HELD_OUT}.txt").read_text()
    return train, held_out


class NGram:
    """Counts of the next letter after every context of up to n-1 letters."""

    def __init__(self, text: str, n: int):
        self.n = n
        self.alphabet = sorted(set(text))
        self.counts: dict[str, Counter] = defaultdict(Counter)
        for i in range(len(text)):
            for k in range(n):  # contexts of length 0..n-1 ending before letter i
                if i - k < 0:
                    break
                self.counts[text[i - k : i]][text[i]] += 1

    def context(self, before: str) -> str:
        """The longest context the book has seen; fall back to shorter ones."""
        ctx = before[len(before) - self.n + 1 :] if self.n > 1 else ""
        while ctx and ctx not in self.counts:
            ctx = ctx[1:]
        return ctx

    def prob(self, before: str, letter: str, alpha: float = 0.1) -> float:
        c = self.counts[self.context(before)]
        return (c[letter] + alpha) / (sum(c.values()) + alpha * len(self.alphabet))

    def sample(self, length: int, rng: random.Random, prompt: str = "") -> str:
        out = prompt
        for _ in range(length):
            c = self.counts[self.context(out)]
            out += rng.choices(list(c), weights=list(c.values()))[0]
        return out[len(prompt) :]

    def surprise(self, text: str) -> float:
        """Average bits per letter on text: how surprised the model is (lower = better guesses)."""
        bits = sum(
            -math.log2(self.prob(text[max(0, i - self.n) : i], text[i])) for i in range(len(text))
        )
        return bits / len(text)


def longest_copy(sample: str, book: str) -> str:
    """The longest piece of the sample that appears word for word in the book."""
    best = ""
    for i in range(len(sample)):
        j = i + len(best) + 1
        while j <= len(sample) and sample[i:j] in book:
            best = sample[i:j]
            j += 1
    return best


def copied_share(sample: str, book: str, size: int = 20) -> float:
    """Share of the sample's 20-letter pieces that are found in the book."""
    pieces = [sample[i : i + size] for i in range(len(sample) - size + 1)]
    return sum(p in book for p in pieces) / len(pieces) if pieces else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, help="one order only (letters of context + 1)")
    ap.add_argument("--len", type=int, default=240, help="letters to write")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--prompt", default="hânum hey\n")
    ap.add_argument("--seeds", type=int, help="average the copy measures over this many samples")
    args = ap.parse_args()

    train, held_out = load()
    if args.seeds:
        print(
            f"copy measures, mean over {args.seeds} samples of {args.len} letters (seeds 1..{args.seeds})"
        )
        for n in [args.n] if args.n else range(1, 9):
            model = NGram(train, n)
            texts = [
                model.sample(args.len, random.Random(s), args.prompt)
                for s in range(1, args.seeds + 1)
            ]
            share = sum(copied_share(t, train) for t in texts) / len(texts)
            longest = [len(longest_copy(t, train)) for t in texts]
            print(
                f"n={n}: 20-letter pieces in the book {share:.0%}, longest copy "
                f"mean {sum(longest) / len(longest):.0f} letters (min {min(longest)}, max {max(longest)})"
            )
        return
    print(
        f"book: {len(train)} letters to learn from, Deli Dumrul held out: {len(held_out)} letters"
    )
    for n in [args.n] if args.n else range(1, 9):
        model = NGram(train, n)
        text = model.sample(args.len, random.Random(args.seed), args.prompt)
        copy = longest_copy(text, train)
        print(f"\n=== n={n} (looks at the last {n - 1} letters)")
        print(f"surprise on Deli Dumrul: {model.surprise(held_out):.2f} bits per letter")
        print(f"longest piece copied from the book: {len(copy)} letters: {copy!r}")
        print(f"20-letter pieces found in the book: {copied_share(text, train):.0%}")
        print("---")
        print(args.prompt + text)


if __name__ == "__main__":
    main()
