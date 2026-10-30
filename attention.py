"""Episode 2: the letters look at each other (one head of self-attention).

Episode 1's bigram saw only the last letter. Here every letter also gets its position, and one
attention head lets each letter gather what it needs from the letters before it:
average of the past -> the same average as a matrix multiplication -> the mask ->
query, key, value -> one head of self-attention, trained on the same book.
Deli Dumrul's story is still held out as the validation set.

    uv run python attention.py                # the trick, the model, training, what the head looks at
    uv run python attention.py --block-size 8 # the same model with episode 1's context window
    uv run python attention.py --no-position  # the same model without position embedding
"""

import argparse
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

DATA = Path(__file__).parent / "data"
HELD_OUT = "05"  # Deli Dumrul

batch_size = 32
block_size = 32  # episode 1: 8; the bigram did not care, attention does
n_embd = 32
head_size = 32  # length of each letter's query, key and value
max_iters = 3000
eval_interval = 300
eval_iters = 200
learning_rate = 1e-2
use_position = True
WINDOW = "örelüm nece yalvarmış eydür\nyüce"  # a window from Deli Dumrul, ends in "yüce"

# --- data and tokenizer: the same as bigram.py ------------------------------------------------
boys = sorted((DATA / "boys").glob("*.txt"))
train_text = "\n".join(b.read_text() for b in boys if b.stem != HELD_OUT)
val_text = (DATA / "boys" / f"{HELD_OUT}.txt").read_text()

chars = sorted(set(train_text + val_text))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}


def encode(s: str) -> list[int]:
    return [stoi[c] for c in s]


def decode(ids: list[int]) -> str:
    return "".join(itos[i] for i in ids)


train_data = torch.tensor(encode(train_text), dtype=torch.long)
val_data = torch.tensor(encode(val_text), dtype=torch.long)


def get_batch(split: str) -> tuple[torch.Tensor, torch.Tensor]:
    """batch_size random windows of block_size letters; y is x shifted by one letter."""
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x, y


# --- the trick: every letter takes the average of the letters before it ----------------------
def the_trick() -> None:
    torch.manual_seed(42)
    T, C = 4, 2  # four letters, two numbers each: small enough to read
    x = torch.randint(0, 10, (T, C)).float()
    print(f"x, four letters with two numbers each:\n{x}")

    # 1) with a loop: letter t averages letters 0..t
    xbow = torch.zeros(T, C)
    for t in range(T):
        xbow[t] = x[: t + 1].mean(dim=0)
    print(f"\n1) loop, average of the past:\n{xbow}")

    # 2) the same with one matrix multiplication: a lower triangle of equal weights
    wei = torch.tril(torch.ones(T, T))
    wei = wei / wei.sum(dim=1, keepdim=True)
    print(f"\n2) weights (each row sums to 1):\n{wei}")
    print(f"wei @ x:\n{wei @ x}\nsame as the loop: {torch.allclose(wei @ x, xbow)}")

    # 3) the same with a mask and softmax: zeros are 'equal interest', -inf is 'the future'
    tril = torch.tril(torch.ones(T, T))
    wei = torch.zeros(T, T).masked_fill(tril == 0, float("-inf"))
    print(f"\n3) scores with the mask:\n{wei}")
    wei = F.softmax(wei, dim=-1)
    print(f"softmax(scores):\n{wei}\nsame as the loop: {torch.allclose(wei @ x, xbow)}")


# --- model ----------------------------------------------------------------------------------
class Head(nn.Module):
    """One head of self-attention."""

    def __init__(self, head_size: int):
        super().__init__()
        self.query = nn.Linear(n_embd, head_size, bias=False)  # what this letter is looking for
        self.key = nn.Linear(n_embd, head_size, bias=False)  # what this letter offers
        self.value = nn.Linear(n_embd, head_size, bias=False)  # what it passes on if chosen
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        T = x.shape[1]  # x is (B, T, n_embd): batch, letters, numbers per letter
        q = self.query(x)  # (B, T, head_size)
        k = self.key(x)  # (B, T, head_size)
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5  # (B, T, T): how well q and k match
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # no looking at the future
        wei = F.softmax(wei, dim=-1)  # each row: shares of attention that sum to 1
        self.last_wei = wei  # kept to show what the head looked at
        v = self.value(x)  # (B, T, head_size)
        return wei @ v  # (B, T, head_size): a weighted average of the past letters' values


class AttentionLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)  # which letter
        self.position_embedding = nn.Embedding(block_size, n_embd)  # where in the window
        self.sa_head = Head(head_size)
        self.lm_head = nn.Linear(head_size, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.token_embedding(idx)  # (B, T, n_embd)
        if use_position:
            x = x + self.position_embedding(torch.arange(T))  # (T, n_embd), added to every row
        x = self.sa_head(x)  # (B, T, head_size)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        if targets is None:
            return logits, None
        B, T, C = logits.shape
        loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]  # there are only block_size positions
            logits, _ = self(idx_cond)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            idx = torch.cat((idx, torch.multinomial(probs, num_samples=1)), dim=1)
        return idx


@torch.no_grad()
def estimate_loss(model) -> dict[str, float]:
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = torch.tensor([model(*get_batch(split))[1].item() for _ in range(eval_iters)])
        out[split] = losses.mean().item()
    model.train()
    return out


def sample(model, n: int = 300) -> str:
    start = torch.tensor([encode("hânum hey\n")], dtype=torch.long)
    return decode(model.generate(start, n)[0].tolist())


@torch.no_grad()
def show_attention(model, text: str, top: int = 5) -> None:
    """Where the last letter of `text` looks: the head's attention row, largest shares first."""
    model(torch.tensor([encode(text)], dtype=torch.long))
    row = model.sa_head.last_wei[0, -1]
    print(f"context {text!r}: the last letter {text[-1]!r} looks at")
    for t in row.argsort(descending=True)[:top].tolist():
        print(f"  position {t:2d} {text[t]!r}  {row[t].item():.0%}")


@torch.no_grad()
def show_matrix(model, text: str, last: int = 6) -> None:
    """The head's attention table for the last `last` letters of `text`: row looks at column."""
    text = text[-block_size:]  # the model has only block_size positions
    model(torch.tensor([encode(text)], dtype=torch.long))
    wei = model.sa_head.last_wei[0, -last:, -last:]
    letters = [repr(c)[1:-1] for c in text[-last:]]
    print("      " + "".join(f"{c:>5}" for c in letters))
    for c, row in zip(letters, wei.tolist()):
        print(f"{c:>5} " + "".join(f"{w:5.0%}" if w > 0 else "    ." for w in row))


@torch.no_grad()
def show_next(model, words: list[str], top: int = 3) -> None:
    """The model's most likely next letters after each word."""
    for w in words:
        logits, _ = model(torch.tensor([encode(w)], dtype=torch.long))
        probs = F.softmax(logits[0, -1], dim=-1)
        best = probs.argsort(descending=True)[:top].tolist()
        print(f"  after {w!r}: " + ", ".join(f"{itos[i]!r} {probs[i].item():.0%}" for i in best))


@torch.no_grad()
def attention_by_distance(model, n_batches: int = 50) -> list[float]:
    """Average share of attention the last letter of a full window gives to 0, 1, 2... letters back."""
    model.eval()
    shares = torch.zeros(block_size)
    for _ in range(n_batches):
        x, _ = get_batch("val")
        model(x)
        shares += (
            model.sa_head.last_wei[:, -1, :].mean(dim=0).flip(0)
        )  # index 0 = the letter itself
    model.train()
    return (shares / n_batches).tolist()


def main() -> None:
    global block_size, use_position
    parser = argparse.ArgumentParser()
    parser.add_argument("--block-size", type=int, default=block_size)
    parser.add_argument("--no-position", action="store_true")
    parser.add_argument("--skip-trick", action="store_true")
    args = parser.parse_args()
    block_size, use_position = args.block_size, not args.no_position

    if not args.skip_trick:
        print("=== the trick ===")
        the_trick()

    torch.manual_seed(1337)
    model = AttentionLanguageModel()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n=== the model: block_size {block_size}, position embedding {use_position} ===")
    for name, p in model.named_parameters():
        print(f"  {name:28} {tuple(p.shape)}")
    print(f"parameters: {n_params}")

    print("\n--- scores before the softmax: why we divide by the square root of head_size ---")
    q, k = (
        torch.randn(batch_size, block_size, head_size),
        torch.randn(batch_size, block_size, head_size),
    )
    raw = q @ k.transpose(-2, -1)
    print(
        f"variance of q @ k: {raw.var().item():.1f}, divided by sqrt({head_size}): {(raw * head_size**-0.5).var().item():.2f}"
    )
    print(f"softmax of [1, 2, 3]:     {F.softmax(torch.tensor([1.0, 2, 3]), -1).tolist()}")
    print(f"softmax of [1, 2, 3] * 8: {F.softmax(torch.tensor([1.0, 2, 3]) * 8, -1).tolist()}")

    print("\n--- the head before training, last 6 letters of a Deli Dumrul window ---")
    show_matrix(model, WINDOW)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    losses = {}
    print()
    for it in range(max_iters + 1):
        if it % eval_interval == 0:
            losses = estimate_loss(model)
            print(
                f"step {it:4d}: train loss {losses['train']:.4f}, Deli Dumrul loss {losses['val']:.4f}"
            )
        if it == max_iters:
            break
        xb, yb = get_batch("train")
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    print(f"(Deli Dumrul loss in bits per letter: {losses['val'] / math.log(2):.2f})")
    print("--- writing after training ---")
    print(sample(model))

    print("\n--- what the head looks at, on Deli Dumrul ---")
    shares = attention_by_distance(model)
    for d, share in enumerate(shares[:8]):
        print(f"  {d} letters back: {share:.0%}")
    print(f"  8 or more back: {sum(shares[8:]):.0%}")
    torch.manual_seed(7)
    for _ in range(6):
        x, _ = get_batch("val")
        show_attention(model, decode(x[0].tolist()))

    print("\n--- the head after training, the same 6 letters ---")
    show_matrix(model, WINDOW)
    print("\n--- same letters, other order: what comes next? ---")
    show_next(model, ["kalem", "kelam"])


if __name__ == "__main__":
    torch.manual_seed(1337)
    torch.set_num_threads(1)
    main()
