"""Episode 1: from text to numbers, and the first neural network.

tokenizer (letters -> numbers) -> batches of context windows -> embedding (input layer) ->
linear (output layer) -> logits -> cross-entropy loss -> training loop -> writing.
The model only looks at the last letter (a bigram); episode 2 lets the letters look at each other.
Deli Dumrul's story is held out as the validation set.

    uv run python bigram.py
"""

import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

DATA = Path(__file__).parent / "data"
HELD_OUT = "05"  # Deli Dumrul

batch_size = 32  # context windows trained on at once
block_size = 8  # context window: how many letters one example sees
n_embd = 32  # length of the vector each letter becomes
max_iters = 3000
eval_interval = 300
eval_iters = 200
learning_rate = 1e-2
torch.manual_seed(1337)
torch.set_num_threads(1)  # tiny model: one thread is faster than many

# --- data and tokenizer ---------------------------------------------------------------------
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


# --- model ----------------------------------------------------------------------------------
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)  # input layer: letter -> vector
        self.lm_head = nn.Linear(n_embd, vocab_size)  # output layer: vector -> a score per letter

    def forward(self, idx, targets=None):
        logits = self.lm_head(self.token_embedding(idx))  # (B, T, vocab_size)
        if targets is None:
            return logits, None
        B, T, C = logits.shape
        loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int):
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            probs = F.softmax(logits[:, -1, :], dim=-1)  # scores of the next letter -> chances
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


def sample(model, n: int = 200) -> str:
    start = torch.tensor([encode("hânum hey\n")], dtype=torch.long)
    return decode(model.generate(start, n)[0].tolist())


def main() -> None:
    print(f"letters: train {len(train_text)}, Deli Dumrul (val) {len(val_text)}")
    print(f"vocabulary: {vocab_size} tokens: {''.join(chars)!r}")
    print(f"encode('hânum hey') = {encode('hânum hey')}")
    print(f"decode(...) = {decode(encode('hânum hey'))!r}")

    torch.manual_seed(42)
    x, y = get_batch("train")
    print(f"\none batch: x {tuple(x.shape)}, y {tuple(y.shape)}")
    print(f"x[0] = {x[0].tolist()}  {decode(x[0].tolist())!r}")
    print(f"y[0] = {y[0].tolist()}  {decode(y[0].tolist())!r}")
    for t in range(block_size):
        print(
            f"  context {decode(x[0, : t + 1].tolist())!r:12} -> target {decode([y[0, t].item()])!r}"
        )

    torch.manual_seed(1337)
    model = BigramLanguageModel()
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"\nmodel: {vocab_size} x {n_embd} embedding + {n_embd} x {vocab_size} linear = {n_params} parameters"
    )
    _, loss = model(x, y)
    print(
        f"loss before training: {loss.item():.4f}  (pure guessing: -ln(1/{vocab_size}) = {math.log(vocab_size):.4f})"
    )
    print("--- writing before training ---")
    print(sample(model))

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
    print(sample(model, 300))


if __name__ == "__main__":
    main()
