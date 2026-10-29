import math

import torch

import bigram


def test_encode_decode_round_trip():
    text = "hânum hey\nkazan beg"
    assert bigram.decode(bigram.encode(text)) == text


def test_vocab_is_the_books_letters():
    assert bigram.vocab_size == 42
    assert "ñ" in bigram.stoi and "I" not in bigram.stoi


def test_batch_targets_are_inputs_shifted_by_one():
    torch.manual_seed(0)
    x, y = bigram.get_batch("train")
    assert x.shape == y.shape == (bigram.batch_size, bigram.block_size)
    assert torch.equal(x[:, 1:], y[:, :-1])


def test_untrained_loss_is_near_pure_guessing():
    torch.manual_seed(0)
    model = bigram.BigramLanguageModel()
    _, loss = model(*bigram.get_batch("train"))
    assert abs(loss.item() - math.log(bigram.vocab_size)) < 0.5


def test_generate_adds_letters():
    model = bigram.BigramLanguageModel()
    out = model.generate(torch.zeros((1, 1), dtype=torch.long), 10)
    assert out.shape == (1, 11)
