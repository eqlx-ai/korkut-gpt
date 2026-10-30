import math

import torch

import attention


def test_trick_three_ways_agree(capsys):
    attention.the_trick()
    out = capsys.readouterr().out
    assert out.count("same as the loop: True") == 2


def test_rows_of_attention_sum_to_one():
    torch.manual_seed(0)
    model = attention.AttentionLanguageModel()
    x, _ = attention.get_batch("train")
    model(x)
    wei = model.sa_head.last_wei
    assert wei.shape == (attention.batch_size, attention.block_size, attention.block_size)
    assert torch.allclose(wei.sum(dim=-1), torch.ones(wei.shape[:2]))


def test_mask_hides_the_future():
    """Changing the last letter must not change any earlier letter's scores."""
    torch.manual_seed(0)
    model = attention.AttentionLanguageModel()
    x, _ = attention.get_batch("train")
    changed = x.clone()
    changed[:, -1] = (changed[:, -1] + 1) % attention.vocab_size
    before, _ = model(x)
    after, _ = model(changed)
    assert torch.allclose(before[:, :-1], after[:, :-1])
    assert not torch.allclose(before[:, -1], after[:, -1])


def test_position_matters():
    """With position embedding the same letters in another order give other scores."""
    torch.manual_seed(0)
    model = attention.AttentionLanguageModel()
    a = torch.tensor([attention.encode("kal")])
    b = torch.tensor([attention.encode("lak")])
    assert not torch.allclose(model(a)[0][0, -1], model(b)[0][0, -1])


def test_untrained_loss_is_near_pure_guessing():
    torch.manual_seed(0)
    model = attention.AttentionLanguageModel()
    _, loss = model(*attention.get_batch("train"))
    assert abs(loss.item() - math.log(attention.vocab_size)) < 0.5


def test_generate_goes_past_the_window():
    model = attention.AttentionLanguageModel()
    out = model.generate(torch.zeros((1, 1), dtype=torch.long), attention.block_size + 5)
    assert out.shape == (1, attention.block_size + 6)


def test_show_matrix_fits_a_short_window(capsys, monkeypatch):
    monkeypatch.setattr(attention, "block_size", 8)
    model = attention.AttentionLanguageModel()
    attention.show_matrix(model, attention.WINDOW)
    assert len(capsys.readouterr().out.splitlines()) == 7


def test_loss_by_position_has_one_value_per_position():
    model = attention.AttentionLanguageModel()
    bits = attention.loss_by_position(model, n_batches=2)
    assert len(bits) == attention.block_size
