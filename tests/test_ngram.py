import random

from ngram import NGram, copied_share, longest_copy

BOOK = "hânum hey\nbir gün kazan beg yerinden turmış idi\nhânum hey\n"


def test_probabilities_sum_to_one():
    model = NGram(BOOK, 3)
    for before in ["", "hâ", "xyz", "kaz"]:
        assert abs(sum(model.prob(before, c) for c in model.alphabet) - 1) < 1e-9


def test_sampling_is_repeatable_and_uses_book_letters():
    model = NGram(BOOK, 4)
    a = model.sample(80, random.Random(1))
    assert a == model.sample(80, random.Random(1))
    assert set(a) <= set(BOOK)


def test_unseen_context_falls_back_to_shorter():
    model = NGram(BOOK, 5)
    assert model.context("qqqq") == ""
    assert model.context("xxazan") == "azan"


def test_long_context_copies_the_book():
    # With context longer than any repeat, the only continuation is the book itself.
    model = NGram(BOOK, 12)
    text = model.sample(30, random.Random(0), prompt="bir gün kaz")
    assert text == BOOK[len("hânum hey\nbir gün kaz") :][:30]


def test_longest_copy_and_share():
    assert longest_copy("zz kazan beg zz", BOOK) == " kazan beg "
    assert copied_share("bir gün kazan beg yerinden", BOOK) == 1.0
    assert copied_share("qqqqqqqqqqqqqqqqqqqqqq", BOOK) == 0.0
