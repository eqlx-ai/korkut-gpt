# korkut-gpt

A tiny language model trained from scratch, letter by letter, on the Book of Dede Korkut
(*Kitâb-ı Dedem Korkut*, Dresden manuscript). Made for the eqlx video series.

```bash
uv run python prepare.py   # download and clean the text -> data/korkut.txt
uv run python bigram.py     # episode 1: the bigram network
uv run python attention.py  # episode 2: one head of self-attention
uv run pytest
```

## Data and licence

- Text: Turkish Wikisource transcription of the Dresden manuscript (Mscr.Dresd.Ea.86),
  <https://tr.wikisource.org/wiki/Kitâb-ı_Dedem_Korkut_Alâ_Lisan-ı_Tâife-i_Oğuzân>, CC BY-SA 4.0.
  Page revisions used are listed in `data/SOURCE.json`. Our changes: wiki markup removed,
  Turkish lower case, stray digits removed. `data/` is shared under CC BY-SA 4.0.
- Code: MIT (`LICENSE`).
