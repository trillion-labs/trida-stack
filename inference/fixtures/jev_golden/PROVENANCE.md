# Jev golden token rows: provenance

`golden.json` holds 24 fixed requests and the token rows (or the `ContextOverflow` message) that the training stack's
own Jev encoder produced for them. `inference/test_jev_golden.py` requires `inference/jev.py` to reproduce them exactly.

- **Encoder:** `train/jev.py` in Trillion Labs' internal training repository, at commit
  `951c21c70ca1e82688ff58a8ad8565b68762d3c2`: `encode_rows(tok, materialize(request), **kwargs)`, with each case's
  `kwargs` stored next to it. `inference/jev.py` is that file's encoding section copied unchanged.
- **Tokenizer:** `trillionlabs/Trida2.1-4B` at Hugging Face revision `a64a0737a79921991a769a99b06872b883f455d3`.
- **Generator:** `generate.py` in this folder. It imports the reference encoder from a checkout at that commit, never
  `inference/jev.py`, so the rows are an independent reference.

Cases:

- Text in Korean and with emoji, nested-object and list states, `<|name|>` text in every field (escaped, never a
  control token), 255 options, a soft target; every question type (`noul`, `choice`, `score`). Each at the default
  limits and at `training_context(7552)`, the public demo's state limit. Without the wider limit the 255-option branch
  overflows, which is recorded as such. The six demo requests' `ids` / `decide` / `opts` equal the public demo's own
  fixtures.
- State cut at the default 384-token state: 383 text tokens (fits) and 384 (cut by one).
- `image_tokens > 0`: a small image; 64 image tokens with the text at the limit and one over; 380 and 381 image
  tokens, which leave one and zero text tokens; 256 image tokens at the demo's limits.
- Branch budget: a branch that fills the row exactly, and one token more (`ContextOverflow`).

Whether the public Trida2.1-4B checkpoint was trained with the encoder at exactly this commit is not established
here: the fixtures prove agreement with that commit's encoder.
