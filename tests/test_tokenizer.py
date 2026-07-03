from cupcast.mini_llm.tokenizer import WordTokenizer


def test_word_tokenizer_roundtrip_keeps_digits() -> None:
    text = "France Elo: 2042\nDraw probability: 24%"
    tokenizer = WordTokenizer.train_from_texts([text], vocab_size=100)

    token_ids = tokenizer.encode(text, add_bos=True, add_eos=True)
    decoded = tokenizer.decode(token_ids)

    assert decoded == text
    assert tokenizer.unknown_tokens(text) == []
