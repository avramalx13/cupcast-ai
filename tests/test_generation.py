import torch

from cupcast.mini_llm.config import ModelConfig
from cupcast.mini_llm.generate import generate_text
from cupcast.mini_llm.model import GPTLanguageModel
from cupcast.mini_llm.tokenizer import WordTokenizer


def test_generation_returns_non_empty_text() -> None:
    tokenizer = WordTokenizer.train_from_texts(["France Brazil 42%"], vocab_size=50)
    config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        context_length=16,
        n_layers=1,
        d_model=16,
        n_heads=4,
        d_ff=32,
        dropout=0.0,
    )
    model = GPTLanguageModel(config)

    text = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt="France",
        device=torch.device("cpu"),
        max_new_tokens=2,
        temperature=1.0,
        top_k=5,
    )

    assert text.strip()
