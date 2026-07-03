import torch

from cupcast.mini_llm.config import ModelConfig
from cupcast.mini_llm.model import GPTLanguageModel


def test_model_forward_returns_expected_logits_shape() -> None:
    config = ModelConfig(
        vocab_size=64,
        context_length=16,
        n_layers=2,
        d_model=32,
        n_heads=4,
        d_ff=64,
        dropout=0.0,
    )
    model = GPTLanguageModel(config)
    inputs = torch.randint(0, config.vocab_size, (3, 12))

    logits, loss = model(inputs)

    assert logits.shape == (3, 12, config.vocab_size)
    assert loss is None
