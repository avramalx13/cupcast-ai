import torch

from cupcast.mini_llm.config import ModelConfig
from cupcast.mini_llm.model import CausalSelfAttention


def test_causal_mask_prevents_future_attention() -> None:
    config = ModelConfig(
        vocab_size=64,
        context_length=8,
        n_layers=1,
        d_model=16,
        n_heads=4,
        d_ff=32,
        dropout=0.0,
    )
    attention = CausalSelfAttention(config)
    attention.eval()
    x = torch.randn(2, 5, config.d_model)

    _, weights = attention(x, return_attention=True)
    future_mask = torch.triu(torch.ones(5, 5, dtype=torch.bool), diagonal=1)

    assert weights.shape == (2, config.n_heads, 5, 5)
    assert torch.allclose(weights[:, :, future_mask], torch.zeros_like(weights[:, :, future_mask]))
