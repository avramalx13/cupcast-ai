from cupcast.mini_llm.dataset import TokenBlockDataset


def test_token_block_dataset_returns_shifted_inputs_and_targets() -> None:
    dataset = TokenBlockDataset(list(range(10)), block_size=4)

    inputs, targets = dataset[0]

    assert inputs.tolist() == [0, 1, 2, 3]
    assert targets.tolist() == [1, 2, 3, 4]
