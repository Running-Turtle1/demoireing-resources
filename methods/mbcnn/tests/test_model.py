import pytest
import torch

from mbcnn import MBCNN, MBCNNLoss, load_model_checkpoint


def test_forward_shapes_and_ranges() -> None:
    model = MBCNN().eval()
    inputs = torch.rand(1, 3, 16, 24)

    with torch.inference_mode():
        output_quarter, output_half, output_full = model(inputs)

    assert output_quarter.shape == (1, 3, 4, 6)
    assert output_half.shape == (1, 3, 8, 12)
    assert output_full.shape == inputs.shape
    for output in (output_quarter, output_half, output_full):
        assert torch.all((0 <= output) & (output <= 1))


def test_backward_with_reference_loss() -> None:
    model = MBCNN()
    criterion = MBCNNLoss()
    inputs = torch.rand(1, 3, 16, 16)
    target = torch.rand_like(inputs)

    outputs = model(inputs)
    loss = criterion(*outputs, target)
    loss.backward()

    assert torch.isfinite(loss)
    assert model.conv_func1.conv.weight.grad is not None


def test_legacy_constructor_and_lightning_checkpoint() -> None:
    source = MBCNN(nFilters=64)
    lightning_state = {
        f"model.{key}": value for key, value in source.state_dict().items()
    }
    lightning_state["loss_fn.unrelated"] = torch.tensor(1.0)
    target = MBCNN()

    result = load_model_checkpoint(target, {"state_dict": lightning_state})

    assert result.missing_keys == []
    assert result.unexpected_keys == []
    for key, value in source.state_dict().items():
        torch.testing.assert_close(target.state_dict()[key], value)


def test_loads_training_checkpoint_model_key() -> None:
    source = MBCNN()
    target = MBCNN()

    result = load_model_checkpoint(target, {"model": source.state_dict(), "epoch": 3})

    assert result.missing_keys == []
    assert result.unexpected_keys == []
    for key, value in source.state_dict().items():
        torch.testing.assert_close(target.state_dict()[key], value)


def test_loads_safetensors_checkpoint(tmp_path) -> None:
    safetensors = pytest.importorskip("safetensors.torch")
    source = MBCNN()
    checkpoint_path = tmp_path / "model.safetensors"
    safetensors.save_file(source.state_dict(), checkpoint_path)
    target = MBCNN()

    result = load_model_checkpoint(target, checkpoint_path)

    assert result.missing_keys == []
    assert result.unexpected_keys == []
    for key, value in source.state_dict().items():
        torch.testing.assert_close(target.state_dict()[key], value)


@pytest.mark.parametrize(
    "shape",
    [
        (3, 16, 16),
        (1, 1, 16, 16),
        (1, 3, 15, 16),
    ],
)
def test_invalid_input_shape(shape: tuple[int, ...]) -> None:
    model = MBCNN()
    with pytest.raises(ValueError):
        model(torch.rand(shape))


def test_rejects_nonstandard_width() -> None:
    with pytest.raises(ValueError, match="fixed"):
        MBCNN(n_filters=32)
