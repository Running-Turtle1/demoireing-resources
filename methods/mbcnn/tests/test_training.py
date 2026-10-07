from pathlib import Path

import torch
from torch import nn

from mbcnn import MBCNNLoss
from mbcnn.training import (
    TrainingState,
    restore_training_checkpoint,
    save_training_checkpoint,
    train_one_epoch,
    validate,
)


class TinyMultiScaleModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 3, 1)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        output = torch.sigmoid(self.conv(inputs))
        output_half = nn.functional.interpolate(output, scale_factor=0.5)
        output_quarter = nn.functional.interpolate(output, scale_factor=0.25)
        return output_quarter, output_half, output


def make_loader() -> torch.utils.data.DataLoader:
    samples = [
        {
            "input": torch.rand(3, 16, 16),
            "target": torch.rand(3, 16, 16),
            "id": "sample",
        }
    ]
    return torch.utils.data.DataLoader(samples, batch_size=1)


def test_training_validation_and_resume(tmp_path: Path) -> None:
    device = torch.device("cpu")
    model = TinyMultiScaleModel()
    criterion = MBCNNLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    state = TrainingState(epoch=0)
    train_generator = torch.Generator().manual_seed(1)
    val_generator = torch.Generator().manual_seed(2)

    train_metrics = train_one_epoch(
        model,
        criterion,
        make_loader(),
        optimizer,
        scaler,
        device,
        state,
        accumulation_steps=1,
        amp=False,
        grad_clip_norm=0.0,
        log_every_batches=0,
        max_batches=None,
    )
    val_metrics = validate(
        model,
        criterion,
        make_loader(),
        device,
        amp=False,
        max_samples=1,
        preview_path=tmp_path / "preview.png",
    )
    state.best_psnr = float(val_metrics["psnr"])
    checkpoint_path = tmp_path / "last.pth"
    save_training_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        scheduler=None,
        state=state,
        config={"test": True},
        train_generator=train_generator,
        val_generator=val_generator,
    )

    restored_model = TinyMultiScaleModel()
    restored_optimizer = torch.optim.Adam(restored_model.parameters(), lr=1e-4)
    restored_scaler = torch.amp.GradScaler("cuda", enabled=False)
    restored_state = restore_training_checkpoint(
        checkpoint_path,
        model=restored_model,
        optimizer=restored_optimizer,
        scaler=restored_scaler,
        scheduler=None,
        train_generator=torch.Generator(),
        val_generator=torch.Generator(),
        device=device,
    )

    assert train_metrics["optimizer_steps"] == 1
    assert val_metrics["samples"] == 1
    assert (tmp_path / "preview.png").is_file()
    assert restored_state.epoch == 1
    assert restored_state.global_step == 1
    for key, value in model.state_dict().items():
        torch.testing.assert_close(restored_model.state_dict()[key], value)

