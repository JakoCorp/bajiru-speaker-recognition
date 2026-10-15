import argparse

import model_preset
import torch
from torch import nn
from torch.utils.data import DataLoader

from nn_model.dataset import (
    AudioScoreDataset,
    RandomOffsetDataset,
)
from nn_model.trainer import (
    CheckpointManager,
    LRScheduler,
    TrainDataLoader,
    Trainer,
)

torch.backends.nnpack.set_flags(False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

segment_samples = model_preset.segment_samples
win_length = model_preset.win_length
hop_length = model_preset.hop_length
n_classes = model_preset.n_classes
model = model_preset.model
model = model.to(device)
model.mel_spec.use_fft = True


def make_source(
    base_path: str,
) -> tuple[str, str]:
    """Create audio and scores source configuration

    Args:
        base_path (str): Base path to the audio/scores dataset files

    Returns:
        tuple[str, str]: The source configuration

    """
    return (base_path + ".audio", base_path + ".scores")


# Prepared training files
train_files = [
    make_source("train"),
]
eval_files = [
    make_source("eval"),
]

# Learning rate per epoch, gradually changes from one epoch to another
epoch_lr = [(1, 1e-3), (10, 1e-4), (40, 1e-5), (200, 1e-6)]

train_batching = {"from": 32, "to": 64, "ep": 10}
eval_batch_size = 64


# Using the random window offset to improve training quality
train_ds = RandomOffsetDataset(
    train_files,
    hop_length,
    win_length,
    segment_samples,
    n_classes,
)
eval_ds = AudioScoreDataset(
    eval_files,
    hop_length,
    win_length,
    segment_samples,
    n_classes,
)


criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(
    model.parameters(), lr=1e-4, betas=(0.9, 0.98), weight_decay=0.02
)


def train_step(batch):
    model.train()
    batch_audio, batch_target = batch
    batch_audio = batch_audio.to(device)
    batch_target: torch.Tensor = batch_target.to(device)

    optimizer.zero_grad()
    _, logits = model.forward_train(batch_audio, True)
    loss: torch.Tensor = criterion(logits, batch_target)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return loss.item()


def eval_step(batch):
    model.eval()
    batch_audio, batch_target = batch
    with torch.no_grad():
        batch_audio = batch_audio.to(device)
        batch_target = batch_target.to(device)
        _, logits = model.forward_train(batch_audio, False)
        return criterion(logits, batch_target).detach().item()


trainer = Trainer(
    train_step,
    eval_step,
    TrainDataLoader(
        train_ds,
        train_batching["from"],
        train_batching["to"],
        train_batching["ep"],
        {"shuffle": True, "drop_last": False},
    ),
    DataLoader(eval_ds, batch_size=eval_batch_size, shuffle=False),
    LRScheduler(optimizer, epoch_lr),
    CheckpointManager(model, optimizer, "checkpoints"),
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--restore", action="store_true")
    args = parser.parse_args()
    if args.restore:
        trainer.load()

    trainer.train()
