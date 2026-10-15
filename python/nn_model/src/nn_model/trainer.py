"""Training utilities"""

import heapq
import os
from collections.abc import Callable
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset


class TrainDataLoader:
    """Manages the batch size depending on the epoch number"""

    def __init__(
        self,
        dataset: Dataset,
        min_bs: int,
        max_bs: int,
        target_epoch: int,
        dataloader_kwargs: dict[str, Any] | None = None,
    ):
        """Create TrainDataLoader

        Args:
            dataset (Dataset): The dataset
            min_bs (int): Start batch size
            max_bs (int): Target batch size
            target_epoch (int): The epoch at which the batch size will be maximum
            dataloader_kwargs (dict[str, Any] | None, optional): The DataLoader arguments

        """
        self.dataset = dataset
        self.min_bs = max(1, min_bs)
        self.max_bs = max(1, max_bs)
        self.target_epoch = target_epoch
        self.dataloader_kwargs = dataloader_kwargs or {}
        self.current_bs = self.min_bs
        self._update_loader(self.current_bs)

    def _update_loader(self, bs):
        self._loader = DataLoader(self.dataset, batch_size=bs, **self.dataloader_kwargs)

    def update_epoch(self, epoch: int):
        if self.target_epoch <= 0:
            target = self.max_bs
        else:
            frac = min(1.0, epoch / self.target_epoch)
            target = round(self.min_bs + frac * (self.max_bs - self.min_bs))
        target = max(1, min(target, self.max_bs))
        if target != self.current_bs:
            self.current_bs = target
            self._update_loader(self.current_bs)

    def __iter__(self):
        """Iterate over the dataset batches

        Returns:
            Iter: Dataset iterator

        """
        return self._loader.__iter__()

    def __len__(self):
        """Get number of batches in the dataset

        Returns:
            int: Number of batches in the dataset

        """
        return self._loader.__len__()


class LRScheduler:
    """Gradually changes the learning rate between epochs"""

    def __init__(
        self, optimizer: torch.optim.Optimizer, epoch_lr: list[tuple[int, float]]
    ):
        """Create LRScheduler

        Args:
            optimizer (torch.optim.Optimizer): The optimizer for which the lr parameter should be changed
            epoch_lr (list[tuple[int, float]]): List of (epoch_index, lr) pairs

        """
        self.opt = optimizer
        self.pairs = sorted(epoch_lr, key=lambda x: x[0])

    def max_epoch(self) -> int:
        return self.pairs[-1][0]

    def step(self, epoch: int, step: int, n_steps: int):
        lr = self._calc_lr(epoch, step / n_steps)
        for g in self.opt.param_groups:
            g["lr"] = lr
        return lr

    def _calc_lr(self, epoch: int, ratio: float) -> float:
        index = len(self.pairs) - 1
        for i, (pair_epoch, _) in enumerate(self.pairs):
            if epoch < pair_epoch:
                index = i
                break
        min_pair = (0, 0.0) if index == 0 else self.pairs[index - 1]
        max_pair = self.pairs[index]
        frac = min(1.0, (epoch - min_pair[0] + ratio) / (max_pair[0] - min_pair[0]))
        return min_pair[1] + frac * (max_pair[1] - min_pair[1])


class CheckpointManager:
    """Manages saving and loading of training state"""

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        save_dir: str,
        keep_last: int = 3,
        keep_best: int = 3,
    ):
        """Create CheckpointManager

        Args:
            model (torch.nn.Module): The model which state will be saved
            optimizer (torch.optim.Optimizer): The optimizer which state will be saved
            save_dir (str): Directory where the training state will be saved to
            keep_last (int, optional): Number of last checkpoints to keep.
            keep_best (int, optional): Number of best checkpoints to keep.

        """
        self.save_dir = save_dir
        self.model = model
        self.optimizer = optimizer
        os.makedirs(save_dir, exist_ok=True)
        self.keep_last = max(1, keep_last)
        self.keep_best = max(1, keep_best)
        self.last = []
        self.best = []

    def save(
        self,
        epoch: int,
        loss_summary: dict[str, float],
        score: float,
    ):
        state = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epoch": epoch,
            "loss_summary": loss_summary,
        }
        path = os.path.join(self.save_dir, f"last_ep{epoch}.pt")
        torch.save(state, path)
        self.last.insert(0, path)
        while len(self.last) > self.keep_last:
            old = self.last.pop()
            if os.path.exists(old):
                os.remove(old)
        self.save_best(epoch, state, score)

    def load(self, path: str):
        state = torch.load(path, map_location="cpu")
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        return state

    def save_best(self, epoch: int, state: dict, score: float):
        path = os.path.join(self.save_dir, f"best_ep{epoch}_s{score:.6f}.pt")
        if len(self.best) < self.keep_best:
            heapq.heappush(self.best, (-score, path))
            torch.save(state, path)
        else:
            worst_neg, worst_path = self.best[0]
            worst_score = -worst_neg
            if score < worst_score:
                heapq.heapreplace(self.best, (-score, path))
                torch.save(state, path)
                if os.path.exists(worst_path):
                    os.remove(worst_path)

    def load_last(self):
        if len(self.last) > 0:
            self.load(self.last[0])

    def save_resume(self, epoch: int):
        resume = {
            "epoch": epoch,
            "last_checkpoints": self.last,
            "best_checkpoints": self.best,
        }
        torch.save(resume, os.path.join(self.save_dir, "resume.pt"))

    def load_resume(self):
        state = torch.load(os.path.join(self.save_dir, "resume.pt"), map_location="cpu")
        self.last = state["last_checkpoints"]
        self.best = state["best_checkpoints"]
        return state


class Trainer:
    """Manages the training process"""

    def __init__(
        self,
        train_fn: Callable[[torch.Tensor], float],
        eval_fn: Callable[[torch.Tensor], float],
        train_loader: TrainDataLoader,
        eval_loader: DataLoader,
        scheduler: LRScheduler,
        ckpt_mgr: CheckpointManager,
        print_interval: int = 100,
    ):
        """Create Trainer

        Args:
            train_fn (Callable[[torch.Tensor], float]): The training step implementation, accepts the batch tensor, returns the loss
            eval_fn (Callable[[torch.Tensor], float]): The validation step implementation, accepts the batch tensor, returns the loss
            train_loader (TrainDataLoader): The training data loader
            eval_loader (DataLoader): The validation data loader
            scheduler (LRScheduler): The learning rate scheduler
            ckpt_mgr (CheckpointManager): Checkpoint manager
            print_interval (int, optional): Print the training result at each step interval

        """
        self.train_fn = train_fn
        self.eval_fn = eval_fn
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.lr_scheduler = scheduler
        self.ckpt_mgr = ckpt_mgr
        self.print_interval = print_interval
        self.epochs = scheduler.max_epoch()
        self.start_epoch = 1

    def load(self):
        resume = self.ckpt_mgr.load_resume()
        self.start_epoch = resume.get("epoch", 1)
        self.ckpt_mgr.load_last()

    def train(self):
        for epoch in range(self.start_epoch, self.epochs + 1):
            self.train_loader.update_epoch(epoch)

            ep_acc = 0.0
            ep_steps = 0
            i_acc = 0.0
            total_steps = len(self.train_loader)
            for batch in self.train_loader:
                self.lr_scheduler.step(epoch, ep_steps, total_steps)
                loss = self.train_fn(batch)
                ep_acc += loss
                i_acc += loss
                ep_steps += 1

                if ep_steps % self.print_interval == 0:
                    print(
                        f"[Epoch {epoch}] iter {ep_steps}/{total_steps} loss={i_acc / self.print_interval:.6g}"
                    )
                    i_acc = 0.0

            avg_train_loss = ep_acc / max(1, ep_steps)
            print(f"[Epoch {epoch}] avg train loss = {avg_train_loss:.6g}")

            eval_acc = 0.0
            eval_steps = 0
            for batch in self.eval_loader:
                loss = self.eval_fn(batch)
                eval_acc += loss
                eval_steps += 1
            avg_eval_loss = eval_acc / max(1, eval_steps)
            print(f"[Epoch {epoch}] avg eval loss = {avg_eval_loss:.6g}")

            loss_summary = {"train_loss": avg_train_loss, "eval_loss": avg_eval_loss}
            self.ckpt_mgr.save(epoch, loss_summary, avg_eval_loss)
            self.ckpt_mgr.save_resume(epoch + 1)
