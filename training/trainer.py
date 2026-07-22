"""
training/trainer.py
-------------------
Model-agnostic training loop.

The Trainer does not import anything from models/. It receives a model
that satisfies the BaseModel interface and trains it. Changing the
architecture requires zero changes here.
"""

import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class Trainer:
    """
    Parameters
    ----------
    model       : nn.Module  — must implement forward(x) -> logits [B,1,H,W]
    loss_fn     : nn.Module  — accepts (logits, targets)
    optimizer   : torch.optim.Optimizer
    scheduler   : LR scheduler (optional)
    device      : "cuda" | "cpu" | "mps"
    output_dir  : path to save checkpoints and logs
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer,
        scheduler=None,
        device: str = "cuda",
        output_dir: str = "runs/experiment",
    ):
        self.model = model.to(device)
        self.loss_fn = loss_fn.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        val_every: int = 1,
        early_stopping_patience: Optional[int] = None,
        resume_from: Optional[str] = None,
    ):
        """
        early_stopping_patience : stop once this many epochs have passed
            since val_loss last improved (measured in epochs, but only
            checked at validation points, i.e. every `val_every` epochs).
            None disables early stopping and always runs `epochs` epochs.
        resume_from : path to a checkpoint saved by this Trainer. Restores
            model/optimizer/scheduler state and continues from the epoch
            after the one the checkpoint was saved at, instead of starting
            over from epoch 1.
        """
        start_epoch = 1
        best_val_loss = float("inf")
        epochs_since_improvement = 0

        if resume_from is not None:
            ckpt = self.load_checkpoint(resume_from)
            start_epoch = ckpt.get("epoch", 0) + 1
            best_val_loss = ckpt.get("best_val_loss", best_val_loss)
            epochs_since_improvement = ckpt.get("epochs_since_improvement", 0)
            print(
                f"Resumed from {resume_from}: starting at epoch {start_epoch} "
                f"(best_val_loss={best_val_loss:.4f})"
            )

        history = []

        for epoch in range(start_epoch, epochs + 1):
            t0 = time.time()
            train_loss = self._run_epoch(train_loader, training=True)
            elapsed = time.time() - t0

            log = {"epoch": epoch, "train_loss": train_loss, "time_s": round(elapsed, 1)}

            if epoch % val_every == 0:
                val_loss = self._run_epoch(val_loader, training=False)
                log["val_loss"] = val_loss
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_since_improvement = 0
                    self._save_checkpoint("best.pth", epoch, best_val_loss, epochs_since_improvement)
                else:
                    epochs_since_improvement += val_every

            if self.scheduler is not None:
                self.scheduler.step()

            history.append(log)
            self._print_log(log)
            self._save_checkpoint("last.pth", epoch, best_val_loss, epochs_since_improvement)

            if (
                early_stopping_patience is not None
                and epochs_since_improvement >= early_stopping_patience
            ):
                print(
                    f"Early stopping: val_loss has not improved in "
                    f"{epochs_since_improvement} epochs (patience={early_stopping_patience})"
                )
                break

        return history

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run_epoch(self, loader: DataLoader, training: bool) -> float:
        self.model.train(training)
        total_loss = 0.0
        context = torch.enable_grad if training else torch.no_grad

        with context():
            for batch in loader:
                images = batch["image"].to(self.device)
                masks = batch["mask"].to(self.device)

                logits = self.model(images)
                loss = self.loss_fn(logits, masks)

                if training:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                total_loss += loss.item()

        return total_loss / len(loader)

    def _save_checkpoint(
        self,
        filename: str,
        epoch: int,
        best_val_loss: float,
        epochs_since_improvement: int,
    ):
        torch.save(
            {
                "epoch": epoch,
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "scheduler_state": self.scheduler.state_dict() if self.scheduler is not None else None,
                "best_val_loss": best_val_loss,
                "epochs_since_improvement": epochs_since_improvement,
            },
            self.output_dir / filename,
        )

    def load_checkpoint(self, path: str) -> dict:
        """Restore model/optimizer/scheduler state from a checkpoint; returns the raw checkpoint dict."""
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        if self.scheduler is not None and ckpt.get("scheduler_state") is not None:
            self.scheduler.load_state_dict(ckpt["scheduler_state"])
        return ckpt

    @staticmethod
    def _print_log(log: dict):
        parts = [f"Epoch {log['epoch']:03d}"]
        parts.append(f"train_loss={log['train_loss']:.4f}")
        if "val_loss" in log:
            parts.append(f"val_loss={log['val_loss']:.4f}")
        parts.append(f"({log['time_s']}s)")
        print("  ".join(parts))
