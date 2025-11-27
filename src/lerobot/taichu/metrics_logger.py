import json
import os
from pathlib import Path
from typing import Dict, Any, Optional

class MetricsLogger:
    def __init__(self, output_dir: str, filename: str = "loss.json"):
        self.output_path = Path(output_dir) / filename
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.train_metrics = []
        self._cumulative_loss = 0.0
        self._count = 0

        # Load existing metrics if resuming
        if self.output_path.exists():
            with open(self.output_path, "r") as f:
                data = json.load(f)
                self.train_metrics = data.get("train", [])
                if self.train_metrics:
                    last = self.train_metrics[-1]
                    self._cumulative_loss = last["mean_loss"] * len(self.train_metrics)
                    self._count = len(self.train_metrics)

    def log_step(
        self,
        epoch: int = 1,
        step: int = 1,
        global_step: int = 1,
        loss: float = 0.0,
        lr: float = 0.0,
    ) -> None:
        self._count += 1
        self._cumulative_loss += loss
        mean_loss = self._cumulative_loss / self._count

        entry = {
            "epoch": epoch,
            "step": step,
            "global_step": global_step,
            "loss": float(loss),
            "lr": float(lr),
            "mean_loss": float(mean_loss),
        }
        self.train_metrics.append(entry)

        # Immediately write to disk (append-safe via full rewrite)
        self._save()

    def _save(self) -> None:
        with open(self.output_path, "w") as f:
            json.dump({"train": self.train_metrics}, f, indent=2)