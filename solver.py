from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from M1LapReg import callLapReg


def weighted_l1(phi_x_hat: torch.Tensor, x_ref: torch.Tensor, weight: float) -> torch.Tensor:
    return float(weight) * torch.mean(torch.abs(phi_x_hat - x_ref))


def weighted_tv(x_img: torch.Tensor, weight: float) -> torch.Tensor:
    w_var = torch.sum(torch.pow(x_img[:, :, :, :-1] - x_img[:, :, :, 1:], 2))
    h_var = torch.sum(torch.pow(x_img[:, :, :-1, :] - x_img[:, :, 1:, :], 2))
    return float(weight) * (h_var + w_var)


@dataclass(frozen=True)
class TrainRuntime:
    amp: bool = True
    grad_clip_norm: Optional[float] = None


class TranIUTrainer:
    def __init__(self, reconstructor: nn.Module, train_loader: Any, args: Any, b_data: Optional[torch.Tensor] = None) -> None:
        if getattr(args, "model_name", None) not in {"TranIU"}:
            raise ValueError(f"Unsupported model_name: {getattr(args, 'model_name', None)!r}")

        self.model_name = str(args.model_name)
        self.reconstructor = reconstructor
        self.train_loader = train_loader

        self.data_dir = str(args.data_dir)
        self.num_epochs = int(args.num_epochs)
        self.start_epoch = int(args.start_epoch)
        self.lr = float(args.lr)

        self.save_root = Path(args.save_path)
        self.multi_gpu = bool(getattr(args, "multi_gpu", False))
        self.log_interval = int(getattr(args, "log_interval", 10))
        self.test_epoch = int(getattr(args, "test_epoch", 10))
        self.save_interval = int(getattr(args, "save_interval", 5))

        self.runtime = TrainRuntime(
            amp=bool(getattr(args, "amp", True)),
            grad_clip_norm=(float(getattr(args, "grad_clip_norm")) if getattr(args, "grad_clip_norm", None) is not None else None),
        )

        self.b_data = b_data
        self.loss_mse = nn.MSELoss()
        self.optimizer = optim.Adam(self.reconstructor.parameters(), lr=self.lr, weight_decay=1e-4)

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.runtime.amp)

    def _ckpt_path(self, epoch: int) -> Path:
        return self.save_root / f"epoch_{int(epoch)}.ckpt"

    def save_model(self, epoch: int) -> None:
        self.save_root.mkdir(parents=True, exist_ok=True)
        torch.save(self.reconstructor.state_dict(), self._ckpt_path(epoch))

    def load_model(self, epoch: int) -> None:
        state = torch.load(self._ckpt_path(epoch), map_location="cuda:0")

        if isinstance(state, dict) and any(k.startswith("module.") for k in state.keys()):
            state = {k[7:]: v for k, v in state.items()}

        self.reconstructor.load_state_dict(state)

    def train(self) -> None:
        device = next(self.reconstructor.parameters()).device
        total_epochs = self.start_epoch + self.num_epochs
        epoch_loss_log = np.zeros(total_epochs + 1, dtype=np.float64)

        for epoch in range(self.start_epoch + 1, total_epochs + 1):
            self.reconstructor.train(True)
            t0 = time.time()
            running = 0.0

            for batch_idx, (phi_raw, x_ref_raw) in enumerate(self.train_loader):
                phi_b1m1 = phi_raw.unsqueeze(1).unsqueeze(-1).to(device=device, dtype=torch.float32, non_blocking=True)
                x_ref = x_ref_raw.unsqueeze(1).to(device=device, dtype=torch.float32, non_blocking=True)

                x_lap_init = callLapReg(data_dir=self.data_dir, y_test=phi_b1m1)

                with torch.cuda.amp.autocast(enabled=self.runtime.amp):
                    x_hat = self.reconstructor(x_lap_init, phi_b1m1)
                    if isinstance(x_hat, (tuple, list)):
                        x_hat = x_hat[0]
                    loss = self.loss_mse(x_hat, x_ref)

                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()

                if self.runtime.grad_clip_norm is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.reconstructor.parameters(), self.runtime.grad_clip_norm)

                self.scaler.step(self.optimizer)
                self.scaler.update()

                loss_val = float(loss.detach().item())
                running += loss_val
                epoch_loss_log[epoch] += loss_val

                if batch_idx % self.log_interval == 0:
                    print(f"loss={loss.detach().item():.4f}")

            if epoch % self.save_interval == 0:
                self.save_model(epoch)

            print("--------------------")
            print(
                f"Epoch {epoch} ({(time.time() - t0) / 60.0:.1f} min):    "
                f"loss={running / max(1, len(self.train_loader)):.4f}"
            )

    def test(self) -> torch.Tensor:
        if self.b_data is None:
            raise ValueError("b_data was not provided; cannot run test().")

        self.load_model(self.test_epoch)
        self.reconstructor.eval()

        device = next(self.reconstructor.parameters()).device
        with torch.no_grad():
            phi = self.b_data.to(device=device, dtype=torch.float32)
            x_lap_init = callLapReg(data_dir=self.data_dir, y_test=phi)
            x_hat = self.reconstructor(x_lap_init, phi)
            if isinstance(x_hat, (tuple, list)):
                x_hat = x_hat[0]
        return x_hat


Solver = TranIUTrainer
