from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch


def _read_csv_matrix(path: Path) -> np.ndarray:
    return np.loadtxt(path, delimiter=",", dtype=np.float64)


@lru_cache(maxsize=32)
def _load_et_linear_system(data_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    root = Path(data_dir)
    laplacian_L = _read_csv_matrix(root / "Lapmat.csv")
    sensitivity_S = _read_csv_matrix(root / "Jmat.csv")
    mask_M = _read_csv_matrix(root / "mask.csv")
    return laplacian_L, sensitivity_S, mask_M


@dataclass(frozen=True)
class LaplacianRegInitSpec:
    data_dir: str
    lambda_reg: float
    image_side: int = 64
    dtype: torch.dtype = torch.float32


class LaplacianRegularizedInitializer:
    def __init__(self, spec: LaplacianRegInitSpec, device: torch.device) -> None:
        lap_L_np, sens_S_np, mask_M_np = _load_et_linear_system(str(Path(spec.data_dir)))

        self.data_dir = str(Path(spec.data_dir))
        self.lambda_reg = float(spec.lambda_reg)
        self.image_side = int(spec.image_side)
        self.dtype = spec.dtype
        self.device = device

        L = torch.from_numpy(lap_L_np).to(device=device, dtype=self.dtype)
        S = torch.from_numpy(sens_S_np).to(device=device, dtype=self.dtype)
        M = torch.from_numpy(mask_M_np).to(device=device, dtype=self.dtype)

        self.L = L
        self.S = S
        self.S_t = S.transpose(0, 1).contiguous()
        self.M = M
        self.M_t = M.transpose(0, 1).contiguous()

        normal_A = self.S_t @ self.S + self.lambda_reg * self.L
        chol, info = torch.linalg.cholesky_ex(normal_A)
        self._normal_A = normal_A
        self._chol = chol
        self._chol_ok = bool((info == 0).all().item())

    def _solve_normal(self, rhs: torch.Tensor) -> torch.Tensor:
        if self._chol_ok:
            return torch.cholesky_solve(rhs, self._chol)
        return torch.linalg.solve(self._normal_A, rhs)

    def reconstruct(self, phi_b1m1: torch.Tensor) -> torch.Tensor:
        phi_bm = phi_b1m1.squeeze(1).squeeze(-1).to(device=self.device, dtype=self.dtype)
        phi_mb = phi_bm.t().contiguous()

        rhs_nb = self.S_t @ phi_mb
        x_hat_nb = self._solve_normal(rhs_nb)

        x_full = self.M_t @ x_hat_nb
        side = self.image_side
        x_img = x_full.view(side, side, -1).permute(2, 1, 0).unsqueeze(1).contiguous()
        return x_img


_INITIALIZER_CACHE: Dict[Tuple[str, float, str, str], LaplacianRegularizedInitializer] = {}


def callLapReg(data_dir: str, y_test: torch.Tensor, reg: float = 1e-4) -> torch.Tensor:
    device = y_test.device
    spec = LaplacianRegInitSpec(data_dir=str(Path(data_dir)), lambda_reg=float(reg), image_side=64, dtype=torch.float32)
    key = (spec.data_dir, float(spec.lambda_reg), str(device), str(spec.dtype))

    initializer = _INITIALIZER_CACHE.get(key)
    if initializer is None:
        initializer = LaplacianRegularizedInitializer(spec=spec, device=device)
        _INITIALIZER_CACHE[key] = initializer

    with torch.no_grad():
        return initializer.reconstruct(y_test)


def LapReg(J: torch.Tensor, y: torch.Tensor, lda: float, Lap: torch.Tensor) -> torch.Tensor:
    Jt = J.t().float()
    A = torch.add(torch.mm(Jt, J), float(lda) * Lap).float()
    rhs = torch.mm(Jt, y).float()
    return torch.linalg.solve(A, rhs).float()


def Convert2dImg(xest: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    result_tensor_1d = mask.t().mm(xest)
    result_tensor_2d = result_tensor_1d.view(64, 64, -1)
    return result_tensor_2d.permute(2, 1, 0)