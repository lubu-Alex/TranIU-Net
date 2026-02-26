from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch

from loader import DataSplit
from Mainnet import TranIUNet
from solver import Solver

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="TranIU")
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--data_dir", type=str, default="./Data")
    parser.add_argument("--save_path", type=str, default="./models/TranIU/")
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--multi_gpu", action="store_true")
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--test_epoch", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=1)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--grad_clip_norm", type=float, default=None)
    parser.add_argument("--sgdm_mode", type=str, default="analytic", choices=["analytic", "learned"])
    return parser


def _load_csv_tensor(path: Path, *, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    arr = np.loadtxt(path, delimiter=",", dtype=float)
    return torch.from_numpy(arr).to(device=device, dtype=dtype)


def main() -> None:
    args = build_parser().parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    batch_size = 40
    dataset_root = Path("./Data/CircleCases_MultiLevel")
    train_loader, _, _, _ = DataSplit(root_dir=dataset_root, batch_size=batch_size, snr_db=30)

    device = torch.device("cuda:0")

    S_sens = _load_csv_tensor(Path("./Data/Jmat.csv"), device=device, dtype=torch.float32)

    mask_raw = _load_csv_tensor(Path("./Data/mask.csv"), device=device, dtype=torch.float32)
    mask_ET = mask_raw.reshape(-1, 64, 64).permute(0, 2, 1).reshape(-1, 4096)

    recon_net = TranIUNet(S_sens, mask_ET, 20, sgdm_mode=args.sgdm_mode).to(device=device)

    print("Total number of parameters TranIU: ", sum(p.numel() for p in recon_net.parameters()))

    if int(args.start_epoch) > 0:
        ckpt = Path(args.save_path) / f"epoch_{int(args.start_epoch)}.ckpt"
        recon_net.load_state_dict(torch.load(ckpt, map_location="cuda:0"))

    trainer = Solver(recon_net, train_loader, args, None)
    trainer.train()


if __name__ == "__main__":
    main()

