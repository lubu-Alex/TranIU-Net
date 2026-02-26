from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Optional, Tuple

import numpy as np
from torch.utils.data import DataLoader, Dataset

SplitName = Literal["train", "val", "test", "test_3obj"]


@dataclass(frozen=True)
class SplitLayout:
    phi_template: str
    x_file: str


_SPLITS: Mapping[SplitName, SplitLayout] = {
    "train": SplitLayout(phi_template="measV_train_{snr_db}dB.npy", x_file="img_train.npy"),
    "val": SplitLayout(phi_template="measV_val_{snr_db}dB.npy", x_file="img_val.npy"),
    "test": SplitLayout(phi_template="measV_test_{snr_db}dB.npy", x_file="img_test.npy"),
    "test_3obj": SplitLayout(phi_template="measV3_{snr_db}dB.npy", x_file="img3.npy"),
}


class ElectricalTomographyDataset(Dataset):
    def __init__(
        self,
        split: SplitName,
        root_dir: str | Path,
        snr_db: int,
        transform: Optional[Callable[[Any], Any]] = None,
    ) -> None:
        if split not in _SPLITS:
            raise ValueError(f"Unsupported split={split!r}. Expected one of {sorted(_SPLITS.keys())}.")

        root = Path(root_dir)
        layout = _SPLITS[split]
        phi_path = root / layout.phi_template.format(snr_db=int(snr_db))
        x_path = root / layout.x_file

        self.phi_boundary = np.load(phi_path)
        self.x_reference = np.load(x_path)
        self.transform = transform

        if int(self.phi_boundary.shape[0]) != int(self.x_reference.shape[0]):
            raise ValueError(
                f"Split size mismatch: phi={int(self.phi_boundary.shape[0])} vs x={int(self.x_reference.shape[0])}."
            )

    def __len__(self) -> int:
        return int(self.phi_boundary.shape[0])

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        phi = self.phi_boundary[idx]
        x_ref = self.x_reference[idx]
        if self.transform is not None:
            phi = self.transform(phi)
        return phi, x_ref


def build_dataloader(
    split: SplitName,
    root_dir: str | Path,
    snr_db: int = 40,
    transform: Optional[Callable[[Any], Any]] = None,
    batch_size: int = 128,
    num_workers: int = 1,
    shuffle: bool = False,
    pin_memory: bool = True,
) -> DataLoader:
    dataset = ElectricalTomographyDataset(split=split, root_dir=root_dir, snr_db=int(snr_db), transform=transform)
    print("Total", split, " data size: ", len(dataset))
    return DataLoader(
        dataset=dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
    )


def create_data_splits(
    root_dir: str | Path,
    snr_db: int,
    batch_size: int = 128,
    transform: Optional[Callable[[Any], Any]] = None,
):
    train_loader = build_dataloader(
        split="train",
        root_dir=root_dir,
        snr_db=int(snr_db),
        batch_size=int(batch_size),
        transform=transform,
        shuffle=True,
    )
    val_loader = build_dataloader(
        split="val",
        root_dir=root_dir,
        snr_db=int(snr_db),
        batch_size=int(batch_size),
        transform=transform,
        shuffle=False,
    )
    test_loader = build_dataloader(
        split="test",
        root_dir=root_dir,
        snr_db=int(snr_db),
        batch_size=int(batch_size),
        transform=transform,
        shuffle=False,
    )
    test_3obj_loader = build_dataloader(
        split="test_3obj",
        root_dir=root_dir,
        snr_db=int(snr_db),
        batch_size=int(batch_size),
        transform=transform,
        shuffle=False,
    )
    return train_loader, val_loader, test_loader, test_3obj_loader


EMTDataset = ElectricalTomographyDataset
get_loader = build_dataloader
DataSplit = create_data_splits