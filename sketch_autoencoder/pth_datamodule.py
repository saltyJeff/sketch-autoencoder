import torch
import lightning as L
from pathlib import Path
from torch.utils.data import DataLoader, Dataset

class PthDataset(Dataset):
    def __init__(self, path: Path):
        super().__init__()
        self.path = path
        self.img_ids = list(set(map(lambda p: p.name.split('.')[0], path.iterdir())))
    def __len__(self):
        return len(self.img_ids)
    def __getitem__(self, index):
        img_id = self.img_ids[index]
        z = torch.load(self.path / (img_id+'.vae.pt'), weights_only=True)
        embed = torch.load(self.path / (img_id+'.clip.pt'), weights_only=True)
        return z, embed

class PthDatamodule(L.LightningDataModule):
    def __init__(self, data_dir: Path, batch_size: int, num_workers: int):
        super().__init__()
        self.num_workers = num_workers
        self.data_dir = data_dir
        self.batch_size = batch_size

    def setup(self, stage: str):
        self.train = PthDataset(self.data_dir / 'train')
        self.val = PthDataset(self.data_dir / 'val')
        self.test = PthDataset(self.data_dir / 'test')

    def train_dataloader(self):
        return DataLoader(self.train, shuffle=True, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val, shuffle=True, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test, shuffle=True, batch_size=self.batch_size)
