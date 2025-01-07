import torch
import lightning as L
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
import torchvision.transforms.v2 as v2

VAE_TRANSFORMS = v2.Compose([
    lambda i: i.convert('RGB') if i.mode != 'RGB' else i,
    v2.PILToTensor(),
    v2.ToDtype(torch.float, scale=True)
])

class CocoDatamodule(L.LightningDataModule):
    def __init__(self, vae, clip, clip_preprocess, clip_tokenizer, batch_size: int, num_workers: int):
        super().__init__()
        self.vae = vae
        self.clip = clip
        self.clip_preprocess = clip_preprocess
        self.clip_tokenizer = clip_tokenizer
        self.num_workers = num_workers
        self.batch_size = batch_size

    def setup(self, stage: str):
        self.train = self.load_split('train')
        self.val = self.load_split('val')
        self.test = self.load_split('test')

    def load_split(self, split: str):
        return load_dataset('romrawinjp/mscoco', split=split) \
            .remove_columns(['filename', 'cocoid']) \
            .map(self.map_coco_row)
    
    def map_coco_row(self, row):
        row['z'] = self.vae.encoder(VAE_TRANSFORMS(row['image']).bfloat16().to('cuda')).float()
        row['clip'] = self.clip.encode_image(self.clip_preprocess(row['image']).unsqueeze(0).bfloat16().to('cuda')).squeeze(0)
        row['en'] = self.clip.encode_text(self.clip_tokenizer(row['en'][:5]).to('cuda'))
        del row['image']

    def train_dataloader(self):
        return DataLoader(self.train, shuffle=True, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val, shuffle=True, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test, shuffle=True, batch_size=self.batch_size)
