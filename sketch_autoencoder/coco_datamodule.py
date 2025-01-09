import torch
import lightning as L
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
import torchvision.transforms.v2 as v2
from taesd.taesd import TAESD
import open_clip

VAE_TRANSFORMS = v2.Compose([
    lambda i: i.convert('RGB') if i.mode != 'RGB' else i,
    v2.PILToTensor(),
    v2.ToDtype(torch.float, scale=True),
    v2.Resize(256),
    v2.CenterCrop(256)
])

class CocoDatamodule(L.LightningDataModule):
    def __init__(self, vae: TAESD, batch_size: int, num_workers: int):
        super().__init__()
        self.vae = vae.bfloat16()
        self.num_workers = num_workers
        self.batch_size = batch_size

        # store CLIP only within this module so we can delete it once all the processing is done
        # saves VRAM
        clip_model = 'ViT-B-32'
        self.clip, _, self.clip_preprocess = open_clip.create_model_and_transforms(clip_model, pretrained='laion2b_s34b_b79k', precision='bf16')
        self.clip_tokenizer = open_clip.get_tokenizer(clip_model)
        self.clip.eval()
        self.clip.requires_grad_(False)
        self.clip = self.clip.to('cuda')

    def setup(self, stage: str):
        self.train = self.load_split('train')
        self.val =  self.load_split('val')
        self.test = self.load_split('test')

        del self.clip
        self.vae = self.vae.float()
        del self.vae

        print('train/val/test sizes', len(self.train), len(self.val), len(self.test))

    def load_split(self, split: str):
        # must use custom hash id
        return load_dataset('romrawinjp/mscoco', split=split) \
            .remove_columns(['filename', 'cocoid']) \
            .map(self.map_coco_row, remove_columns=['image', 'en'], new_fingerprint=f'{split}-myhash') \
            .with_format("torch")
    
    @torch.no_grad()
    def map_coco_row(self, row):
        vae_img = VAE_TRANSFORMS(row['image']).bfloat16().to('cuda')
        vae_img = self.vae.encoder(vae_img).clip(min=-3, max=3)
        
        clip_img = self.clip_preprocess(row['image']).unsqueeze(0).bfloat16().to('cuda')
        clip_img = self.clip.encode_image(clip_img).squeeze(0)

        # each image has at least 5 captions
        clip_txt = self.clip_tokenizer(row['en'][:5]).to('cuda')
        clip_txt = self.clip.encode_text(clip_txt)

        return {'vae_img': vae_img.bfloat16(), 'clip_img': clip_img.bfloat16(), 'clip_txt': clip_txt.bfloat16()}

    def train_dataloader(self):
        return DataLoader(self.train, shuffle=True, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True)
