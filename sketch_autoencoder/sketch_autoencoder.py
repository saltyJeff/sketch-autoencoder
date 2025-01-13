import PIL.Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.v2.functional as Fv2
import lightning as L
import PIL
from typing import TypedDict
from taesd.taesd import TAESD
from .model_blocks import ImgEmbedder, InvertibleLinear, LayerNorm2d
import numpy as np
import random

class Losses(TypedDict):
    sem_clip: torch.Tensor
    style_clip: torch.Tensor
    ortho: torch.Tensor

class SketchAutoencoder(L.LightningModule):
    def __init__(self, vae_chans, vae: TAESD, clip_embed_dims: int,
                 sem_chans: int,
                 embed_init_dims: int, num_embed_blocks: int = 3,
                 lr: float = 1e-4):
        super().__init__()
        self.save_hyperparameters(ignore=['vae'])
        self.eye = None
        self.vae_chans = vae_chans
        self.sem_chans = sem_chans
        self.style_chans = self.vae_chans - self.sem_chans
        
        self.vae = vae
        self.vae.requires_grad_(False)

        self.z_transform = InvertibleLinear(self.vae_chans)
        self.sem_embedder = ImgEmbedder(self.sem_chans, clip_embed_dims, embed_init_dims, num_embed_blocks)
        self.style_embedder = ImgEmbedder(self.style_chans, clip_embed_dims, embed_init_dims, num_embed_blocks)
        
        # training parameters
        self.lr = lr

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        sem, style = self.z_transform.encode(z).split([self.sem_chans, self.style_chans], dim=1)
        sem_e = self.sem_embedder(sem)
        style_e = self.style_embedder(style)

        return sem, sem_e, style, style_e
    
    def calc_losses(self, sem_e_hat: torch.Tensor, sem_e: torch.Tensor,
                    style_e_hat: torch.Tensor, style_e: torch.Tensor) -> Losses:
        if self.eye is None:
            self.eye = torch.eye(self.vae_chans).to(self.device)
        return {
            'ortho': F.mse_loss(self.z_transform.weight.T @ self.z_transform.weight, self.eye, reduction='sum'),
            'sem_clip': F.l1_loss(sem_e_hat, sem_e),
            'style_clip': F.l1_loss(style_e_hat, style_e)
        }

    def load_batch(self, batch):
        z: torch.Tensor = batch['vae_img']
        clip_txt_idx = random.randrange(batch['clip_txt'].shape[1])
        clip_txt: torch.Tensor = batch['clip_txt'][:, clip_txt_idx, :].squeeze(1)
        clip_img: torch.Tensor = batch['clip_img']

        sem_e = F.normalize(clip_txt)
        style_e = F.normalize(clip_img) - sem_e

        return z.float(), sem_e.float(), style_e.float()
    
    def training_step(self, batch):
        z, sem_e, style_e = self.load_batch(batch)
        sem, sem_e_hat, style, style_e_hat = self.forward(z)

        losses = self.calc_losses(sem_e_hat, sem_e, style_e_hat, style_e)
        self.log_dict(losses, on_epoch=True)
        loss = losses['ortho'] + losses['sem_clip'] + losses['style_clip']
        self.log('loss', loss, prog_bar=True, on_epoch=True)
        return loss
    
    def decode_top_vae_latent(self, x: torch.Tensor) -> PIL.Image.Image:
        x = x[0].unsqueeze(0)
        img = self.vae.decoder(x).float().clip(min=0, max=1)
        return Fv2.to_pil_image(img[0])
    
    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        z, sem_e, style_e = self.load_batch(batch)
        sem, sem_e_hat, style, style_e_hat = self.forward(z)

        losses = self.calc_losses(sem_e_hat, sem_e, style_e_hat, style_e)
        self.log_dict(losses, prog_bar=True, on_epoch=True)

        if batch_idx == 0:
            with torch.no_grad():
                z_hat = torch.cat((sem, style), dim=1)
                z_hat = self.z_transform.decode(z_hat)
                self.logger.log_image(key='original', images=[self.decode_top_vae_latent(z)])
                self.logger.log_image(key='recon_img', images=[self.decode_top_vae_latent(z_hat)])
                sem_hat = torch.cat((sem, torch.zeros_like(style)), dim=1)
                sem_hat = self.z_transform.decode(sem_hat)
                self.logger.log_image(key='semantic', images=[self.decode_top_vae_latent(sem_hat)])

    def configure_optimizers(self):
        optim = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optim
        # sched = torch.optim.lr_scheduler.OneCycleLR(optim, epochs=10, steps_per_epoch=400, max_lr = self.lr)
        # return [optim], [sched]
