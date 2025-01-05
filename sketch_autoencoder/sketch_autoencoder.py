import PIL.Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.v2.functional as Fv2
import lightning as L
import PIL
from typing import TypedDict
from taesd.taesd import TAESD
from .model_blocks import DownBlock, ReEncoder, LayerNorm2d
import numpy as np

class Losses(TypedDict):
    recon: torch.Tensor
    clip: torch.Tensor
    cos: torch.Tensor
    ortho: torch.Tensor

class SketchAutoencoder(L.LightningModule):
    def __init__(self, vae_img_size: torch.Size, vae: TAESD, clip_embed_dims: int, 
                 sem_chans: int,
                 init_dims: int, num_blocks: int = 3,
                 lr: float = 1e-4):
        super().__init__()
        self.save_hyperparameters(ignore=['vae'])
        self.vae_chans = vae_img_size[0]
        self.sem_chans = sem_chans
        self.style_chans = self.vae_chans - self.sem_chans
        
        self.vae = vae
        self.vae.requires_grad_(False)

        self.reencoder = ReEncoder(self.vae_chans)
        last_chans = init_dims * (2**num_blocks)
        self.embedder = nn.Sequential(
            nn.Conv2d(self.sem_chans, init_dims, kernel_size=1),
            *[DownBlock(init_dims * 2**i) for i in range(num_blocks)],
            nn.AdaptiveAvgPool2d(1),
            LayerNorm2d(last_chans, affine=True),
            nn.Flatten(),
            nn.Linear(last_chans, last_chans, bias=False),
            nn.Linear(last_chans, clip_embed_dims)
        )
        
        # training parameters
        self.lr = lr

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        style, sem = self.reencoder.encode(z).split([self.style_chans, self.sem_chans], dim=1)
        embed = self.embedder(sem)
        with torch.no_grad():
            z_hat = self.reencoder.decode(torch.cat((style, sem), dim=1))
        return style, sem, embed, z_hat
    
    def calc_losses(self, z_hat: torch.Tensor, z: torch.Tensor,
                    e_hat: torch.Tensor, e: torch.Tensor) -> Losses:

        return {
            'recon': F.smooth_l1_loss(z_hat, z),
            'clip': F.smooth_l1_loss(e_hat, e),
            'cos': (1 - F.cosine_similarity(e_hat, e)).mean(),
            'ortho': F.mse_loss(self.reencoder.weight.T @ self.reencoder.weight, torch.eye(self.vae_chans).to(self.device))
        }
    
    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor]):
        z, e = batch # VAE latents, clip embeddings
        style, sem, e_hat, z_hat = self.forward(z)

        losses = self.calc_losses(z_hat, z, e_hat, e)
        self.log_dict(losses, on_epoch=True)
        loss = losses['recon'] + losses['ortho'] + losses['clip'] + 0.1*losses['cos']
        self.log('loss', loss, prog_bar=True, on_epoch=True)
        return loss
    
    def decode_top_vae_latent(self, x: torch.Tensor) -> PIL.Image.Image:
        x = x[0].unsqueeze(0)
        img = self.vae.decoder(x).float().clip(min=0, max=1)
        return Fv2.to_pil_image(img[0])
    
    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        z, e = batch # VAE latents, clip embeddings
        style, sem, e_hat, z_hat = self.forward(z)

        losses = self.calc_losses(z_hat, z, e_hat, e)
        self.log_dict(losses, on_epoch=True)

        if batch_idx == 0:
            with torch.no_grad():
                self.logger.log_image(key='original', images=[self.decode_top_vae_latent(z)])
                self.logger.log_image(key='recon_img', images=[self.decode_top_vae_latent(z_hat)])
                sem_pad = torch.cat((torch.zeros_like(style), sem), dim=1)
                self.logger.log_image(key='semantic', images=[self.decode_top_vae_latent(self.reencoder.decode(sem_pad))])

    def configure_optimizers(self):
        optim = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optim
        # sched = torch.optim.lr_scheduler.OneCycleLR(optim, epochs=10, steps_per_epoch=400, max_lr = self.lr)
        # return [optim], [sched]
