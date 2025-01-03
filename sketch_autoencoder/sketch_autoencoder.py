import PIL.Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.v2.functional as Fv2
import lightning as L
import PIL
from typing import TypedDict
from taesd.taesd import TAESD
from .model_blocks import ConvNextBlock, ReEncoder, LayerNorm2d
from math import prod

class Losses(TypedDict):
    recon: torch.Tensor
    clip: torch.Tensor
    cos: torch.Tensor

class SketchAutoencoder(L.LightningModule):
    def __init__(self, vae_img_size: torch.Size, vae: TAESD, clip_embed_dims: int, 
                 sem_chans: int,
                 hidden_chans: int, num_blocks: int,
                 lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters(ignore=['vae'])
        self.vae_chans = vae_img_size[0]
        self.sem_chans = sem_chans
        self.style_chans = self.vae_chans - self.sem_chans
        
        self.vae = vae
        self.vae.requires_grad_(False)

        self.reencoder = ReEncoder(self.vae_chans)
        hidden_dims = prod((hidden_chans, vae_img_size[1], vae_img_size[2]))
        self.embedder = nn.Sequential(
            nn.Conv2d(sem_chans, sem_chans, kernel_size=1),
            nn.AdaptiveAvgPool2d(1),
            LayerNorm2d(sem_chans, affine=True),
            nn.Flatten(),
            nn.Linear(sem_chans, clip_embed_dims, bias=False),
            nn.Linear(clip_embed_dims, clip_embed_dims)
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
            'recon': F.mse_loss(z_hat, z),
            'clip': F.mse_loss(e_hat, e),
            'cos': (1 - F.cosine_similarity(e_hat, e)).mean()
        }
    
    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor]):
        z, e = batch # VAE latents, clip embeddings
        style, sem, e_hat, z_hat = self.forward(z)

        losses = self.calc_losses(z_hat, z, e_hat, e)
        self.log_dict(losses)
        loss = losses['recon'] + losses['clip']
        self.log('loss', loss, prog_bar=True)
        return loss
    
    def decode_top_vae_latent(self, x: torch.Tensor) -> PIL.Image.Image:
        x = x[0].unsqueeze(0)
        img = self.vae.decoder(x).float().clip(min=0, max=1)
        return Fv2.to_pil_image(img[0])
    
    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        z, e = batch # VAE latents, clip embeddings
        style, sem, e_hat, z_hat = self.forward(z)

        losses = self.calc_losses(z_hat, z, e_hat, e)
        self.log_dict(losses, prog_bar=True)

        if batch_idx == 0:
            with torch.no_grad():
                self.logger.log_image(key='original', images=[self.decode_top_vae_latent(z)])
                self.logger.log_image(key='recon_img', images=[self.decode_top_vae_latent(z_hat)])
                sem_pad = torch.cat((torch.zeros_like(style), sem), dim=1)
                self.logger.log_image(key='semantic', images=[self.decode_top_vae_latent(self.reencoder.decode(sem_pad))])

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
