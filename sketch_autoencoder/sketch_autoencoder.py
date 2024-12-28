import PIL.Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.v2.functional as Fv2
import lightning as L
import PIL
from typing import TypedDict
from taesd.taesd import TAESD
from .model_blocks import ResBlock, InstanceNorm2d, ScaleTanh, ImgEmbedder, ImgChannelTransform
from .clip_wrapper import CLIPWrapper

class Losses(TypedDict):
    recon: torch.Tensor
    clip: torch.Tensor
    kl: torch.Tensor

class SketchAutoencoder(L.LightningModule):
    def __init__(self, vae_img_size: torch.Size, vae: TAESD, clip_embed_dims: int, # 640
                 sem_embed_dims: int,
                 tex_dims: int, tex_hidden_dims: int, num_tex_blocks: int,
                 lr: float = 1e-4):
        super().__init__()
        self.save_hyperparameters(ignore=['vae'])
        self.vae_img_size = vae_img_size
        self.vae = vae
        self.vae.requires_grad_(False)
        self.clip_embed_dims = clip_embed_dims

        # create semantic encoder
        self.embedder = ImgEmbedder(vae_img_size, clip_embed_dims, sem_embed_dims)

        # create texture autoencoder
        self.tex_encoder = nn.Sequential(
            nn.Conv2d(vae_img_size[0], tex_hidden_dims, kernel_size=1),
            *[ResBlock(tex_hidden_dims) for _ in range(num_tex_blocks)],
            nn.Conv2d(tex_hidden_dims, 2*tex_dims, kernel_size=1),
            nn.Tanh(3)
        )
        
        # training parameters
        self.lr = lr

    def encode(self, z: torch.Tensor):
        e_hat = self.embedder.img_to_embed(z)
        tex = self.tex_encoder(z)
        return e_hat, tex
        # tex_latents = self.tex_encoder(z)
        # tex_mu, tex_log_var = torch.tensor_split(tex_latents, 2, dim=1)
        # return e_hat, tex_mu, tex_log_var

    def decode(self, e: torch.Tensor, tex: torch.Tensor):
        sem = self.embedder.embed_to_img(e)
        z_hat = sem + tex
        return z_hat, sem
        # z_hat = self.z_norm(sem + self.tex_decoder(tex))
        # return z_hat, sem
    
    def calc_losses(self, e: torch.Tensor, z: torch.Tensor, e_hat: torch.Tensor, z_hat: torch.Tensor) -> tuple[Losses, torch.Tensor]:
        e = e.squeeze(1) # i goofed the dataset preparation, so there is an extra dim that needs to be removed
        clip_loss = F.mse_loss(F.normalize(e_hat), F.normalize(e)) 
        recon_loss = F.mse_loss(z, z_hat)
        # tex_kl_loss = (-0.5 * torch.sum(1 + tex_log_var - tex_mu**2 - tex_log_var.exp(), dim=1)).mean()

        return {
            'recon': recon_loss,
            'clip': clip_loss,
            # 'kl': tex_kl_loss
        }
    
    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor]):
        z, e = batch # VAE latents, clip embeddings
        e_hat, tex = self.encode(z)
        z_hat, _ = self.decode(e_hat, tex)

        losses = self.calc_losses(e, z, e_hat, z_hat)
        self.log_dict(losses)
        loss = losses['clip'] + losses['recon']
        self.log('loss', loss, prog_bar=True)
        return loss
    
    def decode_top_vae_latent(self, x: torch.Tensor) -> PIL.Image.Image:
        x = x[0].unsqueeze(0)
        img = self.vae.decoder(x).float().clip(min=0, max=1)
        return Fv2.to_pil_image(img[0])
    
    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        z, e = batch # VAE latents, clip embeddings
        e_hat, tex = self.encode(z)
        z_hat, sem = self.decode(e_hat, tex)

        losses = self.calc_losses(e, z, e_hat, z_hat)
        self.log_dict(losses, prog_bar=True)
        if batch_idx == 0:
            self.logger.log_image(key='original', images=[self.decode_top_vae_latent(z)])
            self.logger.log_image(key='recon_img ', images=[self.decode_top_vae_latent(z_hat)])
            self.logger.log_image(key='semantic', images=[self.decode_top_vae_latent(sem)])

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
    
def sample_vae(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    sigma = torch.exp(0.5*log_var) # 1/2 because variance = deviation square
    return torch.randn_like(mu) * sigma + mu