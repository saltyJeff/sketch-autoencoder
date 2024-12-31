import PIL.Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.v2.functional as Fv2
import lightning as L
import PIL
from typing import TypedDict
from taesd.taesd import TAESD
from .model_blocks import ConvNextBlock, ScaleTanh, ImgUnembedder
from .clip_wrapper import CLIPWrapper

class Losses(TypedDict):
    recon: torch.Tensor
    clip: torch.Tensor
    kl: torch.Tensor

class SketchAutoencoder(L.LightningModule):
    def __init__(self, vae_img_size: torch.Size, vae: TAESD, clip_embed_dims: int, 
                 tex_chans: int, 
                 sem_dims: int,
                 enc_hidden_dims: int, num_enc_blocks: int,
                 dec_hidden_dims: int, num_dec_blocks: int,
                 lr: float = 1e-4):
        super().__init__()
        self.save_hyperparameters(ignore=['vae'])
        self.vae_chans = vae_img_size[0]
        
        self.vae = vae
        self.vae.requires_grad_(False)
        self.clip_embed_dims = clip_embed_dims

        # create img unembedder
        self.img_unembedder = ImgUnembedder(clip_embed_dims, vae_img_size, sem_dims)
        
        # create texture autoencoder
        self.tex_encoder = nn.Sequential(
            nn.Conv2d(self.vae_chans, enc_hidden_dims, kernel_size=1),
            *[ConvNextBlock(enc_hidden_dims) for _ in range(num_enc_blocks)],
            nn.Conv2d(enc_hidden_dims, tex_chans, kernel_size=1),
            nn.Tanh()
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(self.vae_chans+tex_chans, dec_hidden_dims, kernel_size=1),
            *[ConvNextBlock(dec_hidden_dims) for _ in range(num_dec_blocks)],
            nn.Conv2d(dec_hidden_dims, self.vae_chans, kernel_size=1),
            ScaleTanh(6)
        )
        
        # training parameters
        self.lr = lr

    def encode_tex(self, z: torch.Tensor):
        tex = self.tex_encoder(z)
        return tex

    def decode(self, e: torch.Tensor, tex: torch.Tensor):
        e = F.normalize(e.squeeze(1)) # i goofed the dataset preparation, so there is an extra dim that needs to be removed
        sem_z = self.img_unembedder(e)
        sem_z_clone = sem_z.clone().detach()
        dec_input = torch.cat((sem_z_clone, tex), dim=1)
        z_hat = sem_z_clone + self.decoder(dec_input)
        return z_hat, sem_z
    
    def calc_losses(self, z: torch.Tensor, z_hat: torch.Tensor, sem: torch.Tensor,
                    e: torch.Tensor) -> Losses:
        e = e.squeeze(1) 
        clip_loss = F.mse_loss(z, sem)
        recon_loss = F.mse_loss(z, z_hat)

        return {
            'recon': recon_loss,
            'clip': clip_loss,
        }
    
    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor]):
        z, e = batch # VAE latents, clip embeddings
        tex = self.encode_tex(z)
        z_hat, sem = self.decode(e, tex)

        losses = self.calc_losses(z, z_hat, sem, e)
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
        tex = self.encode_tex(z)
        z_hat, sem = self.decode(e, tex)

        losses = self.calc_losses(z, z_hat, sem, e)
        self.log_dict(losses, prog_bar=True)

        if batch_idx == 0:
            self.logger.log_image(key='original', images=[self.decode_top_vae_latent(z)])
            self.logger.log_image(key='recon_img ', images=[self.decode_top_vae_latent(z_hat)])
            self.logger.log_image(key='semantic', images=[self.decode_top_vae_latent(sem)])

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
    
