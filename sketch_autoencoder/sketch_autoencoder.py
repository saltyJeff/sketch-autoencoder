import PIL.Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.v2.functional as Fv2
import lightning as L
import PIL
from typing import TypedDict
from taesd.taesd import TAESD
from .model_blocks import ConvNextBlock, ScaleTanh, ImgUnembedder, DownBlock, UpBlock
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
            DownBlock(self.vae_chans, enc_hidden_dims),
            *[ConvNextBlock(enc_hidden_dims) for _ in range(num_enc_blocks)],
            nn.Conv2d(enc_hidden_dims, 2*tex_chans, kernel_size=1)
        )
        self.dec_up_tex = nn.Upsample(scale_factor=2)
        self.decoder = nn.Sequential(
            nn.Conv2d(self.vae_chans + tex_chans, dec_hidden_dims, kernel_size=1),
            *[ConvNextBlock(dec_hidden_dims) for _ in range(num_dec_blocks)],
            nn.Conv2d(dec_hidden_dims, self.vae_chans, kernel_size=1),
            ScaleTanh(3)
        )
        
        # training parameters
        self.lr = lr

    def encode_tex(self, z: torch.Tensor):
        tex_mu, tex_log_var = self.tex_encoder(z).tensor_split(2, dim=1)
        return tex_mu, tex_log_var

    def decode(self, e: torch.Tensor, tex: torch.Tensor):
        e = F.normalize(e.squeeze(1)) # i goofed the dataset preparation, so there is an extra dim that needs to be removed
        sem_z = self.img_unembedder(e)
        sem_z_clone = sem_z.clone()
        sem_z_clone.detach()
        tex = self.dec_up_tex(tex)
        dec_input = torch.cat((sem_z_clone, tex), dim=1)
        z_hat = self.decoder(dec_input)
        return z_hat, sem_z
    
    def calc_losses(self, z: torch.Tensor, z_hat: torch.Tensor, sem: torch.Tensor,
                    e: torch.Tensor,
                    tex_mu: torch.Tensor, tex_logvar: torch.Tensor) -> Losses:
        e = e.squeeze(1) 
        clip_loss = F.mse_loss(z, sem)
        recon_loss = F.mse_loss(z, z_hat)
        tex_kl_loss = (-0.5 * torch.sum(1 + tex_logvar - tex_mu**2 - tex_logvar.exp(), dim=1)).mean()

        return {
            'recon': recon_loss,
            'clip': clip_loss,
            'kl': tex_kl_loss
        }
    
    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor]):
        z, e = batch # VAE latents, clip embeddings
        tex_mu, tex_logvar = self.encode_tex(z)
        tex = sample_vae(tex_mu, tex_logvar)
        z_hat, sem = self.decode(e, tex)

        losses = self.calc_losses(z, z_hat, sem, e, tex_mu, tex_logvar)
        self.log_dict(losses)
        loss = losses['recon'] + losses['kl'] + losses['clip']
        self.log('loss', loss, prog_bar=True)
        return loss
    
    def decode_top_vae_latent(self, x: torch.Tensor) -> PIL.Image.Image:
        x = x[0].unsqueeze(0)
        img = self.vae.decoder(x).float().clip(min=0, max=1)
        return Fv2.to_pil_image(img[0])
    
    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        z, e = batch # VAE latents, clip embeddings
        tex_mu, tex_logvar = self.encode_tex(z)
        tex = sample_vae(tex_mu, tex_logvar)
        z_hat, sem = self.decode(e, tex)

        losses = self.calc_losses(z, z_hat, sem, e, tex_mu, tex_logvar)
        self.log_dict(losses, prog_bar=True)

        if batch_idx == 0:
            self.logger.log_image(key='original', images=[self.decode_top_vae_latent(z)])
            self.logger.log_image(key='recon_img ', images=[self.decode_top_vae_latent(z_hat)])
            self.logger.log_image(key='semantic', images=[self.decode_top_vae_latent(sem)])

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
    
def sample_vae(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    sigma = torch.exp(0.5*log_var) # 1/2 because variance = stddev square
    return torch.randn_like(mu) * sigma + mu