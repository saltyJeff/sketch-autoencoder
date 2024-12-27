import PIL.Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.v2.functional as Fv2
import lightning as L
import PIL
from typing import TypedDict
from taesd.taesd import TAESD
from .model_blocks import ResBlock, AdaNormBlock, InstanceNorm2d, ScaleTanh
from .clip_wrapper import CLIPWrapper

class Losses(TypedDict):
    recon: torch.Tensor
    clip: torch.Tensor
    sparsity: torch.Tensor
    kl: torch.Tensor


class SketchAutoencoder(L.LightningModule):
    def __init__(self, hidden_dims: int, vae_dims: int, semantic_dims: int, texture_dims: int, num_enc_blocks: int, num_tex_blocks: int,
                 vae: TAESD,
                 clip: CLIPWrapper,
                 lr: float = 1e-4):
        super().__init__()
        self.save_hyperparameters(ignore=['vae', 'clip'])
        self.hidden_dims = hidden_dims
        self.vae_dims = vae_dims
        self.semantic_dims = semantic_dims
        self.texture_dims = texture_dims
        self.num_enc_blocks = num_enc_blocks
        self.num_texture_blocks = num_tex_blocks
        self.vae = vae
        self.vae.requires_grad_(False)
        self.clip = clip
        self.clip.requires_grad_(False)
        self.lr = lr

        encoder_dims = semantic_dims + 2 * texture_dims # semantic dims + (mu, sigma) * texture dims
        self.encoder_stem = nn.Sequential(
            nn.Conv2d(vae_dims, encoder_dims, kernel_size=3, padding=1),
            InstanceNorm2d()
        )
        self.encoder = nn.Sequential(
            *[ResBlock(encoder_dims) for _ in range(num_enc_blocks)]
        )

        self.semantic_decoder_in = nn.Sequential(
            nn.Conv2d(semantic_dims, hidden_dims, kernel_size=1),
            InstanceNorm2d()
        )
        self.texture_decoder_blocks = nn.ModuleList([
            AdaNormBlock(hidden_dims, texture_dims) for _ in range(num_tex_blocks)
        ])
        self.decoder_out = nn.Sequential(
            nn.Conv2d(hidden_dims, vae_dims, kernel_size=1),
            ScaleTanh(scale=self.vae.latent_magnitude)
        )
        
    def encode(self, x: torch.Tensor):
        latents = self.encoder_stem(x)
        latents = self.encoder(latents)
        sem, tex_mu, tex_log_var = torch.split(latents, [self.semantic_dims, self.texture_dims, self.texture_dims], dim=1)
        sem = F.relu(sem)

        return sem, tex_mu, tex_log_var
    
    def decode(self, sem: torch.Tensor, tex: torch.Tensor):
        hidden_dims = self.semantic_decoder_in(sem)
        x_sem_hat = self.decoder_out(hidden_dims)

        for block in self.texture_decoder_blocks:
            hidden_dims = block(hidden_dims, tex)

        x_hat = self.decoder_out(hidden_dims)

        return x_hat, x_sem_hat

    @staticmethod
    def sample_vae(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        sigma = torch.exp(0.5*log_var) # 1/2 because variance = deviation square
        return torch.randn_like(mu) * sigma + mu
    
    def calc_losses(self, x: torch.Tensor, x_hat: torch.Tensor, x_sem_hat: torch.Tensor, 
                    clip_embed: torch.Tensor, sem: torch.Tensor, tex_mu: torch.Tensor, tex_log_var: torch.Tensor) -> tuple[Losses, torch.Tensor]:
        sem_img = self.vae.decoder(x_sem_hat).clip(min=0, max=1)
        clip_embeds_hat = self.clip(sem_img)
        clip_loss = F.cosine_similarity(clip_embed, clip_embeds_hat).mean()
        recon_loss = F.mse_loss(x, x_hat)
        sem_sparsity_loss = sem.abs().sum() / (sem.shape[0] * sem.shape[1])
        tex_kl_loss = (-0.5 * torch.sum(1 + tex_log_var - tex_mu**2 - tex_log_var.exp(), dim=1)).mean()

        return {
            'recon': recon_loss,
            'clip': clip_loss,
            'sparsity': sem_sparsity_loss,
            'kl': tex_kl_loss
        }, sem_img
    
    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor]):
        x, clip_embeds = batch
        x = x.to(memory_format=torch.channels_last)
        sem, tex_mu, tex_log_var = self.encode(x)
        tex = self.sample_vae(tex_mu, tex_log_var)
        x_hat, x_sem_hat = self.decode(sem, tex)

        losses, _ = self.calc_losses(x, x_hat, x_sem_hat, clip_embeds, sem, tex_mu, tex_log_var)
        self.log_dict(losses)
        loss = -10*losses['clip'] + losses['recon'] + 0.01*losses['sparsity'] + 2*losses['kl']
        self.log('loss', loss, prog_bar=True)
        return loss
    
    def decode_top_vae_latent(self, x: torch.Tensor) -> PIL.Image.Image:
        x = x[0].unsqueeze(0)
        img = self.vae.decoder(x).float().clip(min=0, max=1)
        return Fv2.to_pil_image(img[0])
    
    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        x, clip_embeds = batch
        sem, tex_mu, tex_log_var = self.encode(x)
        tex = self.sample_vae(tex_mu, tex_log_var)
        x_hat, x_sem_hat = self.decode(sem, tex)

        losses, sem_img = self.calc_losses(x, x_hat, x_sem_hat, clip_embeds, sem, tex_mu, tex_log_var)
        self.log_dict(losses)
        if batch_idx == 0:
            self.logger.log_image(key='original', images=[self.decode_top_vae_latent(x)])
            self.logger.log_image(key='recon_img ', images=[self.decode_top_vae_latent(x_hat)])
            self.logger.log_image(key='semantic', images=[Fv2.to_pil_image(sem_img[0, :].float())])

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)