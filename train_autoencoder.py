from dotenv import load_dotenv
import lightning as L
from lightning import Trainer
from lightning.pytorch.loggers import WandbLogger
import torch
from sketch_autoencoder.sketch_autoencoder import SketchAutoencoder
from sketch_autoencoder.ade20k_datamodule import Ade20kDatamodule
from sketch_autoencoder.clip_wrapper import CLIPWrapper
from taesd.taesd import TAESD
from pathlib import Path

# Train the model ⚡
if __name__ == "__main__":
    load_dotenv()
    torch.set_float32_matmul_precision('medium')
    torch.backends.cudnn.benchmark = True

    vae = TAESD('taesd/taesdxl_encoder.pth', 'taesd/taesdxl_decoder.pth')
    vae.eval()
    vae = vae.to('cuda')
    clip = CLIPWrapper('convnext_base_w', pretrained='laion2b_s13b_b82k')
    clip.eval()
    clip = clip.to('cuda')

    model = SketchAutoencoder(16, 4, 64, 8, 4, vae, clip).to(memory_format=torch.channels_last)
    data = Ade20kDatamodule(Path('./dataset/'), batch_size=32, num_workers=6)

    # Initialize a trainer
    logger = WandbLogger(project="sketch_autoencoder", log_model=True, save_dir="./.checkpoints/")
    logger.watch(model)
    trainer = L.Trainer(
        max_epochs=50,
        logger=logger,
        precision='bf16-mixed'
    )
    trainer.fit(model, datamodule=data)