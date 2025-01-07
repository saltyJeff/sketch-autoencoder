from dotenv import load_dotenv
import lightning as L
from lightning import Trainer
from lightning.pytorch.loggers import WandbLogger
import torch
from sketch_autoencoder.sketch_autoencoder import SketchAutoencoder
from sketch_autoencoder.coco_datamodule import CocoDatamodule
from sketch_autoencoder.clip_wrapper import CLIPWrapper
from taesd.taesd import TAESD
from pathlib import Path
import open_clip
import taesd.taesd

# Train the model ⚡
if __name__ == "__main__":
    load_dotenv()
    torch.set_float32_matmul_precision('medium')
    torch.backends.cudnn.benchmark = True 

    TAESD_ROOT = Path(taesd.taesd.__file__).parent

    vae = TAESD(TAESD_ROOT / 'taesdxl_encoder.pth', TAESD_ROOT / 'taesdxl_decoder.pth').bfloat16()
    vae.eval()
    vae.requires_grad_(False)
    vae = vae.to('cuda')

    clip, _, clip_preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k', precision='bf16')
    clip_tokenizer = open_clip.get_tokenizer('ViT-B-32')
    clip.eval()
    clip.requires_grad_(False)
    clip = clip.to('cuda')

    model = SketchAutoencoder(4, vae, 640, 
                              1, 
                              4)
    data = CocoDatamodule(vae, clip, clip_preprocess, clip_tokenizer, batch_size=128, num_workers=6)

    # Initialize a trainer
    logger = WandbLogger(project="sketch_autoencoder", log_model=True, save_dir="./.checkpoints/")
    logger.watch(model)
    trainer = L.Trainer(
        max_epochs=2,
        logger=logger,
        precision='bf16-mixed'
    )
    trainer.fit(model, datamodule=data)
    torch.save(model.reencoder.weight, 'reencoder.pth')