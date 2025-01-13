from dotenv import load_dotenv
import lightning as L
from lightning import Trainer
from lightning.pytorch.loggers import WandbLogger
import torch
from sketch_autoencoder.sketch_autoencoder import SketchAutoencoder
from sketch_autoencoder.coco_datamodule import CocoDatamodule
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

    vae = TAESD(TAESD_ROOT / 'taesdxl_encoder.pth', TAESD_ROOT / 'taesdxl_decoder.pth')
    vae.eval()
    vae.requires_grad_(False)
    vae = vae.to('cuda')

    model = SketchAutoencoder(4, vae, 512, 
                              2, 
                              4)
    data = CocoDatamodule(vae, batch_size=64, num_workers=6)

    # Initialize a trainer
    logger = WandbLogger(project="sketch_autoencoder", log_model=True, save_dir="./.checkpoints/")
    logger.watch(model)
    trainer = L.Trainer(
        max_epochs=5,
        logger=logger,
        precision='bf16-mixed' 
    )
    trainer.fit(model, datamodule=data)
    Q, R = torch.linalg.qr(model.z_transform.weight)
    model.z_transform.weight = torch.nn.Parameter(Q, requires_grad=False)
    print("R from QR decomp", R)
    torch.save(model.z_transform.state_dict(), '.checkpoints/z_transform.pth')