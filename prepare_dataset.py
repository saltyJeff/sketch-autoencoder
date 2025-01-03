import torch
from pathlib import Path
from itertools import islice
import torchvision.transforms.v2 as v2
from tqdm import tqdm
from taesd.taesd import TAESD
from PIL import Image
import sys
import tarfile
import io
import open_clip

INPUT_PATH = Path("./ade20k-DatasetNinja/")
OUTPUT_PATH = Path('./dataset/')
vae = TAESD('taesd/taesdxl_encoder.pth', 'taesd/taesdxl_decoder.pth')
vae.eval()
vae = vae.to('cuda')

clip, _, preprocess = open_clip.create_model_and_transforms('convnext_base_w', pretrained='laion2b_s13b_b82k')
clip.eval()
clip = clip.to('cuda')

TRANSFORMS = v2.Compose([
    v2.Resize(256),
    v2.CenterCrop(256),
])
VAE_TRANSFORMS = v2.Compose([
    v2.PILToTensor(),
    v2.ToDtype(torch.float, scale=True)
])

@torch.no_grad()
def convert_split(split: str):
    img_iter = tqdm(list((INPUT_PATH / split / 'img').iterdir()))
    OUTPUT_SPLIT_PATH = OUTPUT_PATH / split
    OUTPUT_SPLIT_PATH.mkdir(exist_ok=True)
    for img_path in img_iter:
        img = Image.open(img_path)
        # convert grayscales
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = TRANSFORMS(img)
        vae_img = VAE_TRANSFORMS(img).to('cuda')
        z = vae.encoder(vae_img).clip(min=-vae.latent_magnitude, max=vae.latent_magnitude)
        clip_img = preprocess(img).to('cuda', dtype=torch.bfloat16).unsqueeze(0)
        embed = clip.encode_image(clip_img).squeeze(0)

        torch.save(z, (OUTPUT_SPLIT_PATH / img_path.with_suffix('.vae.pt').name))
        torch.save(embed, (OUTPUT_SPLIT_PATH / img_path.with_suffix('.clip.pt').name))

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        for split in ['training', 'validation']:
            convert_split(split)
