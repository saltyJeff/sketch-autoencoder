import torch
from pathlib import Path
from itertools import islice
import torchvision.transforms.v2 as v2
from tqdm import tqdm
from taesd.taesd import TAESD
from PIL import Image
import pandas as pd
import open_clip
import io

INPUT_PATH = Path("D:/jeffe/Downloads/imagenet_1k_resized_256/data/")
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
    file_iter = filter(lambda p: p.name.startswith(split), INPUT_PATH.iterdir())
    file_iter = tqdm(list(file_iter))
    OUTPUT_SPLIT_PATH = OUTPUT_PATH / split
    OUTPUT_SPLIT_PATH.mkdir(exist_ok=True)
    i = 0
    for file in file_iter:
        df = pd.read_parquet(file)
        for img in tqdm(df['image']):
            img = Image.open(io.BytesIO(img['bytes']))
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img = TRANSFORMS(img)
            vae_img = VAE_TRANSFORMS(img).to('cuda')
            z = vae.encoder(vae_img).clip(min=-vae.latent_magnitude, max=vae.latent_magnitude)
            clip_img = preprocess(img).to('cuda', dtype=torch.bfloat16).unsqueeze(0)
            e = clip.encode_image(clip_img).squeeze(0)

            torch.save(z, (OUTPUT_SPLIT_PATH / f'{i:07d}.vae.pt'))
            torch.save(e, (OUTPUT_SPLIT_PATH / f'{i:07d}.clip.pt'))
            i = i+1

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        for split in ['train', 'test', 'val']:
            convert_split(split)
