import torch
from pathlib import Path
import taesd.taesd
from taesd.taesd import TAESD
import torchvision.transforms.v2 as v2
from PIL import Image
from sketch_autoencoder.model_blocks import InvertibleLinear
import itertools

TRANSFORMS = v2.Compose([
    v2.PILToTensor(),
    v2.ToDtype(torch.float, scale=True),
])

IMG_PATH = Path(__file__).parent / 'img'
Z_PATH = Path(__file__).parent / '.checkpoints' / 'z_transform.pth'
LERP = 0.9

def lerp_dims(left: torch.Tensor, right: torch.Tensor, chans: tuple[int, int], alpha: float):
    left = left.clone()
    for chan in chans:
        left[:, chan, :, :] = torch.lerp(left[:, chan, :, :], right[:, chan, :, :], alpha)
    return left

# Train the model ⚡
if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')
    torch.backends.cudnn.benchmark = True 

    TAESD_ROOT = Path(taesd.taesd.__file__).parent

    vae = TAESD(TAESD_ROOT / 'taesdxl_encoder.pth', TAESD_ROOT / 'taesdxl_decoder.pth')
    vae.eval()
    vae.requires_grad_(False)
    vae = vae.to('cuda')

    LEFT = TRANSFORMS(Image.open(IMG_PATH / 'left.jpg')).unsqueeze(0).to('cuda')
    RIGHT = TRANSFORMS(Image.open(IMG_PATH / 'right.jpg')).unsqueeze(0).to('cuda')
    z_transform = InvertibleLinear(4)
    z_transform.load_state_dict(torch.load(Z_PATH, weights_only=True))
    z_transform.eval()
    z_transform = z_transform.to('cuda')
    z_transform.requires_grad_(False)

    with torch.autocast('cuda'):
        left_z = vae.encoder(LEFT)
        right_z = vae.encoder(RIGHT)

        def recon_and_save(z: torch.Tensor, name: str):
            orig = vae.decoder(z).clip(0, 1)
            z = z_transform.decode(z_transform.encode(z))
            out = vae.decoder(z).clip(0, 1)
            out_img = v2.functional.to_pil_image(out.squeeze(0))
            out_img.save(IMG_PATH / (name+'.jpg'))

            delta = (orig - out).abs() * 10
            out = v2.functional.to_pil_image(delta.clip(0, 1).squeeze(0))
            out.save(IMG_PATH / (name+'_delta.jpg'))
        
        recon_and_save(left_z, 'left_recon')
        recon_and_save(right_z, 'right_recon')

        def lerp_and_save(chans: tuple[int, int], use_z_transform: bool, alpha: float = LERP):
            if use_z_transform:
                left = z_transform.encode(left_z)
                right = z_transform.encode(right_z)
            else:
                left = left_z.clone()
                right = right_z.clone()
            out = lerp_dims(left, right, chans, alpha)

            if use_z_transform:
                out = z_transform.decode(out)
            out = vae.decoder(out)
            out = v2.functional.to_pil_image(out.clip(0, 1).squeeze(0))

            out_name = 'lerp'
            if use_z_transform:
                out_name += '-z'
            out_name += ('-' + ''.join(str(c) for c in chans))
            out_name += '.jpg'
            out.save(IMG_PATH / out_name)

        lerp_and_save((0, 1), True)
        lerp_and_save((2, 3), True)

        for chan_pair in itertools.combinations(range(4), 2):
            lerp_and_save(chan_pair, False)
        for chan_pair in itertools.combinations(range(4), 1):
            lerp_and_save(chan_pair, False)