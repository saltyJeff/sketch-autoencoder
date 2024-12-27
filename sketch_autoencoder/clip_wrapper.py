import torch
import torch.nn as nn
import open_clip
import torchvision.transforms.v2 as v2
from enum import Enum

class CLIPWrapper(nn.Module):
    def __init__(self, model_name: str, pretrained: str):
        super().__init__()
        clip, _ = open_clip.create_model_from_pretrained(model_name, pretrained, precision='bf16')
        self.clip: open_clip.CLIP = clip
        self.preprocess = v2.Normalize(open_clip.constants.OPENAI_DATASET_MEAN, open_clip.constants.OPENAI_DATASET_STD)
        # delete text portion of the CLIP model to save memory
        del self.clip.transformer
        del self.clip.token_embedding
        del self.clip.positional_embedding
    def forward(self, x: torch.Tensor):
        return self.clip.encode_image(self.preprocess(x))
