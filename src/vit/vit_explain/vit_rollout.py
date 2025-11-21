import torch
from PIL import Image
import numpy
import sys
from torchvision import transforms
import numpy as np
import cv2

def rollout(attentions, discard_ratio, head_fusion):

    attentions = torch.stack([att[0] for att in attentions])
    print(attentions.size())

    n_samples = attentions.size(1)
    masks = []

    for sid in range(n_samples):
        result = torch.eye(attentions[0].size(1))
        with torch.no_grad():
            for attention in attentions:
                if head_fusion == "mean":
                    attention_heads_fused = attention[sid].mean(axis=1).cpu()      #(axis=1)
                elif head_fusion == "max":
                    attention_heads_fused = attention[sid].max(axis=1)[0].cpu()
                elif head_fusion == "min":
                    attention_heads_fused = attention.min(axis=1)[0]
                else:
                    raise "Attention head fusion type Not supported"

                # Drop the lowest attentions, but
                # don't drop the class token
                flat = attention_heads_fused
                _, indices = flat.topk(int(flat.size(0)*discard_ratio), -1, False)
                # flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
                # _, indices = flat.topk(int(flat.size(-1)*discard_ratio), -1, False)
                # print(indices)
                indices = indices[indices != 0]
                # flat[0, indices] = 0
                flat[indices] = 0

                print(f'Flat size: {flat.size()}')

                print(f'Att head fused: {attention_heads_fused.size()}')

                I = torch.eye(attention_heads_fused.size(-1))
                a = (attention_heads_fused + 1.0*I)/2
                a = a / a.sum(dim=-1)

                result = torch.matmul(a, result)
        
        print(result.size())
        # Look at the total attention between the class token,
        # and the image patches
        # mask = result[0, 0 , 1 :]
        mask = result[0, 1:]
        # In case of 224x224 image, this brings us from 196 to 14
        width = int(mask.size(-1)**0.5)
        mask = mask.reshape(width, width).numpy()
        mask = mask / np.max(mask)

        masks.append(mask)
    return masks

class VITAttentionRollout:
    def __init__(self, model, attention_layer_name='attn_drop', head_fusion="mean",
        discard_ratio=0.9):
        self.model = model
        self.head_fusion = head_fusion
        self.discard_ratio = discard_ratio
        for name, module in self.model.named_modules():
            if attention_layer_name in name:
                module.register_forward_hook(self.get_attention)

        self.attentions = []

    def get_attention(self, module, input, output):
        self.attentions.append(output)      #.cpu())

    def __call__(self, input_tensor):
        self.attentions = []
        with torch.no_grad():
            output = self.model(input_tensor)

        print(f'Attention layers: {len(self.attentions)}')

        return rollout(self.attentions, self.discard_ratio, self.head_fusion)