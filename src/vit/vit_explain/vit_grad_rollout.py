import torch
from PIL import Image
import re
import numpy
import sys
from torchvision import transforms
import numpy as np
import cv2

def grad_rollout(attentions, gradients, discard_ratio):

    attentions = torch.stack([att for att in attentions])
    print(attentions.size())

    n_samples = attentions.size(1)
    masks = []

    for sid in range(n_samples):
    # index = 0
    # print(attentions[0][index].size())
        result = torch.eye(attentions[0].size(1))

        print(f'result size: {result.size()}')
        with torch.no_grad():
            for attention, grad in zip(attentions, gradients):

                # weights = grad
                # print(f'{attention[sid].size()} : {weights.size()}')
        
                attention = attention[sid].cpu()
                # weights = grad[:, sid, :].cpu()
                weights = grad[sid].cpu()

                # weights = weights.clip(min=0.)
                # print(torch.max(weights, axis=1).size())
                # weights = weights / torch.max(weights, dim=1)[0]

                # print(f'{attention.size()} : {weights.size()}')

                # attention_heads_fused = (attention*weights).mean(axis=1)
                attention_heads_fused = (attention*weights).max(axis=1)[0]
                attention_heads_fused[attention_heads_fused < 0] = 0

                # Drop the lowest attentions, but
                # don't drop the class token
                flat = attention_heads_fused
                _, indices = flat.topk(int(flat.size(0)*discard_ratio), -1, False)
                # flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
                # _, indices = flat.topk(int(flat.size(-1)*discard_ratio), -1, False)
                # flat[0, indices] = 0
                indices = indices[indices != 0]
                # indices = indices.prepend(0)
                flat[indices] = 0

                # print(f'Att head fused: {attention_heads_fused.size()}')

                I = torch.eye(attention_heads_fused.size(-1))
                a = (attention_heads_fused + 1.0*I)/2
                a = a / a.sum(dim=-1)
                result = torch.matmul(a, result)
        
        # Look at the total attention between the class token,
        # and the image patches
        mask = result[0 , 1 :]
        # In case of 224x224 image, this brings us from 196 to 14
        width = int(mask.size(-1)**0.5)
        mask = mask.reshape(width, width).numpy()
        mask = mask / np.max(mask)

        masks.append(mask)
    return masks

class VITAttentionGradRollout:
    def __init__(self, model, attention_layer_name='attn_drop', discard_ratio=0.9):
        self.model = model
        self.discard_ratio = discard_ratio
        for name, module in self.model.named_modules():
            # if attention_layer_name in name: # and 'out_proj' not in name:
            prog = re.compile(attention_layer_name)
            res = prog.search(name)
            # if name.endswith(attention_layer_name):
            if res:
                print(name)
                module.register_forward_hook(self.get_attention)
                module.register_backward_hook(self.get_attention_gradient)

        self.attentions = []
        self.attention_gradients = []

    def get_attention(self, module, input, output):
        # self.attentions.append(output.cpu())
        # print(f'input : {input}')
        self.attentions.append(output)

    def get_attention_gradient(self, module, grad_input, grad_output):
        # self.attention_gradients.append(grad_input[0].cpu())
        # print(grad_input)
        # print(grad_output)
        # print(module.input)
        print(f'{grad_input[0].size()} : {grad_output[0].size()}')
        # raise ValueError
        self.attention_gradients.append(grad_input[0])

    def __call__(self, input_tensor, category_index):
        self.model.zero_grad()
        output = self.model(input_tensor)
        # category_mask = torch.zeros(output.size())
        category_mask = torch.zeros_like(output)
        category_mask[:, category_index] = 1
        loss = (output*category_mask).sum()
        loss.backward()

        print(f'Attentions size: {len(self.attentions)}')

        return grad_rollout(self.attentions, self.attention_gradients,
            self.discard_ratio)