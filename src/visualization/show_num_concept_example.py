import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

num_concepts = [1, 10, 20, 50, 100, 200, 300]

dir = '/results/counterfactuals/'

model = 'vgg16bn_40'

img_ids = range(0, 50)

for img_id in img_ids:

    fig, ax = plt.subplots(1, len(num_concepts), figsize=(21, 4))

    for i, n_concept in enumerate(num_concepts):

        dir_path = os.path.join(dir, f'imagenet_{model}_concept_{n_concept}')
        
        f_path = os.path.join(dir_path, 'bucket_0_10', 'counterfactual', f'{str(img_id).zfill(5)}.png')
        img = Image.open(f_path)

        ax[i].imshow(img)
        ax[i].axis('off')

    plt.tight_layout()
    plt.savefig(f'/results/counterfactuals/num_concepts/{model}_{str(img_id).zfill(5)}.svg')
    plt.close()
