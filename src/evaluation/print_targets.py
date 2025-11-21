import os
import torch

# result_dir = "/results/counterfactuals/imagenet_vgg16bn_baseline_attrtarget"
result_dir = "/results/counterfactuals/flowers_vgg16bn_37_concept_20_final"

for uidx in range(1000):

    pth_file = os.path.join(result_dir, 'bucket_0_1', f'{str(uidx).zfill(5)}.pth')
    
    data = torch.load(pth_file, map_location="cpu")

    # data["target"]
    # print(data.keys())

    print(f'Converting {uidx} from {data["source"]} to {data["target"]}')

    # raise ValueError
