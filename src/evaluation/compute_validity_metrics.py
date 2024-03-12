import os
import torch
import torchvision
import argparse
import numpy as np
import sys
sys.path.append("./")
sys.path.append("./ldce")

from PIL import Image
from tqdm import tqdm
import glob
from torch.utils import data
import yaml

from data.imagenet_classnames import name_map
from ldce.sampling_helpers import disabled_train

class CFDataset():
    def __init__(self, path, idx_to_tgt):

        self.images = []
        self.path = path
        for bucket_folder in sorted(glob.glob(self.path + "/bucket*")):
            for original, counterfactual in zip(sorted(glob.glob(bucket_folder + "/original/*.png")), sorted(glob.glob(bucket_folder + "/counterfactual/*.png"))):
                self.images.append((original, counterfactual, os.path.join(bucket_folder, os.path.basename(original).replace("png", "pth"))))

        imagenet_mean = (0.485, 0.456, 0.406)
        iamgenet_std = (0.229, 0.224, 0.225)
        self.transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((256, 256)),
                torchvision.transforms.CenterCrop((224, 224)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=imagenet_mean, std=iamgenet_std),
            ]
        )

        with open(idx_to_tgt, 'r') as file:
            idx_to_tgt_cls = yaml.safe_load(file)
            if isinstance(idx_to_tgt_cls, dict):
                idx_to_tgt_cls = [idx_to_tgt_cls[i] for i in range(len(idx_to_tgt_cls))]
        
        self.idx_to_tgt_cls = idx_to_tgt_cls
        idx_to_tgt_cls = []
        for idx in range(50000//1000):
            idx_to_tgt_cls.extend(self.idx_to_tgt_cls[idx::50000//1000])
        self.idx_to_tgt_cls = idx_to_tgt_cls

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        original_path, counterfactual_path, pth_file = self.images[idx]

        original = self.load_img(original_path)
        counterfactual = self.load_img(counterfactual_path)
        # data = torch.load(pth_file, map_location="cpu")
        # counterfactual = data["gen_image"]

        return original, idx%1000, counterfactual, self.idx_to_tgt_cls[idx]

    def load_img(self, path):
        img = Image.open(path).convert("RGB")
        return self.transform(img)

@torch.inference_mode()
def compute_validity_metrics(args):
    device = torch.device("cuda")

    n_samples, correct = 0, 0
    confidence = []
    for pth_file in tqdm(sorted(glob.glob(args.output_path + "/bucket*/*.pth")), leave=False):
        data = torch.load(pth_file, map_location="cpu")
        # print(data.keys())
        # print(data["target_confidence"])
        # print(data["target"])
        # print(data["out_pred"])
        # print(data["out_confid"])
        # raise ValueError
        if "target" in data and "out_pred" in data:
            correct += (data["target"] == data["out_pred"])
            n_samples += 1
            confidence.append(data["out_confid"])
        # if "target" in data and "out_probability" in data:
        #     correct += (data["target"]==torch.argmax(data["out_probability"])).int().item()
        #     n_samples += 1
        #     confidence.append(data["out_probability"][data["target"].item()].item())

    print(f'{n_samples} samples processed.')
    return correct/n_samples, np.mean(confidence)

    # dataset = CFDataset(args.output_path, args.idx_to_tgt)
    # loader = data.DataLoader(dataset, batch_size=args.batch_size,
    #                             shuffle=False,
    #                             num_workers=16, pin_memory=True)

    # classifier_model = getattr(torchvision.models, args.target_model)(pretrained=True).to(device)
    # classifier_model = classifier_model.eval()
    # classifier_model.train = disabled_train
    
    # flipped, confidences = [], []
    # softmax = torch.nn.Softmax(dim=-1)
    # for original, original_label, counterfactual, counterfactual_label in tqdm(loader, leave=False):
    #     logits = classifier_model(counterfactual.to(device))
    #     preds = torch.argmax(logits, dim=1).cpu()
    #     softmax_logits = softmax(logits)
    #     flipped += list((preds == counterfactual_label).type(torch.uint8).cpu().numpy())
    #     confidences += list(softmax_logits[range(counterfactual_label.shape[0]), counterfactual_label].cpu().numpy())

    # return np.sum(flipped)/len(dataset), np.mean(confidences)

def arguments():
    parser = argparse.ArgumentParser(description="Validity metrics evaluation")
    parser.add_argument('--output-path', required=True, type=str)
    parser.add_argument('--target-model', required=True, type=str, default="resnet50")
    parser.add_argument('--batch_size', required=False, type=int, default=32)
    return parser.parse_args()

if __name__ == '__main__':
    args = arguments()
    
    flip_ratio, confidence = compute_validity_metrics(args)

    print(f"Flip ratio: {flip_ratio}")
    print(f"Confidence: {confidence}")
