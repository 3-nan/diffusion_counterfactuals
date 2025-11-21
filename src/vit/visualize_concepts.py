import os
import sys
sys.path.append("./")
sys.path.append("./ldce")
sys.path.append("./data")
import hydra
import numpy as np
from omegaconf import DictConfig
import random
import torch
from torchvision.models import vit_b_16
import torchvision.transforms.functional as tf
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2

from src.vit.vit_explain.vit_rollout import VITAttentionRollout
from src.vit.vit_explain.vit_grad_rollout import VITAttentionGradRollout
from ldce.sampling_helpers import normalize
from src.helpers.data_model_helpers import get_classifier, get_dataset


def set_seed(seed: int = 0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)


@hydra.main(version_base=None, config_path="../../configs/ldce", config_name="v1")
def main(cfg : DictConfig) -> None:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # classifier_model = vit_b_16(weights='IMAGENET1K_V1')
    classifier_model = vit_b_16(pretrained=True)

    classifier_model = classifier_model.to(device)
    classifier_model.eval()

    batch_size = 16 # cfg.data.batch_size

    print(f'{cfg.data.start_sample} -> {cfg.data.end_sample}')
    last_data_idx = 0
    dataset = get_dataset(cfg, last_data_idx=last_data_idx, base=False)
    print(type(dataset))
    print("dataset length: ", len(dataset))
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=1)

    # Iterate dataset (max 100 samples per class?)

    for i, batch in enumerate(tqdm(data_loader)):

        if "fixed_seed" in cfg:
            set_seed(seed=cfg.get("seed", 0)) if cfg.fixed_seed else None
            seed = seed if cfg.fixed_seed else -1
            
        if "return_tgt_cls" in cfg.data and cfg.data.return_tgt_cls:
            image, label, tgt_classes, unique_data_idx = batch
            tgt_classes = tgt_classes.to(device) #squeeze()
        else:
            image, label, unique_data_idx = batch
            if "ImageNet" in cfg.data._target_:
                tgt_classes = torch.tensor([random.choice(synset_closest_idx[l.item()]) for l in label]).to(device)
            # elif "CelebAHQDataset" in cfg.data._target_:
            #     tgt_classes = (1 - label).type(torch.float32)
            # elif "Flowers102" in cfg.data._target_ or "OxfordIIIPets" in cfg.data._target_:
            #     tgt_classes = torch.tensor([closest_indices[unique_data_idx[l].item()*cfg.data.num_shards + cfg.data.shard][0] for l in range(label.shape[0])]).to(device)
            else:
                raise NotImplementedError

        # # Load generated counterfactual images
        # file_paths = [os.path.join(counterfactuals_dir, f'{str(udx).zfill(5)}.png') for udx in unique_data_idx.numpy()]


        # cf_imgs = [dataset.transform(pil_loader(fp)) for fp in file_paths]
        # cf_imgs = torch.stack(cf_imgs, dim=0)

        # # encode counterfactuals
        # image = cf_imgs
        # label = tgt_classes

        # raise ValueError
        image = image.to(device)
        label = label.to(device)

        #get classifcation prediction
        with torch.inference_mode():
            #with precision_scope():
            if "classifier_wrapper" in cfg.classifier_model and cfg.classifier_model.classifier_wrapper:
                logits = classifier_model(image)
            else:
                # logits = sampler.get_classifier_logits(_unmap_img(image)) #converting to -1, 1
                # x = _map_img(x)
                x = image
                if "classifier_wrapper" not in cfg.classifier_model: # only works for ImageNet!
                    x = tf.center_crop(x, 224)
                    x = normalize(x)
                logits = classifier_model(x)
            # TODO: handle binary vs multi-class
            if "ImageNet" in cfg.data._target_ or "OxfordIIIPets" in cfg.data._target_ or "Flowers102" in cfg.data._target_: # multi-class
                in_class_pred = logits.argmax(dim=1)
                in_confid = logits.softmax(dim=1).max(dim=1).values
                # in_confid_tgt =  logits.softmax(dim=1)[torch.arange(batch_size), tgt_classes]
            else: # binary
                in_class_pred = (logits >= 0).type(torch.int8)
                in_confid = torch.where(logits >= 0, logits.sigmoid(), 1 - logits.sigmoid())

        for n,m in classifier_model.named_modules():
            print(f'{n} : {type(m)}')
        
        # raise ValueError

        print('Before attention rollout')
        x_clone = x.clone().to(device)

        # grad_rollout = VITAttentionRollout(classifier_model, discard_ratio=0.2, attention_layer_name='self_attention', head_fusion='max')
        # masks = grad_rollout(x_clone)
    
        grad_rollout = VITAttentionGradRollout(classifier_model, discard_ratio=0.8, attention_layer_name=r'layer_[0-9][0-9]?$') #, head_fusion='max')
        # masks = grad_rollout(x_clone, category_index=tgt_classes)
        # masks = grad_rollout(x_clone, category_index=in_class_pred)
        masks = grad_rollout(x_clone, category_index=label)

        print(f'Rollout mask: {masks[0].shape}')

        for m, mask in enumerate(masks):

            mask = cv2.resize(mask, (256, 256))
            heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
            # heatmap = np.float32(heatmap) / 255
            plt.figure()
            plt.imshow(heatmap)
            plt.axis('off')
            # plt.savefig(os.path.join(cfg.output_dir, f'attention_mask_{m}.png'))
            plt.savefig(os.path.join(cfg.output_dir, f'attention_grad_mask_{m}.png'))
            plt.close()

        raise ValueError

if __name__ == '__main__':
    main()
