""" Retrieving examples of wrong classifications for concept counterfactual analysis. """
import os
import sys
import yaml
import json
import hydra
import numpy as np
from omegaconf import DictConfig
import random
import torch
import torchvision.transforms.functional as tf

sys.path.append("./")
sys.path.append("./ldce")
from src.concept_conditioning import compute_concept_conditioning
from src.helpers.data_model_helpers import get_classifier, get_dataset, set_seed
from ldce.data.imagenet_classnames import name_map
from ldce.sampling_helpers import normalize


@hydra.main(version_base=None, config_path="../../configs/ldce", config_name="v1")
def main(cfg : DictConfig) -> None:

    if "ImageNet" in cfg.data._target_:
        out_dir = os.path.join(cfg.output_dir, f"bucket_{cfg.data.start_sample}_{cfg.data.end_sample}")
    else:
        out_dir = os.path.join(cfg.output_dir, f"bucket_{cfg.data.shard}_{cfg.data.num_shards}")
    os.makedirs(out_dir, exist_ok=True)
    os.chmod(out_dir, 0o777)
    checkpoint_path = os.path.join(out_dir, "last_saved_id.pth")

    last_data_idx = 0
    if cfg.resume: # or os.path.isfile(checkpoint_path): resume only if asked to, allow restarts
        print(f"resuming from {checkpoint_path}")
        #check if checkpoint exists
        if not os.path.exists(checkpoint_path):
            print("checkpoint does not exist! starting from 0 ...")
        else:
            checkpoint = torch.load(checkpoint_path)# torch.load(restored_file.name)
            last_data_idx = checkpoint["last_data_idx"] + 1 if "last_data_idx" in checkpoint else 0
        print(f"resuming from batch {last_data_idx}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu") # there seems to be a CUDA/autograd instability in gradient computation
    print(f"using device: {device}")

    classifier_model = get_classifier(cfg, device)
    classifier_model.to(device).eval()
    # classifier_model.train = disabled_train

    batch_size = cfg.data.batch_size
    shuffle = cfg.get("shuffle", False)
    
    #data_path = cfg.data_path
    dataset = get_dataset(cfg, last_data_idx=last_data_idx)
    dataset = torch.utils.data.Subset(dataset, np.arange(1000))
    print(type(dataset))
    print("dataset length: ", len(dataset))
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)

    if "ImageNet" in cfg.data._target_:
        i2h = name_map
    # elif "CelebAHQDataset" in cfg.data._target_:
    #     # query label 31 (smile): label=0 <-> no smile and label=1 <-> smile
    #     # query label 39 (age): label=0 <-> old and label=1 <-> young
    #     assert cfg.data.query_label in [31, 39]
    #     if 31 == cfg.data.query_label:
    #         i2h = ["no smile", "smile"]
    #     elif 39 == cfg.data.query_label:
    #         i2h = ["old", "young"]
    #     else:
    #         raise NotImplementedError
    # elif "Flowers102" in cfg.data._target_:
    #     with open("data/flowers_idx_to_label.json", "r") as f:
    #         flowers_idx_to_classname = json.load(f)
    #     flowers_idx_to_classname = {int(k)-1: v for k, v in flowers_idx_to_classname.items()}
    #     i2h = flowers_idx_to_classname
    # elif "OxfordIIIPets" in cfg.data._target_:
    #     with open("data/pets_idx_to_label.json", "r") as f:
    #         pets_idx_to_classname = json.load(f)
    #     i2h = {int(k): v for k, v in pets_idx_to_classname.items()}
    else:
        raise NotImplementedError

    if "ImageNet" in cfg.data._target_:
        with open('data/synset_closest_idx.yaml', 'r') as file:
            synset_closest_idx = yaml.safe_load(file)
    # elif "Flowers102" in cfg.data._target_:
    #     with open("data/flowers_closest_indices.json") as file:
    #         closest_indices = json.load(file)
    #     closest_indices = {int(k):v for k,v in closest_indices.items()}
    # elif "OxfordIIIPets" in cfg.data._target_:
    #     with open("data/pets_closest_indices.json") as file:
    #         closest_indices = json.load(file)
    #     closest_indices = {int(k):v for k,v in closest_indices.items()}

    concept_layer = cfg.concept_layer       # "backbone.features.29"
    spatial = cfg.spatial

    if not cfg.resume:
        torch.save({"last_data_idx": -1}, checkpoint_path)
    
    seed = cfg.seed if "seed" in cfg else 0
    set_seed(seed=seed)

    wrong_prediction_counter = 0
    cf_correct_counter = 0

    for i, batch in enumerate(data_loader):

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
            elif "CelebAHQDataset" in cfg.data._target_:
                tgt_classes = (1 - label).type(torch.float32)
            elif "Flowers102" in cfg.data._target_ or "OxfordIIIPets" in cfg.data._target_:
                tgt_classes = torch.tensor([closest_indices[unique_data_idx[l].item()*cfg.data.num_shards + cfg.data.shard][0] for l in range(label.shape[0])]).to(device)
            else:
                raise NotImplementedError

        if "counterfactual_target" in cfg:
            if cfg.counterfactual_target == "baseline":
                tgt_json = os.path.join('/results/counterfactuals/', 'concept_selection', f'conditions_{cfg.classifier_model.name}_{cfg.counterfactual_target}.json')
            elif cfg.target_norm:
                tgt_json = os.path.join('/results/counterfactuals/', 'concept_selection', f'conditions_{cfg.classifier_model.name}_{cfg.concept_layer}_{cfg.counterfactual_target}_norm.json')
            else:
                tgt_json = os.path.join('/results/counterfactuals/', 'concept_selection', f'conditions_{cfg.classifier_model.name}_{cfg.concept_layer}_{cfg.counterfactual_target}.json')
            with open(tgt_json) as f:
                d = json.load(f)

            tgt_classes = [d[str(uix.item())]['target'] for uix in unique_data_idx]
            tgt_classes = torch.tensor(tgt_classes, dtype=torch.int64).to(device)

        image = image.to(device) #squeeze()
        label = label.to(device) #.item() #squeeze()

        # Compute concept conditions
        # ToDo: add sampler.classifier_wrapper as parameter
        # if spatial:
        #     conditions, concept_conds, concept_diff = compute_concept_conditioning(classifier_model, image, tgt_classes, concept_layer, num_concepts=cfg.num_concepts, spatial=spatial, cond_option=cfg.cond_option)
        # else:
        #     conditions, concept_conds, concept_diff = compute_concept_conditioning(classifier_model, image, tgt_classes, concept_layer, num_concepts=cfg.num_concepts, cond_option=cfg.cond_option)

        #get classifcation prediction
        with torch.inference_mode():
            #with precision_scope():
            if "classifier_wrapper" in cfg.classifier_model and cfg.classifier_model.classifier_wrapper:
                logits = classifier_model(image)
            else:
                # logits = sampler.get_classifier_logits(_unmap_img(image)) #converting to -1, 1
                # x = _map_img(x)
                x = image
                if "classifier_wrapper" not in cfg.classifier_model:# and not cfg.classifier_model.classifier_wrapper: # only works for ImageNet!
                    # print('cropping and normalizing')
                    x = tf.center_crop(x, 224)
                    x = normalize(x)
                logits = classifier_model(x)
            # TODO: handle binary vs multi-class
            if "ImageNet" in cfg.data._target_ or "OxfordIIIPets" in cfg.data._target_ or "Flowers102" in cfg.data._target_: # multi-class
                in_class_pred = logits.argmax(dim=1)
                in_confid = logits.softmax(dim=1).max(dim=1).values
                in_confid_tgt =  logits.softmax(dim=1)[torch.arange(batch_size), tgt_classes]
            else: # binary
                in_class_pred = (logits >= 0).type(torch.int8)
                in_confid = torch.where(logits >= 0, logits.sigmoid(), 1 - logits.sigmoid())
                in_confid_tgt =  torch.where(tgt_classes.to(device) == 0, 1 - logits.sigmoid(), logits.sigmoid())
            # print("in class_pred: ", in_class_pred, in_confid)
        
        # for j, l in enumerate(label):
        #     print(f"converting {i} from : {i2h[l.item()]} to: {i2h[int(tgt_classes[j].item())]}")

        # Is original class correctly predicted
        miss = torch.where(in_class_pred != label)[0]

        if miss.numel():
            for miss_val in miss:
                # print(miss_val.item() + i*batch_size)
                print(f"converting {miss_val.item() + i*batch_size} from wrong: {i2h[in_class_pred[miss_val].item()]} to right: {i2h[int(tgt_classes[miss_val].item())]} and label {i2h[int(label[miss_val].item())]}")
                
                wrong_prediction_counter += 1

                if int(tgt_classes[miss_val].item()) == int(label[miss_val].item()):
                    cf_correct_counter +=1
            # raise ValueError
        # print(torch.where(in_class_pred != label)[0])
        # print(f'Pred: {in_class_pred}')
        # print(f'Annotation: {label}')

        # If yes: Is counterfactual class == original class?
        # if miss.numel() and (tgt_classes[miss] == label[miss]).any():
        #     for miss_val in miss:
        #         print(f'Target: {tgt_classes[miss_val]} --> {label[miss_val]} and pred {in_class_pred[miss_val]}')
        #         print(f"converting {miss_val} from wrong: {i2h[in_class_pred[miss_val].item()]} to right: {i2h[int(tgt_classes[miss_val].item())]} and label {i2h[int(label[miss_val].item())]}")
        #     # raise ValueError
        

        # raise ValueError
    
    print(f'Wrong predictions: {wrong_prediction_counter}')
    print(f'Correctly chosen cf target: {cf_correct_counter}')

if __name__ == '__main__':
    main()
