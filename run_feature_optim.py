""" Extension of run_ldce to maximize single channels/neurons or arbitrary latent space representations. """
import argparse
import os
# import psutil
import yaml
import json
import copy
import random

# import matplotlib.pyplot as plt
import numpy as np
import pathlib
import sys
sys.path.append("./")
sys.path.append("./ldce")


import torch
torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.benchmark = True
# from contextlib import nullcontext
from torch import autocast

from omegaconf import OmegaConf, open_dict, DictConfig
import hydra
from hydra.utils import instantiate
# from omegaconf import DictConfig, OmegaConf
# import wandb
import torchvision
from torchvision import transforms, datasets
from torchvision.utils import save_image

# from src.clipseg.models.clipseg import CLIPDensePredT
# try:
#     from segment_anything import build_sam, SamPredictor
# except:
#     print("segment_anything not installed")
# from ldce.sampling_helpers import disabled_train, get_model, _unmap_img, generate_samples
from ldce.sampling_helpers import disabled_train, get_model, _unmap_img
# from ldce.sampling_helpers import load_model_hf
# import json


import sys
import regex as re
from ldce.ldm import *
# from src.cc_ddim import ConceptCCMDDIMSampler as CCMDDIMSampler
from src.dreamer.dreamer_cc_ddim import ConceptCCMDDIMSampler as CCMDDIMSampler

from ldce.data.imagenet_classnames import name_map, openai_imagenet_classes

from ldce.utils.preprocessor import Normalizer, CropAndNormalizer, ResizeAndNormalizer, GenericPreprocessing, Crop
from ldce.utils.madry_net import MadryNet

# from src.concept_extraction import compute_concept_conditioning
from src.dreamer.dreamer_sampling import generate_samples


def set_seed(seed: int = 0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)

def blockPrint():
    sys.stdout = open(os.devnull, 'w')

def get_classifier(cfg, device):
    if "ImageNet" in cfg.data._target_:
        classifier_name = cfg.classifier_model.name
        if classifier_name == "robust_resnet50":
            classifier_model = MadryNet(cfg.classifier_model.ckpt, device)
            if "classifier_wrapper" in cfg.classifier_model and cfg.classifier_model.classifier_wrapper:
                classifier_model = Crop(classifier_model)
        else:
            classifier_model = getattr(torchvision.models, classifier_name)(pretrained=True)
            if "classifier_wrapper" in cfg.classifier_model and cfg.classifier_model.classifier_wrapper:
                classifier_model = CropAndNormalizer(classifier_model)
    else:
        raise NotImplementedError
    return classifier_model

def get_dataset(cfg, last_data_idx: int = 0):
    if "ImageNet" in cfg.data._target_:
        out_size = 256
        transform_list = [
            transforms.Resize((out_size, out_size)),
            transforms.ToTensor()
        ]
        transform = transforms.Compose(transform_list)
        dataset = instantiate(cfg.data, start_sample=cfg.data.start_sample, end_sample=cfg.data.end_sample, transform=transform, restart_idx=last_data_idx)
    else:
        raise NotImplementedError
    return dataset

@hydra.main(version_base=None, config_path="configs/ldce", config_name="v1")
def main(cfg : DictConfig) -> None:
    if "verbose" not in cfg:
        with open_dict(cfg):
            cfg.verbose = True
    if "record_intermediate_results" not in cfg:
        with open_dict(cfg):
            cfg.record_intermediate_results = True

    if "verbose" in cfg and not cfg.verbose:
        blockPrint()

    os.makedirs(cfg.output_dir, exist_ok=True)
    os.chmod(cfg.output_dir, 0o777)
    if "ImageNet" in cfg.data._target_:
        out_dir = os.path.join(cfg.output_dir, f"bucket_{cfg.data.start_sample}_{cfg.data.end_sample}")
    else:
        out_dir = os.path.join(cfg.output_dir, f"bucket_{cfg.data.shard}_{cfg.data.num_shards}")
    os.makedirs(out_dir, exist_ok=True)
    os.chmod(out_dir, 0o777)
    checkpoint_path = os.path.join(out_dir, "last_saved_id.pth")

    config = {}
    if "ImageNet" in cfg.data._target_:
        run_id = f"{cfg.data.start_sample}_{cfg.data.end_sample}"
    else:
        run_id = f"{cfg.data.shard}_{cfg.data.num_shards}"
    if cfg.resume:
        print("run ID to resume: ", run_id)
    else:
        print("starting new run", run_id)
    config.update(OmegaConf.to_container(cfg, resolve=True))
    print("current run id: ", run_id)
    
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

    model = get_model(cfg_path=cfg.diffusion_model.cfg_path, ckpt_path = cfg.diffusion_model.ckpt_path).to(device).eval()
    print("Model successfully loaded.")
    
    classifier_model = get_classifier(cfg, device)
    classifier_model.to(device).eval()
    classifier_model.train = disabled_train

    print([f"{n}: {type(m)}" for n, m in classifier_model.named_modules()])

    ddim_steps = cfg.ddim_steps
    ddim_eta = cfg.ddim_eta
    scale = cfg.scale #for unconditional guidance
    strength = cfg.strength #for unconditional guidance

    if "seg_model" not in cfg or cfg.seg_model is None or "name" not in cfg.seg_model:
        sampler = CCMDDIMSampler(model, classifier_model, seg_model= None, classifier_wrapper="classifier_wrapper" in cfg.classifier_model and cfg.classifier_model.classifier_wrapper, record_intermediate_results=cfg.record_intermediate_results, verbose=cfg.verbose, **cfg.sampler)
    elif cfg.seg_model.name == "clipseg":
        sampler = CCMDDIMSampler(model, classifier_model, seg_model= model_seg, classifier_wrapper="classifier_wrapper" in cfg.classifier_model and cfg.classifier_model.classifier_wrapper, record_intermediate_results=cfg.record_intermediate_results, verbose=cfg.verbose, **cfg.sampler)
    else:
        sampler = CCMDDIMSampler(model, classifier_model, seg_model= model_seg, detect_model = detect_model, classifier_wrapper="classifier_wrapper" in cfg.classifier_model and cfg.classifier_model.classifier_wrapper, record_intermediate_results=cfg.record_intermediate_results, verbose=cfg.verbose, **cfg.sampler)

    sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=ddim_eta, verbose=False)

    assert 0. <= strength <= 1., 'can only work with strength in [0.0, 1.0]'
    t_enc = int(strength * len(sampler.ddim_timesteps))
    assert len(sampler.ddim_timesteps) == ddim_steps, "ddim_steps should be equal to len(sampler.ddim_timesteps)"
    n_samples_per_class = cfg.n_samples_per_class
    batch_size = cfg.data.batch_size
    shuffle = cfg.get("shuffle", False)
      

    #save config to the output directory
    #check if the config file already exists else create a config file
    config_path = os.path.join(out_dir, "config.yaml") 
    if os.path.exists(config_path):
        print("config file already exists! skipping ...")
    else:
        with open(os.path.join(out_dir, "config.yaml"), 'w') as f:
            print("saving config to ", os.path.join(out_dir, "config.yaml  ..."))
            yaml.dump(config, f)
            os.chmod(os.path.join(out_dir, "config.yaml"), 0o555)
    
    #data_path = cfg.data_path
    dataset = get_dataset(cfg, last_data_idx=last_data_idx)
    print("dataset length: ", len(dataset))
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=1)

    if "ImageNet" in cfg.data._target_:
        i2h = name_map
    else:
        raise NotImplementedError

    if "ImageNet" in cfg.data._target_:
        with open('data/synset_single_idx.yaml', 'r') as file:
        # with open('data/synset_closest_idx.yaml', 'r') as file:
            synset_closest_idx = yaml.safe_load(file)

    # layer_name = 'features.27'

    # with open(os.path.join(cfg.output_dir, 'concept_selection', f'conditions_{layer_name}.json'), 'r') as conditions_file:
    #     conditions = json.load(conditions_file)
    # conditions = {'features.28': 3}
    # concept_layers = ['classifier.0', 'classifier.1', 'classifier.2']
    # concept_layers = ['classifier.2', 'classifier.3', 'classifier.4']
    concept_layers = ['features.27']
    # concept_layers = ['classifier.6']
    concept_ids = np.arange(10, 21)

    if not cfg.resume:
        torch.save({"last_data_idx": -1}, checkpoint_path)
    
    seed = cfg.seed if "seed" in cfg else 0
    set_seed(seed=seed)

    for concept_layer in concept_layers:
        for concept_id in concept_ids:

            concept_condition = {concept_layer: concept_id}

            print(concept_condition)

            for i, batch in enumerate(data_loader):

                if "fixed_seed" in cfg:
                    set_seed(seed=cfg.get("seed", 0)) if cfg.fixed_seed else None
                    seed = seed if cfg.fixed_seed else -1
                    
                if "cond_tgt_cls" in cfg.data and cfg.data.cond_tgt_cls:
                    image,label, tgt_classes, unique_data_idx = batch

                    print(tgt_classes)

                    # tgts = []
                    # concept_conditions = []
                    # for udi in unique_data_idx:
                    #     udi_cond = conditions[str(udi.item())]
                    #     tgts.append(int(udi_cond['y']))
                    #     cond = {layer_name: udi_cond[layer_name][:4]}
                    #     concept_conditions.append(cond)

                    # tgt_classes = torch.tensor(tgts).to(device)     #from_numpy(np.array(tgts)).to(device)

                    # print(tgt_classes)
                elif "return_tgt_cls" in cfg.data and cfg.data.return_tgt_cls:
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

                image = image.to(device) #squeeze()
                label = label.to(device) #.item() #squeeze()
                #tgt_classes = torch.tensor([random.choice(synset_closest_idx[l.item()]) for l in label]).to(device)
                #tgt_classes = synset_closest_idx[label]
                #tgt_classes = torch.tensor([random.choice(synset_closest_idx[l.item()]) for l in label]).to(device)
                #shuffle tgt_classes
                #random.shuffle(tgt_classes)
                #get classifcation prediction
                with torch.inference_mode():
                    #with precision_scope():
                    if "classifier_wrapper" in cfg.classifier_model and cfg.classifier_model.classifier_wrapper:
                        logits = classifier_model(image)
                    else:
                        logits = sampler.get_classifier_logits(_unmap_img(image)) #converting to -1, 1
                    # TODO: handle binary vs multi-class
                    if "ImageNet" in cfg.data._target_ or "OxfordIIIPets" in cfg.data._target_ or "Flowers102" in cfg.data._target_: # multi-class
                        in_class_pred = logits.argmax(dim=1)
                        in_confid = logits.softmax(dim=1).max(dim=1).values
                        in_confid_tgt =  logits.softmax(dim=1)[torch.arange(batch_size), tgt_classes]
                    else: # binary
                        in_class_pred = (logits >= 0).type(torch.int8)
                        in_confid = torch.where(logits >= 0, logits.sigmoid(), 1 - logits.sigmoid())
                        in_confid_tgt =  torch.where(tgt_classes.to(device) == 0, 1 - logits.sigmoid(), logits.sigmoid())
                    print("in class_pred: ", in_class_pred, in_confid)
                
                for j, l in enumerate(label):
                    print(f"converting {i} from : {i2h[l.item()]} to: {i2h[int(tgt_classes[j].item())]}")

                # Compute concept conditioning
                # concept_conditions = compute_concept_conditioning(sampler.classifier, image, 'features.21', tgt_classes)
                # print(f"Concept conditions: {concept_conditions}")

                init_image = image.clone() #image.repeat(n_samples_per_class, 1, 1, 1).to(device)
                sampler.init_images = init_image.to(device)
                sampler.init_labels = label # n_samples_per_class * [label]
                if isinstance(cfg.sampler.lp_custom, str) and "dino_" in cfg.sampler.lp_custom:
                    if device != next(sampler.distance_criterion.dino.parameters()).device:
                        sampler.distance_criterion.dino = sampler.distance_criterion.dino.to(device)
                    sampler.dino_init_features = sampler.get_dino_features(sampler.init_images, device=device).clone()
                #mapped_image = _unmap_img(init_image)
                init_latent = model.get_first_stage_encoding(
                    model.encode_first_stage(_unmap_img(init_image)))  # move to latent space
                
                if "txt" == model.cond_stage_key: # text-conditional
                    if "ImageNet" in cfg.data._target_:
                        prompts = [f"a photo of a {openai_imagenet_classes[idx.item()]}." for idx in tgt_classes]
                    else:
                        raise NotImplementedError
                else:
                    prompts = None
                
                out = generate_samples(
                    model,
                    sampler,
                    tgt_classes,
                    ddim_steps,
                    scale,
                    temperature=cfg.temperature,
                    init_latent=None,           # init_latent.to(device),
                    t_enc=t_enc,
                    init_image=init_image.to(device),
                    ccdddim=True,
                    latent_t_0=cfg.get("latent_t_0", False),
                    prompts=prompts,
                    seed=seed,
                    concept_conditions=concept_condition,
                )
                print("Samples generated successfully.")

                all_samples = out["samples"]
                all_videos = out["videos"] 
                all_probs = out["probs"]
                all_masks = out["masks"] 
                all_cgs = out["cgs"]

                with torch.inference_mode():
                    if "classifier_wrapper" in cfg.classifier_model and cfg.classifier_model.classifier_wrapper:
                        logits = classifier_model(all_samples[0])
                    else:
                        logits = sampler.get_classifier_logits(_unmap_img(all_samples[0])) #converting to -1, 1 (it is converted back in the function)
                    if "ImageNet" in cfg.data._target_ or "CUB" in cfg.data._target_ or "OxfordIIIPets" in cfg.data._target_ or "Flowers102" in cfg.data._target_: # multi-class
                        out_class_pred = logits.argmax(dim=1)
                        out_confid = logits.softmax(dim=1).max(dim=1).values
                        out_confid_tgt = logits.softmax(dim=1)[torch.arange(batch_size), tgt_classes]
                    else: # binary
                        out_class_pred = (logits >= 0).type(torch.int8)
                        out_confid = torch.where(logits >= 0, logits.sigmoid(), 1 - logits.sigmoid())
                        out_confid_tgt =  torch.where(tgt_classes.to(device) == 0, 1 - logits.sigmoid(), logits.sigmoid())
                    print("out class_pred: ", out_class_pred, out_confid)
                    print(out_confid_tgt)

                # Loop through your data and update the table incrementally
                for j in range(batch_size):
                    # Generate data for the current row
                    src_image = copy.deepcopy(sampler.init_images[j].cpu()) #all_samples[j][0])
                    gen_image = copy.deepcopy(all_samples[0][j].cpu())
                    class_prediction = copy.deepcopy(all_probs[0][j]) if all_probs is not None else out_confid[j] # all_probs[j]
                    
                    source = i2h[label[j].item()]
                    target = i2h[int(tgt_classes[j].item())]
                    in_pred_cls = i2h[in_class_pred[j].item()]
                    out_pred_cls = i2h[out_class_pred[j].item()]

                    #diff =  (init_image - all_samples[j][1:])
                    diff = sampler.init_images[j]-all_samples[0][j]   
                    lp1 = int(torch.norm(diff, p=1, dim=-1).mean().cpu().numpy())
                    lp2 = int(torch.norm(diff, p=2, dim=-1).mean().cpu().numpy())
                    #print(f"lp1: {lp1}, lp2: {lp2}")

                    data_dict = {
                        "unique_id": unique_data_idx[j].item(), 
                        "image": src_image, 
                        "source": source, 
                        "target": target, 
                        "gen_image": gen_image,
                        "target_confidence": class_prediction, 
                        "in_pred": in_pred_cls, 
                        "out_pred": out_pred_cls, 
                        "out_confid": out_confid[j].cpu().item(), 
                        "out_tgt_confid": out_confid_tgt[j].cpu().item(), 
                        "in_confid": in_confid[j].cpu().item(), 
                        "in_tgt_confid": in_confid_tgt[j].cpu().item(), 
                        "closness_1": lp1, 
                        "closness_2": lp2,
                        # "concept_conditions": concept_conditions,
                    }
                    if cfg.record_intermediate_results:
                        if all_videos is not None:
                            video_results = {
                                "video": (255. * all_videos[0][j]).to(torch.uint8).cpu(), 
                            }
                            data_dict = dict(data_dict, **video_results)
                        if all_cgs is not None:
                            cgs_results = {
                                "cgs": (255.*all_cgs[0][j]).to(torch.float32).cpu(),
                            }
                            data_dict = dict(data_dict, **cgs_results)

                    if "Flowers102" in cfg.data._target_ or "OxfordIIIPets" in cfg.data._target_:
                        uidx = unique_data_idx[j].item()*cfg.data.num_shards + cfg.data.shard
                    else:
                        uidx = unique_data_idx[j].item()

                    dict_save_path = os.path.join(out_dir, f'{str(uidx).zfill(5)}.pth')
                    torch.save(data_dict, dict_save_path)
                    os.chmod(dict_save_path, 0o555)

                    pathlib.Path(os.path.join(out_dir, 'original')).mkdir(parents=True, exist_ok=True, mode=0o777)
                    os.chmod(os.path.join(out_dir, 'original'), 0o777)
                    pathlib.Path(os.path.join(out_dir, f'{concept_layer}_{concept_id}_counterfactual')).mkdir(parents=True, exist_ok=True, mode=0o777)
                    os.chmod(os.path.join(out_dir, f'{concept_layer}_{concept_id}_counterfactual'), 0o777)
                    orig_save_path = os.path.join(out_dir, 'original', f'{str(uidx).zfill(5)}.png')
                    save_image(src_image.clip(0, 1), orig_save_path)
                    os.chmod(orig_save_path, 0o555)

                    cf_save_path = os.path.join(out_dir, f'{concept_layer}_{concept_id}_counterfactual', f'{str(uidx).zfill(5)}.png')
                    save_image(gen_image.clip(0, 1), cf_save_path)
                    os.chmod(cf_save_path, 0o555)

                if (i + 1) % cfg.log_rate == 0:
                    last_data_idx = unique_data_idx[-1].item()
                    torch.save({
                        #"table": copy.deepcopy(my_table),
                        "last_data_idx": last_data_idx,
                    }, checkpoint_path)
                    os.chmod(checkpoint_path, 0o777)
                    print(f"saved {checkpoint_path}, with data_id {last_data_idx}")

                del out

                if i > 15:
                    break
            
    return None

if __name__ == "__main__":
    main()
