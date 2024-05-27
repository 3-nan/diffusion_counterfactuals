""" Per sample, the localization constraint for single concepts shall be visualized in input space. """
import os
import sys
sys.path.append("./")
sys.path.append("./ldce")
sys.path.append("./data")
import copy
import cv2
import h5py
import hydra
import json
from omegaconf import OmegaConf, DictConfig, open_dict
import numpy as np
import random
import torch
from torch import nn
import torchvision.transforms.functional as F
from torchvision.transforms.functional import InterpolationMode
from torchvision.models.feature_extraction import create_feature_extractor
import yaml
import matplotlib.pyplot as plt
import zennit

from ldce.data.imagenet_classnames import name_map

from src.concept_conditioning import compute_concept_conditioning
from src.sampling_helpers import disabled_train
from src.helpers.data_model_helpers import get_classifier, get_dataset, set_seed

def create_mask_overlay(image, mask, blur_th=0.25):

    if image.shape[0] == 3:

        image = image.transpose((1,2,0))
        mask = mask.transpose((1,2,0))

    image = np.array(image * 255, dtype=np.uint8).copy()

    inv_mask = np.array(mask == 0, dtype=np.uint8)
    bgr_mask = cv2.cvtColor((inv_mask * 255).copy(), cv2.COLOR_GRAY2BGR)

    masked_image = cv2.addWeighted(image, 1.0, bgr_mask, blur_th, 0)

    _, thresh = cv2.threshold(mask*255, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(masked_image, contours, -1, (0, 255, 0), 2)

    return masked_image

def show_localization_gradient(model, image, target, conditions, cinds, start_idx=0):

    start_idx = start_idx * len(image)

    for i, (img, tgt, cond, inds) in enumerate(zip(image, target, conditions, cinds)):

        inp_img = _map_img(img)
        if not self.classifier_wrapper: # only works for ImageNet!
            inp_img = tf.center_crop(inp_img, 224)
            inp_img = normalize(inp_img)

        fig, ax = plt.subplots(len(inds), 1, figsize=(4, int(2.5*len(inds))))

        for d, ind in enumerate(inds):

            hook_map, y_targets = {}, []
            for key in concept_conditions.keys():
                if key not in hook_map:
                    hook_map[key] = MaskHook([])

                _register_mask_fn(hook_map[key], spatial_map, 0, concept_conditions[key], key)

            name_map = [([name], hook) for name, hook in hook_map.items()]
            mask_composite = NameMapComposite(name_map)

            with zennit.attribution.Gradient(model) as modified:

                pred, attr = modified(inp_img)

            t = cond[ind].cpu()
            t_resized = nn.functional.interpolate(t[None, None, :, :], 256, mode='bilinear')

            binary_mask = np.array(t_resized[0].numpy() > 0.5, dtype=np.uint8)

            cv_img = create_mask_overlay(img.cpu().numpy(), binary_mask)


            ax[d].imshow(cv_img)

            ax[d].set_title(ind, x=-0.1, y=0.45, rotation=90)
            ax[d].axis('off')

        plt.tight_layout()
        plt.savefig(f'/results/counterfactuals/localization/{str(start_idx + i).zfill(5)}_localization.svg')
        plt.close()

def show_localization_constraints(model, image, conditions, cinds, concept_layer, start_idx=0):

    start_idx = start_idx * len(image)

    for i, (img, cond, inds) in enumerate(zip(image, conditions, cinds)):

        # summed = torch.sum(cond, dim=0).cpu()
        # summed = (summed > 0)
        # t_resized = F.resize(summed[None, None, :, :], 256)

        # fig, ax = plt.subplots(1, len(inds), figsize=(12, 4))

        # for d, ind in enumerate(inds):

        #     # print(cond.size())
        #     # print(cond[ind].size())
        #     t = cond[ind].cpu()
        #     # t_resized = nn.functional.interpolate(t[None, None, :, :], 256, mode='bilinear')
        #     # t_resized = F.resize(t[None, None, :, :], 256, interpolation=InterpolationMode.BILINEAR)
        #     ax[d].imshow(zennit.image.imgify(t.float()))
        
        # plt.savefig(f'/results/counterfactuals/localization/raw_{start_idx + i}.png')
        # plt.close()

        fig, ax = plt.subplots(len(inds), 1, figsize=(4, int(2.5*len(inds))))

        # ax[0].imshow(zennit.image.imgify(img.cpu()))
        # ax[0].axis('off')

        for d, ind in enumerate(inds[::-1]):

            # print(cond.size())
            # print(cond[ind].size())
            t = cond[ind].cpu()
            t_resized = nn.functional.interpolate(t[None, None, :, :], 256, mode='bilinear')
            # t_resized = F.resize(t[None, None, :, :], 256, interpolation=InterpolationMode.BILINEAR)

            # binary_mask = np.array(t_resized[0].numpy() * 255).astype(np.uint8)
            binary_mask = np.array(t_resized[0].numpy() > 0.5, dtype=np.uint8)

            cv_img = create_mask_overlay(img.cpu().numpy(), binary_mask)

            # binary_mask = np.array(t_resized.numpy() * 255).astype(np.uint8)
            # binary_mask = np.array((binary_mask > 0.5), dtype=np.uint8)

            # gray = cv2.cvtColor(binary_mask, cv2.COLOR_BGR2GRAY)
            # _, thresh = cv2.threshold(binary_mask, 1, 255, cv2.THRESH_BINARY)
            # contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # cv2.drawContours(base_img, contours, -1, (0, 255, 0), 2)

            ax[d].imshow(cv_img)

            ax[d].set_title(ind, x=-0.1, y=0.45, rotation=90)
            ax[d].axis('off')

        plt.tight_layout()
        plt.savefig(f'/results/counterfactuals/localization/flowers_{concept_layer}_{str(start_idx + i).zfill(5)}_localization.svg')
        plt.close()

        # model = copy.deepcopy(model_original)


        # fig, ax = plt.subplots(1, len(inds) + 1, figsize=(12, 4))
        # ax[0].imshow(zennit.image.imgify(img.cpu()))

        # for d, ind in enumerate(inds):

        #     # print(cond.size())
        #     # print(cond[ind].size())
        #     t = cond[ind].cpu()

        #     # compute gradient
        #     # model = model.train()

        #     # for module in model.modules():
        #     # #     # skip errors on container modules, like nn.Sequential
        #     #     try:
        #     # #         # Make all convolution weights equal.
        #     # #         # Set all biases to zero.
        #     #         nn.init.constant_(module.weight, 0.05)
        #     #         nn.init.zeros_(module.bias)
            
        #     # #         # Set BatchNorm means to zeros, 
        #     # #         # variances - to 1.
        #     #         nn.init.zeros_(module.running_mean)
        #     #         nn.init.ones_(module.running_var)
        #     #     except:
        #     #         pass

        #     #     # # Freeze the BatchNorm stats. 
        #     #     if isinstance(module, torch.nn.modules.BatchNorm2d):
        #     #         module.eval()

        #     return_nodes = {
        #         "features.40": "layer4"
        #     }
        #     model2 = create_feature_extractor(model, return_nodes=return_nodes)
        #     # intermediate_outputs = model2(x)

        #     input = torch.ones_like(img)
        #     input = F.center_crop(input, 224)
        #     input.requires_grad = True
        #     # out = model(input[None])
        #     intermediate_outputs = model2(input[None])

        #     # # Set the gradient to 0.
        #     # # Only set the pixel of interest to 1.
        #     # print(cond.size())
        #     grad = torch.zeros_like(cond, requires_grad=False)[None]
        #     # print(grad.size())
        #     grad[0, ind, :, :] = cond[ind]

        #     grad.requires_grad = True

        #     # # Run the backprop.
        #     intermediate_outputs['layer4'].backward(gradient=grad)
            
        #     # # Retrieve the gradient of the input image.
        #     # gradient_of_input = input.grad[0, 0].data.numpy()
        #     input_grad = input.grad.data.cpu().numpy()

        #     # # Normalize the gradient.
        #     # gradient_of_input = gradient_of_input / np.amax(gradient_of_input)
        #     input_grad = input_grad / np.amax(input_grad)

        #     # def normalize(activations):
        #     #     # transform activations so that all the values be in range [0, 1]
        #     #     activations = activations - np.min(activations)
        #     #     activations = activations / np.max(activations)
        #     #     return activations
            
            
        #     def visualize_activations(image, activations):
        #         # activations = normalize(activations)
        #         activations = activations / np.linalg.norm(activations)
            
        #         # replicate the activations to go from 1 channel to 3
        #         # as we have colorful input image
        #         # we could use cvtColor with GRAY2BGR flag here, but it is not
        #         # safe - our values are floats, but cvtColor expects 8-bit or
        #         # 16-bit integers
        #         # activations = np.stack([activations, activations, activations], axis=2)
        #         masked_image = (image * activations * 255).astype(np.uint8)
                
        #         return masked_image

        #     input_grad = visualize_activations(input.cpu().detach().numpy(), input_grad)

        #     ax[d+1].imshow(zennit.image.imgify(input_grad))
        #     ax[1 + d].set_title(ind)
        #     # ax[d+1].imshow(input_grad.transpose((1,2,0)))

        # plt.savefig(f'/results/counterfactuals/localization/input_grad_{start_idx + i}.png')
        # plt.close()


@hydra.main(version_base=None, config_path="../../configs/ldce", config_name="v1")
def main(cfg : DictConfig) -> None:
    if "verbose" not in cfg:
        with open_dict(cfg):
            cfg.verbose = True
    if "record_intermediate_results" not in cfg:
        with open_dict(cfg):
            cfg.record_intermediate_results = True

    # if "verbose" in cfg and not cfg.verbose:
    #     blockPrint()

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

    # model = get_model(cfg_path=cfg.diffusion_model.cfg_path, ckpt_path = cfg.diffusion_model.ckpt_path).to(device).eval()
    
    classifier_model = get_classifier(cfg, device)

    if "Flowers102" in cfg.data._target_:
        weights_path = "/results/models/vgg16bn_flowers_20240503_122623_78_0.870"
        num_ftrs = classifier_model.classifier[6].in_features
        classifier_model.classifier[6] = torch.nn.Linear(num_ftrs, 103)
        classifier_model.load_state_dict(torch.load(weights_path))
        # classifier_model.to(device)
    elif "OxfordIIIPets" in cfg.data._target_:
        weights_path = "/results/models/vgg16bn_pets_20240503_092321_7_0.92"
        num_ftrs = classifier_model.classifier[6].in_features
        classifier_model.classifier[6] = torch.nn.Linear(num_ftrs, 37)
        classifier_model.load_state_dict(torch.load(weights_path))
        # model.to(device)

    classifier_model.to(device).eval()
    # classifier_model.train = disabled_train

    ddim_steps = cfg.ddim_steps
    ddim_eta = cfg.ddim_eta
    scale = cfg.scale #for unconditional guidance
    strength = cfg.strength #for unconditional guidance

    # sampler = ConceptCCMDDIMSampler(model, classifier_model, seg_model= None, classifier_wrapper="classifier_wrapper" in cfg.classifier_model and cfg.classifier_model.classifier_wrapper, record_intermediate_results=cfg.record_intermediate_results, verbose=cfg.verbose, **cfg.sampler)

    # sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=ddim_eta, verbose=False)

    assert 0. <= strength <= 1., 'can only work with strength in [0.0, 1.0]'
    # t_enc = int(strength * len(sampler.ddim_timesteps))
    # assert len(sampler.ddim_timesteps) == ddim_steps, "ddim_steps should be equal to len(sampler.ddim_timesteps)"
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
    elif "Flowers102" in cfg.data._target_:
        with open("data/flowers_idx_to_label.json", "r") as f:
            flowers_idx_to_classname = json.load(f)
        flowers_idx_to_classname = {int(k)-1: v for k, v in flowers_idx_to_classname.items()}
        i2h = flowers_idx_to_classname
    elif "OxfordIIIPets" in cfg.data._target_:
        with open("data/pets_idx_to_label.json", "r") as f:
            pets_idx_to_classname = json.load(f)
        i2h = {int(k): v for k, v in pets_idx_to_classname.items()}
    else:
        raise NotImplementedError

    if "ImageNet" in cfg.data._target_:
        with open('data/synset_closest_idx.yaml', 'r') as file:
            synset_closest_idx = yaml.safe_load(file)
    elif "Flowers102" in cfg.data._target_:
        with open("data/flowers_closest_indices.json") as file:
            closest_indices = json.load(file)
        closest_indices = {int(k):v for k,v in closest_indices.items()}
        num_classes = 103
    elif "OxfordIIIPets" in cfg.data._target_:
        with open("data/pets_closest_indices.json") as file:
            closest_indices = json.load(file)
        closest_indices = {int(k):v for k,v in closest_indices.items()}
        num_classes = 37

    concept_layer = cfg.concept_layer       # "backbone.features.29"
    spatial = cfg.spatial

    # conditioning_file = os.path.join(out_dir, f'conditioning_{cfg.classifier_model.name}.h5')

    if not cfg.resume:
        torch.save({"last_data_idx": -1}, checkpoint_path)
    
    seed = cfg.seed if "seed" in cfg else 0
    set_seed(seed=seed)

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
                num_classes = 1000
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

        # for layer in ['features.37', 'features.40']:

        # for cond_option in ['sumabs', 'sum', 'sumequal', 'absmean', 'abssum', 'absmax']:
        cond_option = "sumabs"

        # Compute concept conditions
        # ToDo: add sampler.classifier_wrapper as parameter
        if spatial:
            conditions, concept_conds, concept_diff, grad = compute_concept_conditioning(classifier_model, image, tgt_classes, concept_layer, num_concepts=cfg.num_concepts, num_classes=num_classes, spatial=spatial, cond_option=cond_option, return_gradient=True)
        else:
            conditions, concept_conds, concept_diff, grad = compute_concept_conditioning(classifier_model, image, tgt_classes, concept_layer, num_concepts=cfg.num_concepts, num_classes=num_classes, cond_option=cond_option, return_gradient=True)

        # print(conditions)
        # print(conditions.keys())

        # print(conditions[concept_layer].size())
        # print(concept_conds[concept_layer])

        show_localization_constraints(classifier_model, image, conditions[concept_layer], concept_conds[concept_layer], concept_layer, start_idx=i)
        # raise ValueError

        # if i > 6:
        #     raise ValueError

if __name__ == '__main__':
    main()
