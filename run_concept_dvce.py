
import random
import sys
from torchvision.utils import save_image
import hydra
import json
import copy

sys.path.append('DVCEs')
# from blended_diffusion.optimization.dff_attack import DiffusionAttack
# from blended_diffusion.optimization.arguments import get_arguments
# from configs import get_config
# from utils_svces.datasets.paths import get_imagenet_path
# from utils_svces.datasets.imagenet import get_imagenet_labels
import utils_svces.datasets as dl
from utils_svces.functions import blockPrint
import torch
# import torch.nn as nn
import numpy as np
from omegaconf import DictConfig, open_dict, OmegaConf
import os
import pathlib
import matplotlib as mpl
mpl.use('Agg')
# import matplotlib.pyplot as plt
from utils_svces.load_trained_model import load_model
from tqdm import trange
from time import sleep

from ldce.data.imagenet_classnames import name_map, openai_imagenet_classes
from utils_svces.train_types.helpers import create_attack_config, get_adversarial_attack

from utils_svces.Evaluator import Evaluator
from src.concept_dvce import ConceptDiffusionAttack
from src.concept_conditioning import compute_concept_conditioning
from run_concept_ldce import get_dataset

def set_seed(seed: int = 0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)

# ldce code...
@hydra.main(version_base=None, config_path="configs/dvce", config_name="v1")
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

    out_dir = cfg.output_dir
    # if "ImageNet" in cfg.data._target_:
    #     out_dir = os.path.join(cfg.output_dir, f"bucket_{cfg.data.start_sample}_{cfg.data.end_sample}")
    # else:
    #     out_dir = os.path.join(cfg.output_dir, f"bucket_{cfg.data.shard}_{cfg.data.num_shards}")
    # os.makedirs(out_dir, exist_ok=True)
    # os.chmod(out_dir, 0o777)
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

    # init DVCE attack
    use_diffusion = False
    method = 'dvces'
    # for method in [hps.method]:
    if method.lower() == 'svces':
        radii = np.array([150.])
        attack_type = 'afw'
        norm = 'L1.5'
        stepsize = None
        steps = 75
    elif method.lower() == 'apgd':
        radii = np.array([12.])
        attack_type = 'apgd'
        norm = 'L2' 
        stepsize = None
        steps = 75
    elif method.lower() == 'dvces':
        attack_type = 'diffusion'
        radii = np.array([0.])
        norm = 'L2'
        steps = 150
        stepsize = None
        use_diffusion=True
    else:
        raise NotImplementedError()


    attack_config = create_attack_config(eps=radii[0], steps=steps, stepsize=stepsize, norm=norm, momentum=0.9,
                                         pgd=attack_type)

    # num_classes = len(in_labels)
    if attack_type == 'diffusion':
        img_dimensions = (3, 256, 256)
    else:
        img_dimensions = imgs.shape[1:]
    num_targets = 1
    num_radii = len(radii)
    # num_imgs = len(imgs)

    # att = DiffusionAttack(hps)
    att = ConceptDiffusionAttack(cfg, device)

    # raise ValueError

    if att.args.second_classifier_type != -1:
        print('setting model to second classifier')
        model = att.second_classifier
    else:
        model = att.classifier

    # for radius_idx in range(len(radii)):
    #     batch_data = batch_data.to(device)
    #     batch_targets = batch_targets.to(device)


    #     if not use_diffusion:
    #         att.eps = radii[radius_idx]

    #     if use_diffusion:
    #         batch_adv_samples_i = att.perturb(batch_data,
    #                                                 batch_targets, dir)[
    #             0].detach()


    dataset = get_dataset(cfg, last_data_idx=last_data_idx)
    dataset = torch.utils.data.Subset(dataset, np.arange(1000))
    print(type(dataset))
    print("dataset length: ", len(dataset))
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=4)

    if "ImageNet" in cfg.data._target_:
        i2h = name_map
        num_classes = 1000

    seed = cfg.seed if "seed" in cfg else 0
    set_seed(seed=seed)

    for i, batch in enumerate(data_loader):

        # if "fixed_seed" in cfg:
        #     set_seed(seed=cfg.get("seed", 0)) if cfg.fixed_seed else None
        #     seed = seed if cfg.fixed_seed else -1
            
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
        #tgt_classes = torch.tensor([random.choice(synset_closest_idx[l.item()]) for l in label]).to(device)
        #tgt_classes = synset_closest_idx[label]
        #tgt_classes = torch.tensor([random.choice(synset_closest_idx[l.item()]) for l in label]).to(device)
        #shuffle tgt_classes
        #random.shuffle(tgt_classes)

        # print(type(att.second_classifier.model))

        # Compute concept conditions
        # ToDo: add sampler.classifier_wrapper as parameter
        if cfg.spatial:
            conditions, concept_conds, concept_diff = compute_concept_conditioning(att.second_classifier.model, image, tgt_classes, cfg.concept_layer, num_concepts=cfg.num_concepts, num_classes=1000, spatial=cfg.spatial, cond_option=cfg.cond_option)
        else:
            conditions, concept_conds, concept_diff = compute_concept_conditioning(att.second_classifier.model, image, tgt_classes, cfg.concept_layer, num_concepts=cfg.num_concepts, num_classes=1000, cond_option=cfg.cond_option)

        #get classifcation prediction
        with torch.inference_mode():
            #with precision_scope():
            if "classifier_wrapper" in cfg.classifier_model and cfg.classifier_model.classifier_wrapper:
                logits = att.second_classifier(image)
            else:
                # logits = sampler.get_classifier_logits(_unmap_img(image)) #converting to -1, 1
                # x = _map_img(x)
                # if not self.classifier_wrapper: # only works for ImageNet!
                #     x = tf.center_crop(x, 224)
                #     x = normalize(x)
                # logits self.classifier(x)
                logits = att.second_classifier(image)
            # TODO: handle binary vs multi-class
            if "ImageNet" in cfg.data._target_ or "OxfordIIIPets" in cfg.data._target_ or "Flowers102" in cfg.data._target_: # multi-class
                in_class_pred = logits.argmax(dim=1)
                in_confid = logits.softmax(dim=1).max(dim=1).values
                in_confid_tgt =  logits.softmax(dim=1)[torch.arange(cfg.batch_size), tgt_classes]
            else: # binary
                in_class_pred = (logits >= 0).type(torch.int8)
                in_confid = torch.where(logits >= 0, logits.sigmoid(), 1 - logits.sigmoid())
                in_confid_tgt =  torch.where(tgt_classes.to(device) == 0, 1 - logits.sigmoid(), logits.sigmoid())
            print("in class_pred: ", in_class_pred, in_confid)
        
        for j, l in enumerate(label):
            print(f"converting {i} from : {i2h[l.item()]} to: {i2h[int(tgt_classes[j].item())]}")
        
        init_image = image.clone() #image.repeat(n_samples_per_class, 1, 1, 1).to(device)
        # sampler.init_images = init_image.to(device)
        # sampler.init_labels = label # n_samples_per_class * [label]
        # if isinstance(cfg.sampler.lp_custom, str) and "dino_" in cfg.sampler.lp_custom:
        #     if device != next(sampler.distance_criterion.dino.parameters()).device:
        #         sampler.distance_criterion.dino = sampler.distance_criterion.dino.to(device)
        #     sampler.dino_init_features = sampler.get_dino_features(sampler.init_images, device=device).clone()
        #mapped_image = _unmap_img(init_image)

        # init_latent = model.get_first_stage_encoding(
        #     model.encode_first_stage(_unmap_img(init_image)))  # move to latent space
        
        # if "txt" == model.cond_stage_key: # text-conditional
        #     if "ImageNet" in cfg.data._target_:
        #         prompts = [f"a photo of a {openai_imagenet_classes[idx.item()]}." for idx in tgt_classes]
        #     elif "CelebAHQDataset" in cfg.data._target_:
        #         # query label 31 (smile): label=0 <-> no smile and label=1 <-> smile
        #         # query label 39 (age): label=0 <-> old and label=1 <-> young
        #         assert cfg.data.query_label in [31, 39]
        #         prompts = []
        #         for target in tgt_classes:
        #             if cfg.data.query_label == 31 and target == 0:
        #                 attr = "non-smiling"
        #             elif cfg.data.query_label == 31 and target == 1:
        #                 attr = "smiling"
        #             elif cfg.data.query_label == 39 and target == 0:
        #                 attr = "old"
        #             elif cfg.data.query_label == 39 and target == 1:
        #                 attr = "young"
        #             else:
        #                 raise NotImplementedError
        #             prompts.append(f"a photo of a {attr} person")
        #     elif "OxfordIIIPets" in cfg.data._target_:
        #         # prompts following https://github.com/openai/CLIP/blob/main/data/prompts.md
        #         prompts = [f"a photo of a {i2h[idx.item()]}, a type of pet." for idx in tgt_classes]
        #     elif "Flowers102" in cfg.data._target_:
        #         # prompts following https://github.com/openai/CLIP/blob/main/data/prompts.md
        #         prompts = [f"a photo of a {i2h[idx.item()]}, a type of flower." for idx in tgt_classes]
        #     else:
        #         raise NotImplementedError
        # else:
        #     prompts = None

        # if "Flowers102" in cfg.data._target_:
        #     print(prompts)
        #     print(tgt_classes)
        
        # batch_adv_samples_i = att.perturb(batch_data, batch_targets, dir)[0].detach()

        # print(seed)
        batch_adv_samples_i = att.perturb(image, tgt_classes, dir, seed, conditions=conditions)[0].detach()


        # out = generate_samples(
        #     model, 
        #     sampler, 
        #     tgt_classes, 
        #     ddim_steps, 
        #     scale, 
        #     init_latent=init_latent.to(device),
        #     t_enc=t_enc, 
        #     init_image=init_image.to(device), 
        #     ccdddim=True, 
        #     latent_t_0=cfg.get("latent_t_0", False),
        #     prompts=prompts, 
        #     seed=seed,
        #     conditions=conditions,
        #     spatial=spatial,
        # )

        all_samples = batch_adv_samples_i

        # # all_samples = out["samples"]
        # all_videos = out["videos"] 
        # all_probs = out["probs"]
        # # all_masks = out["masks"] 
        # all_cgs = out["cgs"]

        with torch.inference_mode():
            if "classifier_wrapper" in cfg.classifier_model and cfg.classifier_model.classifier_wrapper:
                logits = att.second_classifier(all_samples[0])
            else:
                # logits = sampler.get_classifier_logits(_unmap_img(all_samples[0])) #converting to -1, 1 (it is converted back in the function)
                logits = att.second_classifier(all_samples)
            if "ImageNet" in cfg.data._target_ or "CUB" in cfg.data._target_ or "OxfordIIIPets" in cfg.data._target_ or "Flowers102" in cfg.data._target_: # multi-class
                out_class_pred = logits.argmax(dim=1)
                out_confid = logits.softmax(dim=1).max(dim=1).values
                out_confid_tgt = logits.softmax(dim=1)[torch.arange(cfg.batch_size), tgt_classes]
            else: # binary
                out_class_pred = (logits >= 0).type(torch.int8)
                out_confid = torch.where(logits >= 0, logits.sigmoid(), 1 - logits.sigmoid())
                out_confid_tgt =  torch.where(tgt_classes.to(device) == 0, 1 - logits.sigmoid(), logits.sigmoid())
            print("out class_pred: ", out_class_pred, out_confid)
            print(out_confid_tgt)

        all_probs = None
        # Loop through your data and update the table incrementally
        for j in range(cfg.batch_size):
            # Generate data for the current row
            src_image = copy.deepcopy(image[j].cpu())  # copy.deepcopy(sampler.init_images[j].cpu()) #all_samples[j][0])
            gen_image = copy.deepcopy(all_samples[j].cpu())
            # class_prediction = copy.deepcopy(all_probs[0][j]) if all_probs is not None else out_confid[j] # all_probs[j]
            class_prediction = copy.deepcopy(all_probs[j]) if all_probs is not None else out_confid[j] # all_probs[j]
            
            source = i2h[label[j].item()]
            target = i2h[int(tgt_classes[j].item())]
            in_pred_cls = i2h[in_class_pred[j].item()]
            out_pred_cls = i2h[out_class_pred[j].item()]

            #diff =  (init_image - all_samples[j][1:])
            # diff = sampler.init_images[j]-all_samples[0][j]   
            diff = image[j] - all_samples[j]
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
                # "closness_1": lp1, 
                # "closness_2": lp2,
                # "conditions": concept_conds[cfg.concept_layer][j],
                # "concept_diff": concept_diff[j],
            }
            if cfg.record_intermediate:
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

            # # Save data dict
            dict_save_path = os.path.join(out_dir, f'{str(uidx).zfill(5)}.pth')
            torch.save(data_dict, dict_save_path)
            os.chmod(dict_save_path, 0o555)

            pathlib.Path(os.path.join(out_dir, 'original')).mkdir(parents=True, exist_ok=True, mode=0o777)
            os.chmod(os.path.join(out_dir, 'original'), 0o777)
            pathlib.Path(os.path.join(out_dir, 'counterfactual')).mkdir(parents=True, exist_ok=True, mode=0o777)
            os.chmod(os.path.join(out_dir, 'counterfactual'), 0o777)
            orig_save_path = os.path.join(out_dir, 'original', f'{str(uidx).zfill(5)}.png')
            save_image(src_image.clip(0, 1), orig_save_path)
            os.chmod(orig_save_path, 0o555)

            cf_save_path = os.path.join(out_dir, 'counterfactual', f'{str(uidx).zfill(5)}.png')
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

        # del out
            
    return None
    raise ValueError



# hps = get_config(get_arguments())

# if not hps.verbose:
#     blockPrint()


# if len(hps.gpu)==0:
#     device = torch.device('cpu')
#     print('Warning! Computing on CPU')
#     num_devices = 1
# elif len(hps.gpu)==1:
#     hps.device_ids = [int(hps.gpu[0])]
#     device = torch.device('cuda:' + str(hps.gpu[0]))
#     num_devices = 1
# else:
#     hps.device_ids = [int(i) for i in hps.gpu]
#     device = torch.device('cuda:' + str(min(hps.device_ids)))
#     num_devices = len(hps.device_ids)
# hps.device = device

# # configs
# img_size = 256
# num_imgs = hps.num_imgs
# pixel_d_min = 5
# conf_threshold = 0.01
# out_dir = 'ImageNetVCEs'
# dataset = 'imagenet'
# imagenet_mode = 'examples'
# bs = hps.batch_size * len(hps.device_ids)

# torch.manual_seed(hps.seed)
# random.seed(hps.seed)
# np.random.seed(hps.seed)

# in_labels = get_imagenet_labels(hps.data_folder)

if __name__ == '__main__':

    main()

# in_loader = dl.get_ImageNet(path=hps.data_folder, train=False, augm_type='crop_0.875', size=img_size)
# in_dataset = in_loader.dataset

# accepted_wnids = []

# some_vces = {
#     14655: [288, 292],
#     10452: [207, 208],
#     46751: [924, 959],
#     48679: [970, 972],
#     48539: [970, 980],
#     48282: [963, 965]
# }


# # def _plot_counterfactuals(dir, original_imgs, orig_labels, segmentations, targets,
# #                           perturbed_imgs, perturbed_probabilities, original_probabilities, radii, class_labels, filenames=None, img_idcs=None, num_plot_imgs=hps.num_imgs):
# #     num_imgs = num_plot_imgs
# #     num_radii = len(radii)
# #     scale_factor = 4.0
# #     target_idx = 0


# #     if img_idcs is None:
# #         img_idcs = torch.arange(num_imgs, dtype=torch.long)

# #     pathlib.Path(dir+'/single_images').mkdir(parents=True, exist_ok=True)

# #     # Two VCEs per starting image - we fix them to 2
# #     num_VCEs_per_image = 2
# #     for lin_idx in trange(int(len(img_idcs)/num_VCEs_per_image), desc=f'Image write'):

# #         # we fix only one radius
# #         radius_idx = 0
# #         lin_idx *= num_VCEs_per_image

# #         img_idx = img_idcs[lin_idx]




# #         in_probabilities = original_probabilities[img_idx, target_idx, radius_idx]


# #         pred_original = in_probabilities.argmax()
# #         pred_value = in_probabilities.max()

# #         num_rows = 1
# #         num_cols = num_VCEs_per_image + 1
# #         fig, ax = plt.subplots(num_rows, num_cols,
# #                                figsize=(scale_factor * num_cols, num_rows * 1.3 * scale_factor))
# #         img_label = orig_labels[img_idx]
# #         title = f'GT: {class_labels[img_label]}' #, predicted: {class_labels[pred_original]},{pred_value:.2f}'

# #         img_segmentation = segmentations[img_idx]
# #         bin_segmentation = torch.sum(img_segmentation, dim=0) > 0.0
# #         img_segmentation[:, bin_segmentation] = 0.5
# #         mask_color = torch.zeros_like(img_segmentation)
# #         mask_color[1, :, :] = 1.0

# #         # plot original:
# #         ax[0].axis('off')
# #         ax[0].set_title(title)
# #         img_original = original_imgs[img_idx, :].permute(1, 2, 0).cpu().detach()
# #         ax[0].imshow(img_original, interpolation='lanczos')

# #         save_image(original_imgs[img_idx, :].clip(0, 1),
# #                    os.path.join(dir, 'single_images', f'{img_idx}_original.png'))
# #         for i in range(num_VCEs_per_image):

# #             img = torch.clamp(perturbed_imgs[img_idx+i, target_idx, radius_idx].permute(1, 2, 0), min=0.0,
# #                               max=1.0)
# #             img_probabilities = perturbed_probabilities[img_idx+i, target_idx, radius_idx]

# #             img_target = targets[img_idx+i]
# #             target_original = in_probabilities[img_target]

# #             target_conf = img_probabilities[img_target]

# #             ax[i+1].axis('off')
# #             ax[i+1].imshow(img, interpolation='lanczos')

# #             title = f'{class_labels[img_target]}: {target_conf:.2f}, i:{target_original:.2f}'
# #             ax[i+1].set_title(title)


# #             save_image(perturbed_imgs[img_idx+i, target_idx, radius_idx].clip(0, 1), os.path.join(dir, 'single_images', f'{img_idx}_{class_labels[img_target]}.png'))

# #         plt.tight_layout()
# #         if filenames is not None:
# #             fig.savefig(os.path.join(dir, f'{filenames[lin_idx]}.png'))
# #             fig.savefig(os.path.join(dir, f'{filenames[lin_idx]}.pdf'))
# #         else:
# #             fig.savefig(os.path.join(dir, f'{lin_idx}.png'))
# #             fig.savefig(os.path.join(dir, f'{lin_idx}.pdf'))

# #         plt.close(fig)

# # plot = False
# # plot_top_imgs = True

# # imgs = torch.zeros((num_imgs, 3, img_size, img_size))
# # segmentations = torch.zeros((num_imgs, 3, img_size, img_size))
# # targets_tensor = torch.zeros(num_imgs, dtype=torch.long)
# # labels_tensor = torch.zeros(num_imgs, dtype=torch.long)
# # filenames = []

# # image_idx = 0
# # kernel = np.ones((5, 5), np.uint8)

# # selected_vces = list(some_vces.items())


# # if hps.world_size > 1:
# #     print('Splitting relevant classes')
# #     print(f'{hps.world_id} out of {hps.world_size}')
# #     splits = np.array_split(np.arange(len(selected_vces)), hps.world_size)
# #     print(f'Using clusters {splits[hps.world_id]} out of {len(targets_tensor)}')

# for i, (img_idx, target_classes) in enumerate(selected_vces):
#     if hps.world_size > 1 and i not in splits[hps.world_id]:
#         pass
#     else:
#         in_image, label = in_dataset[img_idx]
#         for i in range(len(target_classes)):
#             targets_tensor[image_idx+i] = target_classes[i]
#             labels_tensor[image_idx+i] = label
#             imgs[image_idx+i] = in_image
#         image_idx += len(target_classes)
#         if image_idx >= num_imgs:
#             break

# imgs = imgs[:image_idx]
# segmentations = segmentations[:image_idx]
# targets_tensor = targets_tensor[:image_idx]

# use_diffusion = False
# for method in [hps.method]:
#     if method.lower() == 'svces':
#         radii = np.array([150.])
#         attack_type = 'afw'
#         norm = 'L1.5'
#         stepsize = None
#         steps = 75
#     elif method.lower() == 'apgd':
#         radii = np.array([12.])
#         attack_type = 'apgd'
#         norm = 'L2' 
#         stepsize = None
#         steps = 75
#     elif method.lower() == 'dvces':
#         attack_type = 'diffusion'
#         radii = np.array([0.])
#         norm = 'L2'
#         steps = 150
#         stepsize = None
#         use_diffusion=True
#     else:
#         raise NotImplementedError()


#     attack_config = create_attack_config(eps=radii[0], steps=steps, stepsize=stepsize, norm=norm, momentum=0.9,
#                                          pgd=attack_type)

#     num_classes = len(in_labels)
#     if attack_type == 'diffusion':
#         img_dimensions = (3, 256, 256)
#     else:
#         img_dimensions = imgs.shape[1:]
#     num_targets = 1
#     num_radii = len(radii)
#     num_imgs = len(imgs)

#     with torch.no_grad():

#         model_bs = bs
#         dir = f'{out_dir}/{imagenet_mode}/{norm}_{hps.l2_sim_lambda}_l1_{hps.l1_sim_lambda}_l{hps.lp_custom}_{hps.lp_custom_value}_classifier_{hps.classifier_type}_{hps.second_classifier_type}_{hps.third_classifier_type}_{hps.classifier_lambda}_reg_lpips_{hps.lpips_sim_lambda}_example_{hps.timestep_respacing}_steps_skip_{hps.skip_timesteps}_start_{hps.gen_type}_deg_{str(hps.deg_cone_projection)}_s_{str(hps.seed)}{"_bl" if hps.use_blended else ""}_wid_{hps.world_id}_{hps.world_size}_{hps.method}/'
#         pathlib.Path(dir).mkdir(parents=True, exist_ok=True)

#         out_imgs = torch.zeros((num_imgs, num_targets, num_radii) + img_dimensions)
#         out_probabilities = torch.zeros((num_imgs, num_targets, num_radii, num_classes))
#         in_probabilities = torch.zeros((num_imgs, num_targets, num_radii, num_classes))
#         model_original_probabilities = torch.zeros((num_imgs, num_classes))

#         n_batches = int(np.ceil(num_imgs / model_bs))


#         if use_diffusion or attack_config['pgd'] in ['afw', 'apgd']:
#             if use_diffusion:
#                 att = ConceptDiffusionAttack(hps)
#             else:
#                 loss = 'log_conf' if attack_config['pgd'] == 'afw' else 'ce-targeted-cfts'
#                 print('using loss', loss)
#                 model = None
#                 att = get_adversarial_attack(attack_config, model, loss, num_classes,
#                                              args=hps, Evaluator=Evaluator)
#             if att.args.second_classifier_type != -1:
#                 print('setting model to second classifier')
#                 model = att.second_classifier
#             else:
#                 model = att.classifier
#         else:
#             model = None

#         for batch_idx in trange(n_batches, desc=f'Batches progress'):
#             sleep(0.1)
#             batch_start_idx = batch_idx * model_bs
#             batch_end_idx = min(num_imgs, (batch_idx + 1) * model_bs)

#             batch_data = imgs[batch_start_idx:batch_end_idx, :]
#             batch_targets = targets_tensor[batch_start_idx:batch_end_idx]
#             print('batch segmentations before', segmentations.shape)
#             batch_segmentations = segmentations[batch_start_idx:batch_end_idx, :]
#             print('batch segmentations after', batch_segmentations.shape)
#             target_idx = 0

#             orig_out = model(batch_data)
#             with torch.no_grad():
#                 orig_confidences = torch.softmax(orig_out, dim=1)
#                 model_original_probabilities[batch_start_idx:batch_end_idx, :] = orig_confidences.detach().cpu()

#             for radius_idx in range(len(radii)):
#                 batch_data = batch_data.to(device)
#                 batch_targets = batch_targets.to(device)


#                 if not use_diffusion:
#                     att.eps = radii[radius_idx]

#                 if use_diffusion:
#                     batch_adv_samples_i = att.perturb(batch_data,
#                                                             batch_targets, dir)[
#                         0].detach()
#                 else:
#                     if attack_config['pgd'] in ['afw']:

#                         batch_adv_samples_i = att.perturb(batch_data,
#                                                                 batch_targets,
#                                                                 targeted=True).detach()



#                     else:
#                         batch_adv_samples_i = att.perturb(batch_data,
#                                                                 batch_targets,
#                                                                 best_loss=True)[0].detach()
#                 out_imgs[batch_start_idx:batch_end_idx, target_idx, radius_idx,
#                 :] = batch_adv_samples_i.cpu().detach()

#                 batch_model_out_i = model(batch_adv_samples_i)
#                 batch_model_in_i = model(batch_data)
#                 batch_probs_i = torch.softmax(batch_model_out_i, dim=1)
#                 batch_probs_in_i = torch.softmax(batch_model_in_i, dim=1)

#                 out_probabilities[batch_start_idx:batch_end_idx, target_idx, radius_idx,
#                 :] = batch_probs_i.cpu().detach()
#                 in_probabilities[batch_start_idx:batch_end_idx, target_idx, radius_idx,
#                 :] = batch_probs_in_i.cpu().detach()

#             if (batch_idx + 1) % hps.plot_freq == 0 or batch_idx == n_batches-1:
#                 data_dict = {}

#                 data_dict['gt_imgs'] = imgs[:batch_end_idx]
#                 data_dict['gt_labels'] = labels_tensor[:batch_end_idx]
#                 data_dict['segmentations'] = segmentations[:batch_end_idx]
#                 data_dict['targets'] = targets_tensor[:batch_end_idx]
#                 data_dict['counterfactuals'] = out_imgs[:batch_end_idx]
#                 data_dict['out_probabilities'] = out_probabilities[:batch_end_idx]
#                 data_dict['in_probabilities'] = in_probabilities[:batch_end_idx]
#                 data_dict['radii'] = radii
#                 torch.save(data_dict, os.path.join(dir, f'{num_imgs}.pth'))
#                 _plot_counterfactuals(dir, imgs[:batch_end_idx], labels_tensor, segmentations[:batch_end_idx],
#                                       targets_tensor[:batch_end_idx],
#                                       out_imgs[:batch_end_idx], out_probabilities[:batch_end_idx], in_probabilities[:batch_end_idx], radii, in_labels, filenames=None, num_plot_imgs=len(imgs[:batch_end_idx]))
