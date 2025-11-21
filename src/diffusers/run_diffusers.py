import os
import yaml
import shutil
import copy
import json
import hydra
import pathlib
import random
import numpy as np
from omegaconf import DictConfig, OmegaConf, open_dict
from torchvision.transforms.functional import center_crop, resize, to_tensor
from torchvision.utils import save_image
from torchvision.models import VGG16_BN_Weights, ResNet18_Weights, ViT_B_16_Weights
import torch
from diffusers import StableDiffusionImg2ImgPipeline, DiffusionPipeline

from ldce.data.imagenet_classnames import name_map, openai_imagenet_classes
from run_ldce_baseline import set_seed, get_dataset, get_classifier, blockPrint
from pipeline import ModifiedStableDiffusionImg2ImgPipeline
from kandinsky_pipeline import ModifiedKandinskyImg2ImgPipeline

from src.concept_conditioning import compute_concept_conditioning

def get_transforms(cfg):

    model_name = cfg.classifier_model.name

    if model_name == "vgg16_bn":
        weights = VGG16_BN_Weights.IMAGENET1K_V1
    elif model_name == "resnet18":
        weights = ResNet18_Weights.IMAGENET1K_V1
    elif model_name.startswith("vit"):
        weights = ViT_B_16_Weights.IMAGENET1K_V1
    else:
        raise NotImplementedError

    transforms = weights.transforms()
    
    return transforms


@hydra.main(version_base=None, config_path="../../configs/ldce", config_name="v1")
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

    batch_size = cfg.data.batch_size
    shuffle = cfg.get("shuffle", False)

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

    classifier_model = get_classifier(cfg, device)
    classifier_model.to(device).eval()

    clf_transform = get_transforms(cfg)

    # model_id_or_path = "runwayml/stable-diffusion-v1-5"
    # stabilityai/stable-diffusion-3-medium 
    # stabilityai/stable-diffusion-xl-base-1.0

    # diffusion_type = "kandinsky"

    if "CelebA" in cfg.data._target_:
        shutil.rmtree("/results/models/ldm-celebahq-256")
        local_model_path = "/results/models/ldm-celebahq-256"
        if os.path.isdir(local_model_path):
            pipe = DiffusionPipeline.from_pretrained(local_model_path)
        else:
            model_id_or_path = "CompVis/ldm-celebahq-256"
            pipe = DiffusionPipeline.from_pretrained(model_id_or_path)
            pipe.save_pretrained(local_model_path, variant="fp16")
        pipe = pipe.to(device)

    elif cfg.diffusion_type == "stable_diffusion":
        # local_model_path = "/results/models/stable-diffusion-v1-4"
        local_model_path = "/results/models/stable-diffusion-2-1-base"
        if os.path.isdir(local_model_path):
            # pipe = ModifiedStableDiffusionImg2ImgPipeline.from_pretrained(local_model_path, variant="fp16", torch_dtype=torch.float16)
            pipe = StableDiffusionImg2ImgPipeline.from_pretrained(local_model_path, variant="fp16", torch_dtype=torch.float16)
        else:
            # model_id_or_path = "CompVis/stable-diffusion-v1-4"
            model_id_or_path = "stabilityai/stable-diffusion-2-1-base"
            # model_id_or_path = "CompVis/stable-diffusion-v1-2"
            pipe = ModifiedStableDiffusionImg2ImgPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float16)     #, height=256, width=256) #, torch_dtype=torch.float16)
            pipe.save_pretrained(local_model_path, variant="fp16")
        pipe = pipe.to(device)
        # pipe.enable_model_cpu_offload()

    elif cfg.diffusion_type == "kandinsky":
        # cpu_device = torch.device('cpu')
        local_model_path = "/results/models/kandinsky-3"
        if os.path.isdir(local_model_path):
            pipe = ModifiedKandinskyImg2ImgPipeline.from_pretrained(local_model_path, variant="fp16", torch_dtype=torch.float16, use_safetensors=True)
        else:
            model_id_or_path = "kandinsky-community/kandinsky-3"
            pipe = ModifiedKandinskyImg2ImgPipeline.from_pretrained(model_id_or_path, variant="fp16", torch_dtype=torch.float16, use_safetensors=True)
            pipe.save_pretrained(local_model_path, variant="fp16")
        # pipe.enable_model_cpu_offload()
        pipe = pipe.to(device)
    # pipe = ModifiedStableDiffusionImg2ImgPipeline.from_pretrained("/Data/models/stable-diffusion-v1-5", use_safetensors=True)

    print(pipe.device)

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
    if len(dataset) > 1000:
        dataset = torch.utils.data.Subset(dataset, np.arange(1000))
    print(type(dataset))
    print("dataset length: ", len(dataset))
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)

    if "ImageNet" in cfg.data._target_:
        i2h = name_map
    elif "CelebAHQDataset" in cfg.data._target_ or "CelebA" in cfg.data._target_:
        # query label 31 (smile): label=0 <-> no smile and label=1 <-> smile
        # query label 39 (age): label=0 <-> old and label=1 <-> young
        assert cfg.data.query_label in [2, 4, 31, 39]
        if 31 == cfg.data.query_label:
            i2h = ["no smile", "smile"]
        elif 39 == cfg.data.query_label:
            i2h = ["old", "young"]
        elif 2 == cfg.data.query_label:
            i2h = ["not attractive", "attractive"]
        elif 4 == cfg.data.query_label:
            i2h = ["not bald", "bald"]
        else:
            raise NotImplementedError
    elif "Flowers102" in cfg.data._target_:
        with open("data/flowers_idx_to_label.json", "r") as f:
            flowers_idx_to_classname = json.load(f)
        flowers_idx_to_classname = {int(k)-1: v for k, v in flowers_idx_to_classname.items()}
        i2h = flowers_idx_to_classname
    elif "OxfordIIIPets" in cfg.data._target_:
        with open("data/pets_idx_to_label.json", "r") as f:
            pets_idx_to_classname = json.load(f)
        i2h = {int(k): v for k, v in pets_idx_to_classname.items()}
    elif "CUB" in cfg.data._target_:
        with open("data/cub_idx_to_label.json", "r") as f:
            cub_idx_to_classname = json.load(f)
        i2h = {int(k): v for k, v in cub_idx_to_classname.items()}
    else:
        raise NotImplementedError

    if "ImageNet" in cfg.data._target_:
        num_classes = 1000
        with open('data/synset_closest_idx.yaml', 'r') as file:
            synset_closest_idx = yaml.safe_load(file)
    elif "Flowers102" in cfg.data._target_:
        with open("data/flowers_closest_indices.json") as file:
            closest_indices = json.load(file)
        closest_indices = {int(k):v for k,v in closest_indices.items()}
    elif "OxfordIIIPets" in cfg.data._target_:
        num_classes = 37
        with open("data/pets_closest_indices.json") as file:
            closest_indices = json.load(file)
        closest_indices = {int(k):v for k,v in closest_indices.items()}
    elif "CelebA" in cfg.data._target_:
        num_classes = 40
    elif "CUB" in cfg.data._target_:
        num_classes = 200

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
            elif "CelebADataset" in cfg.data._target_:
                tgt_classes = (1 - label).type(torch.float32)
            elif "CelebAHQDataset" in cfg.data._target_:
                tgt_classes = (1 - label).type(torch.float32)
            elif "Flowers102" in cfg.data._target_ or "OxfordIIIPets" or "CUB" in cfg.data._target_:
                tgt_classes = torch.tensor([closest_indices[unique_data_idx[l].item()*cfg.data.num_shards + cfg.data.shard][0] for l in range(label.shape[0])]).to(device)
            else:
                raise NotImplementedError
        
        # if "CelebA" not in cfg.data._target_:
        if "ImageNet" in cfg.data._target_:
            if "counterfactual_target" in cfg:
                if cfg.counterfactual_target == "baseline":
                    if cfg.classifier_model.name.startswith("vit"):
                        tgt_json = os.path.join('/results/counterfactuals/', 'concept_selection', f'conditions_vit_b_16_{cfg.counterfactual_target}.json')
                    else:
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

        print(f'Input min {torch.min(image):.4f} max {torch.max(image):.4f}')
        
        # clf_image = center_crop(resize(image, (256, 256), antialias=False),(224, 224))
        clf_image = clf_transform(image)

        print(clf_image.size())

        conditions = None
        if cfg.concept_conditioning:
            if cfg.spatial:
                conditions, concept_conds, concept_diff = compute_concept_conditioning(classifier_model, clf_image, tgt_classes, cfg.concept_layer, num_concepts=cfg.num_concepts, num_classes=num_classes, spatial=cfg.spatial, cond_option=cfg.cond_option)
            else:
                conditions, concept_conds, concept_diff = compute_concept_conditioning(classifier_model, clf_image, tgt_classes, cfg.concept_layer, num_concepts=cfg.num_concepts, num_classes=num_classes, cond_option=cfg.cond_option)

        print(conditions)
        #get classifcation prediction
        with torch.inference_mode():
            #with precision_scope():
            if "classifier_wrapper" in cfg.classifier_model and cfg.classifier_model.classifier_wrapper:
                logits = classifier_model(clf_image)
            else:
                # print(f'Img min {torch.min(clf_image)} max {torch.max(clf_image)}')
                logits = classifier_model(clf_image)
                print(logits.size())
                # logits = sampler.get_classifier_logits(_unmap_img(image)) #converting to -1, 1
                # raise NotImplementedError
            # TODO: handle binary vs multi-class
            if "ImageNet" in cfg.data._target_ or "OxfordIIIPets" in cfg.data._target_ or "Flowers102" in cfg.data._target_ or "CUB" in cfg.data._target_: # multi-class
                in_class_pred = logits.argmax(dim=1)
                in_confid = logits.softmax(dim=1).max(dim=1).values
                in_confid_tgt =  logits.softmax(dim=1)[torch.arange(batch_size), tgt_classes]
            else: # binary
                in_class_pred = (logits >= 0).type(torch.int8)
                in_confid = torch.where(logits >= 0, logits.sigmoid(), 1 - logits.sigmoid())
                in_confid_tgt =  torch.where(tgt_classes.to(device) == 0, 1 - logits.sigmoid(), logits.sigmoid())

            print(f"in class_pred: {in_class_pred} {in_confid}")
        
        for j, l in enumerate(label):
            print(f"converting {i} from : {i2h[l.item()]} to: {i2h[int(tgt_classes[j].item())]}")
        
        init_image = image.clone() #image.repeat(n_samples_per_class, 1, 1, 1).to(device)


        # init_latent = model.get_first_stage_encoding(
            # model.encode_first_stage(_unmap_img(init_image)))  # move to latent space

        text_conditional = True
        
        if text_conditional: # text-conditional
            if "ImageNet" in cfg.data._target_:
                prompts = [f"a photo of a {openai_imagenet_classes[idx.item()]}." for idx in tgt_classes]
                # negative_prompts = [f"a photo of a {openai_imagenet_classes[idx.item()]}." for idx in label]
                negative_prompts = ["" for idx in label]
                # prompts = [f"a photo with a {openai_imagenet_classes[idx.item()]}." for idx in tgt_classes]
                # prompts = [f"a {openai_imagenet_classes[idx.item()]}." for idx in tgt_classes]
            elif "CelebAHQDataset" in cfg.data._target_ or "CelebA" in cfg.data._target_:
                # query label 31 (smile): label=0 <-> no smile and label=1 <-> smile
                # query label 39 (age): label=0 <-> old and label=1 <-> young
                assert cfg.data.query_label in [2, 4, 31, 39]
                prompts = []
                negative_prompts = []
                for target in tgt_classes:
                    if cfg.data.query_label == 31 and target == 0:
                        attr = "frowning"   #"non-smiling"
                    elif cfg.data.query_label == 31 and target == 1:
                        attr = "smiling"
                    elif cfg.data.query_label == 39 and target == 0:
                        attr = "old"
                    elif cfg.data.query_label == 39 and target == 1:
                        attr = "young"
                    elif cfg.data.query_label == 2 and target == 0:
                        attr = "not attractive"
                    elif cfg.data.query_label == 2 and target == 1:
                        attr = "attractive"
                    elif cfg.data.query_label == 4 and target == 0:
                        attr = "not bald"
                    elif cfg.data.query_label == 4 and target == 1:
                        attr = "bald"
                    else:
                        raise NotImplementedError
                    prompts.append(f"a photo of a {attr} person")
                    negative_prompts.append("")
            elif "OxfordIIIPets" in cfg.data._target_:
                # prompts following https://github.com/openai/CLIP/blob/main/data/prompts.md
                prompts = [f"a photo of a {i2h[idx.item()]}, a type of pet." for idx in tgt_classes]
                negative_prompts = ["" for idx in tgt_classes]
            elif "Flowers102" in cfg.data._target_:
                # prompts following https://github.com/openai/CLIP/blob/main/data/prompts.md
                prompts = [f"a photo of a {i2h[idx.item()]}, a type of flower." for idx in tgt_classes]
                negative_prompts = ["" for idx in tgt_classes]
            elif "CUB" in cfg.data._target_:
                # prompts following https://github.com/openai/CLIP/blob/main/data/prompts.md
                prompts = [f"a photo of a {i2h[idx.item()]}, a type of bird." for idx in tgt_classes]
                negative_prompts = ["" for idx in tgt_classes]
            else:
                raise NotImplementedError
        else:
            prompts = None

        print(f'Prompts: {prompts}')

        inference_steps = cfg.ddim_steps # int(cfg.strength * cfg.ddim_steps)

        # init_image = init_image.resize((512, 512))
        if cfg.diffusion_type == "kandinsky":
            init_image_scaled = init_image * 2 - 1.
        else:
            init_image_scaled = init_image

        out = pipe(prompt=prompts,
                image=init_image_scaled,
                num_inference_steps=inference_steps,
                strength=cfg.strength,
                guidance_scale=cfg.scale,
                negative_prompt=negative_prompts,
                classifier=classifier_model,
                clf_transform=clf_transform,
                tgt=tgt_classes,
                classifier_lambda=cfg.sampler.classifier_lambda,
                deg_cone_projection=cfg.sampler.deg_cone_projection,
                lp_custom=cfg.sampler.lp_custom,
                dist_lambda=cfg.sampler.dist_lambda,
                concept_conditioning=cfg.concept_conditioning,
                concept_conditions=conditions,
                spatial=cfg.spatial,
                uncondition_end=True)
        # gen_images = out.images
        # image = pipe(prompt=prompt, image=original_image, strength=0.3).images[0]

        all_samples = out.images
        all_probs = None    # out.probs

        with torch.inference_mode():
            if "classifier_wrapper" in cfg.classifier_model and cfg.classifier_model.classifier_wrapper:
                logits = classifier_model(all_samples[0])
            else:
                # logits = sampler.get_classifier_logits(_unmap_img(all_samples[0])) #converting to -1, 1 (it is converted back in the function)
                # print(f'Img min {torch.min(image)} max {torch.max(image)}')
                # print(type(all_samples))
                # print(type(all_samples[0]))
                # print(all_samples[0].size)
                # print(f'Generated Img min {np.min(all_samples[0]):.3f} max {torch.max(all_samples[0]):.3f}')
                gen_images = torch.stack([clf_transform(s) for s in all_samples]).to(device)
                # gen_images = torch.stack([to_tensor(sam) for sam in all_samples]).to(device)
                # gen_images = center_crop(resize(gen_images, (256, 256), antialias=False), (224, 224))
                # print(type(gen_images))
                print(f'is size 224? {gen_images.size()}')
                logits = classifier_model(gen_images)
            if "ImageNet" in cfg.data._target_ or "CUB" in cfg.data._target_ or "OxfordIIIPets" in cfg.data._target_ or "Flowers102" in cfg.data._target_ or "CUB" in cfg.data._target_: # multi-class
                out_class_pred = logits.argmax(dim=1)
                out_confid = logits.softmax(dim=1).max(dim=1).values
                out_confid_tgt = logits.softmax(dim=1)[torch.arange(batch_size), tgt_classes]
            else: # binary
                out_class_pred = (logits >= 0).type(torch.int8)
                out_confid = torch.where(logits >= 0, logits.sigmoid(), 1 - logits.sigmoid())
                out_confid_tgt =  torch.where(tgt_classes.to(device) == 0, 1 - logits.sigmoid(), logits.sigmoid())
            print(f"out class_pred: {out_class_pred} {out_confid}")
            print("out targets: ", tgt_classes) #.item())
            print(f"tgt confidence: {out_confid_tgt}")  #.item():.5f}")

        # Loop through your data and update the table incrementally
        for j in range(batch_size):
            # Generate data for the current row
            src_image = copy.deepcopy(init_image[j].cpu())
            # src_image = copy.deepcopy(sampler.init_images[j].cpu()) #all_samples[j][0])
            # gen_image = copy.deepcopy(all_samples[0][j].cpu())
            print(len(all_samples))
            gen_image = copy.deepcopy(all_samples[j])
            class_prediction = copy.deepcopy(all_probs[0][j]) if all_probs is not None else out_confid[j] # all_probs[j]
            
            source = i2h[label[j].item()]
            target = i2h[int(tgt_classes[j].item())]
            in_pred_cls = i2h[in_class_pred[j].item()]
            out_pred_cls = i2h[out_class_pred[j].item()]

            #diff =  (init_image - all_samples[j][1:])
            # diff = init_image[j]-all_samples[0][j]   
            # lp1 = int(torch.norm(diff, p=1, dim=-1).mean().cpu().numpy())
            # lp2 = int(torch.norm(diff, p=2, dim=-1).mean().cpu().numpy())
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
            }

            if "Flowers102" in cfg.data._target_ or "OxfordIIIPets" in cfg.data._target_ or "CUB" in cfg.data._target_:
                uidx = unique_data_idx[j].item()*cfg.data.num_shards + cfg.data.shard
            else:
                uidx = unique_data_idx[j].item()

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
            # save_image(gen_image.clip(0, 1), cf_save_path)
            gen_image.save(cf_save_path)
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
            
    return None


if __name__ == '__main__':
    main()
