import os
import sys
sys.path.append("./")
sys.path.append("./ldce")
sys.path.append("./data")
import h5py
import hydra
from omegaconf import OmegaConf, DictConfig, open_dict
import numpy as np
import random
import torch
import yaml

from ldce.data.imagenet_classnames import name_map

from src.concept_conditioning import compute_concept_conditioning
from src.sampling_helpers import disabled_train
from src.helpers.data_model_helpers import get_classifier, get_dataset, set_seed


@hydra.main(version_base=None, config_path="../configs/ldce", config_name="v1")
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
    classifier_model.to(device).eval()
    classifier_model.train = disabled_train

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

    conditioning_file = os.path.join(out_dir, f'conditioning_{cfg.classifier_model.name}.h5')

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
            elif "CelebAHQDataset" in cfg.data._target_:
                tgt_classes = (1 - label).type(torch.float32)
            elif "Flowers102" in cfg.data._target_ or "OxfordIIIPets" in cfg.data._target_:
                tgt_classes = torch.tensor([closest_indices[unique_data_idx[l].item()*cfg.data.num_shards + cfg.data.shard][0] for l in range(label.shape[0])]).to(device)
            else:
                raise NotImplementedError


        image = image.to(device) #squeeze()
        label = label.to(device) #.item() #squeeze()

        # for layer in ['features.37', 'features.40']:

        for cond_option in ['sumabs', 'sum', 'sumequal', 'absmean', 'abssum', 'absmax']:

        # Compute concept conditions
        # ToDo: add sampler.classifier_wrapper as parameter
            if spatial:
                conditions, concept_conds, concept_diff, grad = compute_concept_conditioning(classifier_model, image, tgt_classes, concept_layer, num_concepts=cfg.num_concepts, spatial=spatial, cond_option=cond_option, return_gradient=True)
            else:
                conditions, concept_conds, concept_diff, grad = compute_concept_conditioning(classifier_model, image, tgt_classes, concept_layer, num_concepts=cfg.num_concepts, cond_option=cond_option, return_gradient=True)

            # Save conditioning to file
            print(unique_data_idx)

            with h5py.File(conditioning_file, 'a', locking=False) as cfile:

                if 'image' in cfile:
                    num_existing = cfile['image'].shape[0]
                else:
                    num_existing = 0
                num_new = image.size()[0]
                image_shape = image.size()[1:]
                grad_shape = grad.shape[1:]

                cfile.require_dataset(
                    'image',
                    shape=(0,) + image_shape,
                    maxshape=(None,) + image_shape,
                    dtype='float32',
                    chunks=True,
                )
                cfile.require_dataset(
                    'label',
                    shape=(0,),
                    maxshape=(None,),
                    dtype='uint16',
                    chunks=True,
                )
                cfile.require_dataset(
                    'cf_label',
                    shape=(0,),
                    maxshape=(None,),
                    dtype='uint16',
                    chunks=True,
                )

                cfile.require_group(concept_layer)
                layer_group = cfile[concept_layer]

                layer_group.require_dataset(
                    'gradient',
                    shape=(0,) + grad_shape,
                    maxshape=(None,) + grad_shape,
                    dtype='float32',
                    chunks=True,
                )
                cond_group = layer_group.require_group(cond_option)
                cond_group.require_dataset(
                    'concepts',
                    shape=(0,) + (cfg.num_concepts,),
                    maxshape=(None,) + (cfg.num_concepts,),
                    dtype='uint32',
                    chunks=True,
                )
                cond_group.require_dataset(
                    'diffs',
                    shape=(0,) + (cfg.num_concepts,),
                    maxshape=(None,) + (cfg.num_concepts,),
                    dtype='float32',
                    chunks=True,
                )

                if num_existing < unique_data_idx.numpy()[-1]:
                    cfile['image'].resize(num_existing + num_new, axis=0)
                    cfile['label'].resize(num_existing + num_new, axis=0)
                    cfile['cf_label'].resize(num_existing + num_new, axis=0)
                    cfile[concept_layer]['gradient'].resize(num_existing + num_new, axis=0)
                    cfile[concept_layer][cond_option]['concepts'].resize(num_existing + num_new, axis=0)
                    cfile[concept_layer][cond_option]['diffs'].resize(num_existing + num_new, axis=0)

                    cfile['image'][num_existing:] = image.cpu().numpy()
                    cfile['label'][num_existing:] = label.cpu().numpy()
                    cfile['cf_label'][num_existing:] = tgt_classes.cpu().numpy()
                    cfile[concept_layer]['gradient'][num_existing:] = grad
                    cfile[concept_layer][cond_option]['concepts'][num_existing:] = concept_conds[concept_layer]
                    cfile[concept_layer][cond_option]['diffs'][num_existing:] = concept_diff
                
                else:
                    if cfile[concept_layer][cond_option]['concepts'].shape[0] < unique_data_idx.numpy()[-1]:
                        cfile[concept_layer][cond_option]['concepts'].resize(num_existing + num_new, axis=0)
                        cfile[concept_layer][cond_option]['diffs'].resize(num_existing + num_new, axis=0)
                    cfile[concept_layer][cond_option]['concepts'][unique_data_idx.numpy()] = concept_conds[concept_layer]
                    cfile[concept_layer][cond_option]['diffs'][unique_data_idx.numpy()] = concept_diff

if __name__ == '__main__':
    main()