""" Visualize concepts. """
import os
import sys
sys.path.append("./")
sys.path.append("./ldce")
sys.path.append("./data")
import hydra
from hydra.utils import instantiate
import json
from omegaconf import OmegaConf, open_dict, DictConfig
import numpy as np
import random
import torch
import torchvision.transforms as T
from zennit.canonizers import SequentialMergeBatchNorm
from zennit.composites import EpsilonPlusFlat
import matplotlib.pyplot as plt
import yaml

from crp.attribution import CondAttribution
from crp.concepts import ChannelConcept
from crp.helper import get_layer_names
from crp.visualization import FeatureVisualization
from crp.image import vis_opaque_img    # plot_grid

from ldce.data.imagenet_classnames import name_map, openai_imagenet_classes

from src.sampling_helpers import disabled_train
from src.concept_conditioning import compute_concept_conditioning
from src.helpers.data_model_helpers import get_classifier
from src.helpers.concept_visualization import plot_grid

def set_seed(seed: int = 0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_dataset(cfg, last_data_idx: int = 0):
    if "ImageNet" in cfg.data._target_:
        out_size = 256
        transform_list = [
            T.Resize((out_size, out_size)),
            T.ToTensor()
        ]
        transform = T.Compose(transform_list)
        dataset = instantiate(cfg.data, start_sample=cfg.data.start_sample, end_sample=cfg.data.end_sample, transform=transform, restart_idx=last_data_idx)
    elif "Flowers102" in cfg.data._target_:
        transform = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor(),
        ])
        dataset = instantiate(
            cfg.data, 
            shard=cfg.data.shard, 
            num_shards=cfg.data.num_shards, 
            transform=transform, 
            restart_idx=last_data_idx
        )
    elif "OxfordIIIPets" in cfg.data._target_: # try running on 224x224 img
        def _convert_to_rgb(image):
            return image.convert('RGB')
        out_size = 256
        transform_list = [
            T.Resize((out_size, out_size)),
            # transforms.CenterCrop(out_size),
            _convert_to_rgb,
            T.ToTensor(),
        ]
        transform = T.Compose(transform_list)
        dataset = instantiate(
            cfg.data, 
            shard=cfg.data.shard, 
            num_shards=cfg.data.num_shards, 
            transform=transform, 
            restart_idx=last_data_idx
        )
    else:
        raise NotImplementedError
    return dataset

# def plot_concepts():



def visualize_concepts(cfg, dataset, model, layer, concepts, concept_diff, uidx, base_label, cf_label, attr="relevance"):

    if cfg.classifier_model.name == 'vgg16_bn' and "ImageNet" in cfg.data._target_:
        fv_path = '/results/counterfactuals/fv_imagenet_vgg16bn'
    else:
        if "ImageNet" in cfg.data._target_:
            fv_path = f'/results/counterfactuals/fv_imagenet_{cfg.classifier_model.name}'
        elif "Flowers" in cfg.data._target_:
            fv_path = f'/results/counterfactuals/fv_flowers_{cfg.classifier_model.name}'
        elif "Pets" in cfg.data._target_:
            fv_path = f'/results/counterfactuals/fv_pets_{cfg.classifier_model.name}'

    print(fv_path)

    attribution = CondAttribution(model)
    canonizers = [SequentialMergeBatchNorm()]
    composite = EpsilonPlusFlat(canonizers)

    cc = ChannelConcept()

    layer_names = get_layer_names(model, [torch.nn.Conv2d, torch.nn.Linear])
    layer_map = {layer : cc for layer in layer_names}

    preprocessing =  T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    fv = FeatureVisualization(attribution, dataset, layer_map, preprocess_fn=preprocessing, path=fv_path)

    # Visualize concepts
    print(concepts)

    ref_c = fv.get_max_reference(concepts, layer, attr, (0, 8), composite=composite, plot_fn=None)
    plot_grid(ref_c, figsize=(6, 9))
    plt.close()

    # plt.savefig(os.path.join('/results/counterfactuals/fv_images', f'{cfg.classifier_model.name}_{str(uidx).zfill(5)}_ref_{attr}_concept.png'))
    # plt.close()

    # ref_c = fv.get_max_reference(concepts, layer, attr, (0, 8), rf=True, composite=composite, plot_fn=vis_opaque_img)
    # plot_grid(ref_c, figsize=(6, 5), padding=False)
    # plt.savefig(os.path.join('/results/counterfactuals/fv_images', f'{cfg.classifier_model.name}_{str(uidx).zfill(5)}_ref_{attr}_concept_receptive.png'))
    # plt.close()

    ref_t_all = {}
    for concept, c_diff in zip(concepts, concept_diff):
        if c_diff > 0:
            ref_t = fv.get_stats_reference(concept, layer, [cf_label], attr, (0, 8), rf=True, composite=composite, plot_fn=vis_opaque_img)
        else:
            ref_t = fv.get_stats_reference(concept, layer, [base_label], attr, (0, 8), rf=True, composite=composite, plot_fn=vis_opaque_img)
        ref_t_all.update(ref_t)
    print(ref_t_all)
    print(concept_diff)
    plot_grid(ref_t_all, concept_diff=concept_diff, figsize=(6, 9), padding=False)
    if "ImageNet" in cfg.data._target_:
        plt.savefig(os.path.join('/results/counterfactuals/fv_images', f'{cfg.classifier_model.name}_{cfg.concept_layer}_{str(uidx).zfill(5)}_ref_{attr}_concept_class.svg'))
    elif "Flowers" in cfg.data._target_:
        plt.savefig(os.path.join('/results/counterfactuals/fv_images', f'flowers_{cfg.classifier_model.name}_{cfg.concept_layer}_{str(uidx).zfill(5)}_ref_{attr}_concept_class.svg'))
    elif "Pets" in cfg.data._target_:
        plt.savefig(os.path.join('/results/counterfactuals/fv_images', f'pets_{cfg.classifier_model.name}_{cfg.concept_layer}_{str(uidx).zfill(5)}_ref_{attr}_concept_class.svg'))
    plt.close()


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
    
    ref_cfg_dict = {
        'data': dict(cfg['data'])
    }
    if "ImageNet" in cfg.data._target_:
        ref_cfg_dict['data'].update({"return_tgt_cls": False})
        ref_cfg_dict['data']['start_sample'] = 10
        ref_cfg_dict['data']['end_sample'] = 50
    else:
        ref_cfg_dict['data'].update({"return_index": False})
    #     ref_cfg_dict['data'].update({"start_sample": 1})
    print(ref_cfg_dict["data"])

    # ref_cfg_dict = {
    #     'data': {
    #         '_target_': 'data.datasets.ImageNet',
    #         'root': '/Data/imagenet/val',
    #         'idx_to_tgt_cls_path': './ldce/data/image_idx_to_tgt.yaml',
    #         'split': 'val',
    #         'return_tgt_cls': False,
    #         # 'batch_size': 4
    #         'start_sample': 10,
    #         'end_sample': 50
    #         }
    # }
    ref_cfg = OmegaConf.create(ref_cfg_dict)
    ref_dataset = get_dataset(ref_cfg)
    print("ref dataset length: ", len(ref_dataset))

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
    elif "OxfordIIIPets" in cfg.data._target_:
        with open("data/pets_closest_indices.json") as file:
            closest_indices = json.load(file)
        closest_indices = {int(k):v for k,v in closest_indices.items()}

    concept_layer = cfg.concept_layer       # "backbone.features.29"
    spatial = cfg.spatial

    attr = "relevance"          # relevance     activation

    if not cfg.resume:
        torch.save({"last_data_idx": -1}, checkpoint_path)
    
    seed = cfg.seed if "seed" in cfg else 0
    set_seed(seed=seed)

    for i, batch in enumerate(data_loader):

        # if "fixed_seed" in cfg:
        #     set_seed(seed=cfg.get("seed", 0)) if cfg.fixed_seed else None
        #     seed = seed if cfg.fixed_seed else -1
            
        # if "return_tgt_cls" in cfg.data and cfg.data.return_tgt_cls:
        #     image, label, tgt_classes, unique_data_idx = batch
        #     tgt_classes = tgt_classes.to(device) #squeeze()
        # else:
        image, label, unique_data_idx = batch
        #     if "ImageNet" in cfg.data._target_:
        #         tgt_classes = torch.tensor([random.choice(synset_closest_idx[l.item()]) for l in label]).to(device)
        #     elif "CelebAHQDataset" in cfg.data._target_:
        #         tgt_classes = (1 - label).type(torch.float32)
        #     elif "Flowers102" in cfg.data._target_ or "OxfordIIIPets" in cfg.data._target_:
        #         tgt_classes = torch.tensor([closest_indices[unique_data_idx[l].item()*cfg.data.num_shards + cfg.data.shard][0] for l in range(label.shape[0])]).to(device)
        #     else:
        #         raise NotImplementedError


        # image = image.to(device) #squeeze()
        # label = label.to(device) #.item() #squeeze()


        # Compute concept conditions
        # ToDo: add sampler.classifier_wrapper as parameter
        # if spatial:
        #     conditions = compute_concept_conditioning(classifier_model, image, tgt_classes, concept_layer, num_concepts=cfg.num_concepts, spatial=spatial, cond_option=cfg.cond_option)
        # else:
        #     conditions = compute_concept_conditioning(classifier_model, image, tgt_classes, concept_layer, num_concepts=cfg.num_concepts, cond_option=cfg.cond_option)

        for j in range(batch_size):
            # Read conditions from file
            if "Flowers102" in cfg.data._target_ or "OxfordIIIPets" in cfg.data._target_:
                uidx = unique_data_idx[j].item()*cfg.data.num_shards + cfg.data.shard
            else:
                uidx = unique_data_idx[j].item()

            dict_save_path = os.path.join(out_dir, f'{str(uidx).zfill(5)}.pth')

            data_dict = torch.load(dict_save_path, map_location="cpu")

            # print(data_dict['source'])
            # print(label)
            source_pred = data_dict['in_pred']
            # print(name_map)
            # print(source_pred)
            # source_pred = list(name_map.keys())[list(name_map.values()).index(source_pred)]
            source_pred = list(i2h.keys())[list(i2h.values()).index(source_pred)]
            # class_source = torch.tensor([class_source], device=device)
            # print(source_pred)
            # print(i2h[source_pred])


            target = data_dict['target']
            # print(target)
            # class_target = list(name_map.keys())[list(name_map.values()).index(target)]
            class_target = list(i2h.keys())[list(i2h.values()).index(target)]
            # class_target = torch.tensor([class_target], device=device)
            # print(class_target)
            
            # raise ValueError

            # if "Flowers102" in cfg.data._target_:
            #     source_pred = source_pred - 1
            #     class_target = class_target - 1

            if 'conditions' in data_dict:
                conditions = data_dict['conditions']
                concept_diff = data_dict['concept_diff']
            else:
                print('Recomputing conditions...')
                # conditions = compute_concept_conditioning(classifier_model, image, tgt_classes, concept_layer, num_concepts=cfg.num_concepts, cond_option=cfg.cond_option)

                def acts_hook(module, input, output):
                    module.out = output

                layer_handle = None
                for n, m in classifier_model.named_modules():
                    if n == cfg.concept_layer:
                        layer_handle = m

                layer_handle.register_forward_hook(acts_hook)
                oc = classifier_model(image.to(device))
                # print(oc)
                
                # for n, m in classifier_model.named_modules():
                #     if n == cfg.concept_layer:
                cout = layer_handle.out
                # print(cout.size())

                concs = cout.detach().cpu().sum((2,3)).abs().numpy()

                conditions = [np.argsort(cg)[-cfg.num_concepts:] for cg in concs]
                conditions = np.array(conditions)[j]

                concept_diff = conditions #cout.detach().cpu().sum((2,3))[conditions]
                print(conditions)
        
            if conditions.shape[0] > 6:
                conditions = conditions[-6:]
                concept_diff = concept_diff[-6:]

            print(uidx)
            # print(conditions)
            # print(label[j].item())
            # print(tgt_classes[j].item())
            # print(type(uidx))

            # print(source_pred)

            # visualize_concepts(cfg, ref_dataset, classifier_model, concept_layer, conditions, concept_diff, uidx, label[j].item(), tgt_classes[j].item(), attr=attr)
            visualize_concepts(cfg, ref_dataset, classifier_model, concept_layer, conditions[::-1], concept_diff[::-1], uidx, source_pred, class_target, attr=attr)


if __name__ == '__main__':
    main()
