""" Analyse the concepts for single samples. """
import hydra
import numpy as np
import torch
import zennit
import matplotlib.pyplot as plt

from run_ldce_baseline import get_classifier, get_dataset
from src.concept_cc_ddim import MaskHook, batch_map, _register_mask_fn


def store_hook(module, input, output):
    # set the current module's attribute 'output' to the its tensor
    module.output = output
    # keep the output tensor gradient, even if it is not a leaf-tensor
    output.retain_grad()

def grad_computation(model, image, target, layer_name=None, attr=False):

    layer = None
    if layer_name:
        for name, lay in model.named_modules():
            print(f'{name} : {type(lay)}')
            if name == layer_name:
                layer = lay

        assert layer

    target_tensor = torch.eye(1000)[target]
    target_tensor = target_tensor.to(image.device)

    cmp = None
    if attr:
        canonizers = [
            zennit.torchvision.VGGCanonizer(),
            zennit.canonizers.SequentialMergeBatchNorm()]

        cmp = zennit.composites.EpsilonPlus(canonizers=canonizers)

    # Get gradient in specified layer
    with zennit.attribution.Gradient(model=model, composite=cmp) as attributor:

        handles = []
        # for n,l in attributor.named_modules():
        #     if n == "layer_name":
        if layer:
            handles.append(layer.register_forward_hook(store_hook))

        # compute the model output and gradient
        output, attr = attributor(image, target_tensor)

    for handle in handles:
        handle.remove()

    # print the gradient tensors for demonstration
    if layer:
        grad = layer.output.grad
        return grad.cpu().numpy()

    # Extract most important channels
    # channel_grads = grad.cpu().sum((2,3)).numpy()
    # mean_channel_grads = grad.cpu().abs().mean((2,3)).numpy()

    # conditions = [np.argsort(np.abs(cg))[-num_concepts:] for cg in channel_grads]
    # mconditions = [np.argsort(cg)[-num_concepts:] for cg in mean_channel_grads]
    # conditions = np.array(conditions)

    return attr.cpu().numpy()

def spatial_map(batch_id, concept_ids, layer_name=None):

    def mask_fct(grad):
        mask = torch.from_numpy(concept_ids).to(grad.device)

        grad = grad * mask

        return grad

    return mask_fct

def concept_attribution(model, image, target, layer_name, conditions):

    layer = None
    if layer_name:
        for name, lay in model.named_modules():
            print(f'{name} : {type(lay)}')
            if name == layer_name:
                layer = lay

        assert layer

    target_tensor = torch.eye(1000)[target]
    target_tensor = target_tensor.to(image.device)

    canonizers = [
        zennit.torchvision.VGGCanonizer(),
        zennit.canonizers.SequentialMergeBatchNorm()]

    cmp = zennit.composites.EpsilonPlus(canonizers=canonizers)

    hook_map, y_targets = {}, []

    # for key in concept_conditions.keys():
    if layer_name not in hook_map:
        hook_map[layer_name] = MaskHook([])

    if isinstance(conditions, np.ndarray):
        _register_mask_fn(hook_map[layer_name], spatial_map, 0, conditions, layer_name)
    else:
        _register_mask_fn(hook_map[layer_name], batch_map, 0, conditions, layer_name)

    name_map = [([name], hook) for name, hook in hook_map.items()]
    mask_composite = zennit.composites.NameMapComposite(name_map)

    inp = image[None]

    with mask_composite.context(model), cmp.context(model) as modified:
    # with mask_composite.context(self.classifier) as modified:

        if not inp.requires_grad:
            inp.requires_grad = True

        output = modified(inp)

        attr, = torch.autograd.grad(
            (output,),
            (inp,),
            grad_outputs=(target_tensor[None],),
            create_graph=False,
            retain_graph=False,
        )

    return attr.cpu().numpy()


def analyze_concepts(classifier_model, image, label, tgt_classes, bn):

    # layer_name = "features.26"
    layer_name = 'features.37'

    # get activations

    # get gradient
    grad = grad_computation(classifier_model, image, tgt_classes, layer_name=layer_name, attr=False)

    conditions = grad.sum((2,3))
    conditions = [np.argsort(np.abs(cg))[-6:] for cg in conditions]

    print(grad.shape)

    spatial_condition_maps = []

    for s, sample_conditions in enumerate(conditions):
        for condition in sample_conditions:

            x, y = np.meshgrid(np.arange(grad.shape[2]), np.arange(grad.shape[3]))
            z = grad[0, condition, :, :]

            filtered_grad = grad[0, condition, :, :].copy()
            th = np.max(np.abs(filtered_grad)) * 0.6                    # randomly chosen 0.6
            filtered_grad[np.abs(filtered_grad) < th] = 0

            spatial_condition_maps.append(np.array(np.abs(filtered_grad) >= th))

            fig = plt.figure()
        
            # syntax for 3-D plotting
            ax = plt.axes(projection='3d')
            
            # syntax for plotting
            # ax.plot_wireframe(x, y, z, color ='green')
            # ax.plot_surface(x, y, z, cmap='viridis', edgecolor='green')
            ax.plot_surface(x, y, z, cmap='cool', alpha=0.6)
            ax.set_title(f'Gradient in spatial dimensions for concept {condition}')
            # plt.show()
            plt.savefig(f'/results/counterfactuals/gradient/spatial_{bn*len(conditions) + s}_{layer_name}_{condition}.png')
            plt.close()

            fig = plt.figure()
        
            # syntax for 3-D plotting
            ax = plt.axes(projection='3d')
            
            # syntax for plotting
            # ax.plot_wireframe(x, y, z, color ='green')
            # ax.plot_surface(x, y, z, cmap='viridis', edgecolor='green')
            ax.plot_surface(x, y, filtered_grad, cmap='cool', alpha=0.6)
            ax.set_title(f'Gradient in spatial dimensions for concept {condition}')
            # plt.show()
            plt.savefig(f'/results/counterfactuals/gradient/spatial_{bn*len(conditions) + s}_{layer_name}_{condition}_filtered.png')
            plt.close()

    # get attribution
    attrs = grad_computation(classifier_model, image, tgt_classes, attr=True)
    print(attrs.shape)

    for a, attr in enumerate(attrs):

        expl = zennit.image.imgify(attr.sum(axis=0), symmetric=True, cmap='coldnhot')
        expl.save(f'/results/counterfactuals/gradient/expl_{bn*len(attrs) + a}.png')

    for i, (img, tgt) in enumerate(zip(image, tgt_classes)):

        for c, cond in enumerate(conditions[i]):
            attr = concept_attribution(classifier_model, img, tgt, layer_name, [cond])

            cexpl = zennit.image.imgify(attr[0].sum(axis=0), symmetric=True, cmap='coldnhot')
            cexpl.save(f'/results/counterfactuals/gradient/concept_expl_{bn*len(tgt_classes)+ i}_{layer_name}_{cond}.png')

            attr = concept_attribution(classifier_model, img, tgt, layer_name, spatial_condition_maps[i*6 + c])

            cexpl = zennit.image.imgify(attr[0].sum(axis=0), symmetric=True, cmap='coldnhot')
            cexpl.save(f'/results/counterfactuals/gradient/concept_expl_{bn*len(tgt_classes)+ i}_{layer_name}_{cond}_spatial.png')


@hydra.main(version_base=None, config_path="../configs/ldce", config_name="v1")
def main(cfg) -> None:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu") # there seems to be a CUDA/autograd instability in gradient computation
    print(f"using device: {device}")

    # model = get_model(cfg_path=cfg.diffusion_model.cfg_path, ckpt_path = cfg.diffusion_model.ckpt_path).to(device).eval()
    
    classifier_model = get_classifier(cfg, device)
    classifier_model.to(device).eval()
    # classifier_model.train = disabled_train

    dataset = get_dataset(cfg, last_data_idx=0)
    dataset = torch.utils.data.Subset(dataset, np.arange(20))

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=cfg.data.batch_size, shuffle=False, num_workers=4)


    # concept_layer = cfg.concept_layer       # "backbone.features.29"

    for i, batch in enumerate(data_loader):

        if "return_tgt_cls" in cfg.data and cfg.data.return_tgt_cls:
            image, label, tgt_classes, unique_data_idx = batch
            tgt_classes = tgt_classes.to(device)

        image = image.to(device) #squeeze()
        label = label.to(device) #.item() #squeeze()

        analyze_concepts(classifier_model, image, label, tgt_classes, i)

        # compute conditioning
        # conditions = compute_concept_conditioning(classifier_model, image, tgt_classes, concept_layer, num_concepts=cfg.num_concepts)

        # compute explanations


    # Compute concept conditions
    # conditions = compute_concept_conditioning(classifier_model, image, tgt_classes, concept_layer, num_concepts=cfg.num_concepts)


    # compare_classes(cfg)


if __name__ == '__main__':
    main()
