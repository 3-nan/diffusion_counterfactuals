
import os
import sys
from PIL import Image
import torch
from tqdm import tqdm
import zennit
import hydra

sys.path.append('./')
sys.path.append('./ldce')
from src.helpers.data_model_helpers import get_classifier
from ldce.data.imagenet_classnames import name_map


def get_target_tensor(target, name_map, device):

    # target = data['target']
    class_target = list(name_map.keys())[list(name_map.values()).index(target)]
    class_target = torch.tensor([class_target], device=device)

    class_target = torch.eye(1000, device=device)[class_target]

    return class_target

def compute_explanation(classifier_model, img, target, model_name):
    """ Compute the explanation. """

    # target = torch.eye(1000)[target]

    pred = classifier_model(img)

    if model_name.startswith('resnet'):
        canonizer = zennit.torchvision.ResNetCanonizer()
    elif model_name.startswith('vgg'):
        canonizer = zennit.torchvision.VGGCanonizer()

    composite = zennit.composites.EpsilonPlus(canonizers=[canonizer])

    # choose a target class for the attribution (label 437 is lighthouse)
    # target = torch.eye(1000)[[437]]

    # create the attributor, specifying model and composite
    with zennit.attribution.Gradient(model=classifier_model, composite=composite) as attributor:
        # compute the model output and attribution
        output, attribution = attributor(img, target)

    relevance = attribution.sum(1).cpu()

    # create an image of the visualize attribution
    expl = zennit.image.imgify(relevance, symmetric=True, cmap='coldnhot')

    return expl


@hydra.main(version_base=None, config_path="../../configs/ldce", config_name="v1")
def run_explanation_computation(cfg):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # base_acts = get_representations(cfg, act_base_file, attribute=attribute)
    # print(base_acts.shape)

    # Load model
    classifier_model = get_classifier(cfg, device)
    classifier_model.to(device).eval()

    # out_size = 256
    # transform_list = [
    #     transforms.Resize((out_size, out_size)),
    #     transforms.ToTensor()
    # ]
    # transform = transforms.Compose(transform_list)

    os.makedirs(os.path.join(cfg.output_dir, "explanations"), exist_ok=True)
    os.chmod(os.path.join(cfg.output_dir, "explanations"), 0o777)

    for i in tqdm(range(0, 50)):

        # load counterfactual
        # img = Image.open(os.path.join(cfg.output_dir, 'bucket_0_10/counterfactual', f'{str(i).zfill(5)}.png'))
        # img = Image.open(os.path.join(cfg.output_dir, 'bucket_0_10/counterfactual', f'{str(i).zfill(5)}.png'))
        # img = transform(img)

        # img = tf.center_crop(img, 224)
        # img = normalize(img)
        # img = img[None].to(device)

        # load pth file
        pth_file = os.path.join(cfg.output_dir, f"bucket_0_10/{str(i).zfill(5)}.pth")
        data = torch.load(pth_file, map_location="cpu")

        # conditions = data['conditions']
        # print(conditions)

        class_source = get_target_tensor(data['source'], name_map, device)

        class_target = get_target_tensor(data['target'], name_map, device)

        # target = data['target']
        # class_target = list(name_map.keys())[list(name_map.values()).index(target)]
        # class_target = torch.tensor([class_target], device=device)
        # print(class_target)

        orig_img = data['image'][None].to(device)
        # print(orig_img.size())

        gen_img = data['gen_image'][None].to(device)

        # pred = classifier_model(orig_img)

        expl = compute_explanation(classifier_model, orig_img, class_source, cfg.classifier_model.name)
        c_expl = compute_explanation(classifier_model, gen_img, class_target, cfg.classifier_model.name)

        expl.save(os.path.join(cfg.output_dir, f'explanations/{str(i).zfill(5)}_orig_expl.png'))
        c_expl.save(os.path.join(cfg.output_dir, f'explanations/{str(i).zfill(5)}_ce_expl.png'))

        # show the image
        # display(img)

if __name__ == '__main__':

    run_explanation_computation()
