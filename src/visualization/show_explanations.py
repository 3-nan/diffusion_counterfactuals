
import os
from PIL import Image
import torch
import zennit


def get_target_tensor(target, name_map, device):

    # target = data['target']
    class_target = list(name_map.keys())[list(name_map.values()).index(target)]
    class_target = torch.tensor([class_target], device=device)

    return class_target

def run_explanation_computation(cfg, act_base_file, attribute='activation'):

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

    for i in range(0, 50):

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
        print(class_target)

        orig_img = data['image'][None].to(device)
        print(orig_img.size())

        gen_img = data['gen_image'][None].to(device)

        pred = classifier_model(orig_img)

        composite = zennit.composites.EpsilonPlus()

        # choose a target class for the attribution (label 437 is lighthouse)
        # target = torch.eye(1000)[[437]]

        # create the attributor, specifying model and composite
        with zennit.attribution.Gradient(model=classifier_model, composite=composite) as attributor:
            # compute the model output and attribution
            output, attribution = attributor(orig_img, class_source)

        relevance = attribution.sum(1)

        # create an image of the visualize attribution
        expl = zennit.image.imgify(relevance, symmetric=True, cmap='coldnhot')
        expl.imsave('/results/....')

        # show the image
        # display(img)
