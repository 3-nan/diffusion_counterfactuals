import os
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm


def visualize_example():

    k = 5
    # model = 'vgg16bn_40'
    model = 'resnet18'

    base_dir = f'/results/counterfactuals/imagenet_{model.split("_")[0]}_baseline_attrtarget'
    output_dir = f'/results/counterfactuals/imagenet_{model}_concept_{k}'

    for uidx in tqdm(range(50)):

        orig_img = Image.open(os.path.join(output_dir, 'bucket_0_10', 'original', f'{str(uidx).zfill(5)}.png'))
        base_cf_img = Image.open(os.path.join(base_dir, 'bucket_0_10', 'counterfactual', f'{str(uidx).zfill(5)}.png'))
        cf_img = Image.open(os.path.join(output_dir, 'bucket_0_10', 'counterfactual', f'{str(uidx).zfill(5)}.png'))
        cf_img_spatial = Image.open(os.path.join(output_dir + '_spatial', 'bucket_0_10', 'counterfactual', f'{str(uidx).zfill(5)}.png'))

        orig_expl = Image.open(os.path.join(output_dir, 'explanations', f'{str(uidx).zfill(5)}_orig_expl.png'))
        base_cf_expl = Image.open(os.path.join(base_dir, 'explanations', f'{str(uidx).zfill(5)}_ce_expl.png'))
        cf_expl = Image.open(os.path.join(output_dir, 'explanations', f'{str(uidx).zfill(5)}_ce_expl.png'))
        cf_expl_spatial = Image.open(os.path.join(output_dir + '_spatial', 'explanations', f'{str(uidx).zfill(5)}_ce_expl.png'))

        fig, ax = plt.subplots(2,4, figsize=(24,12))

        ax[0,0].imshow(orig_img)
        ax[0,1].imshow(base_cf_img)
        ax[0,2].imshow(cf_img)
        ax[0,3].imshow(cf_img_spatial)
        ax[1,0].imshow(orig_expl)
        ax[1,1].imshow(base_cf_expl)
        ax[1,2].imshow(cf_expl)
        ax[1,3].imshow(cf_expl_spatial)

        ax[0,0].set_title('Original', fontsize=24)
        ax[0,1].set_title('LDCE', fontsize=24)
        ax[0,2].set_title('CoLa-DCE', fontsize=24)
        ax[0,3].set_title('CoLa-DCE (spatial)', fontsize=24)

        for a in ax:
            for ab in a:
                ab.axis('off')
        plt.tight_layout()
        plt.savefig(f'/results/counterfactuals/imgs/spatial_comparison_{model}_{k}_{str(uidx).zfill(5)}.png')
        plt.close()

if __name__ == '__main__':

    visualize_example()
