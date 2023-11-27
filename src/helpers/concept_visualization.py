""" Script for loading and visualizing concepts. """
import h5py
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms.functional import gaussian_blur
import zennit


def create_visual(attr):

    expl_img = zennit.image.imgify(attr)
    return expl_img

def max_norm(rel, stabilize=1e-10):
    
    return rel / (rel.max() + stabilize)

def get_crop_range(heatmap, crop_th):
    """
    Returns indices in order to crop the supplied heatmap where relevance is greater than heatmap > crop_th.

    Parameters:
    ----------
    heatmaps: torch.Tensor
        ouput heatmap tensor of the CondAttribution call
    crop_th: between [0 and 1)
        Cropping Threshold: Crops the image in regions where relevance is smaller than max(relevance)*crop_th. 
        Cropping is only applied, if receptive field 'rf' is set to True.
    """

    crop_mask = heatmap > crop_th
    rows, columns = np.where(crop_mask)

    if len(rows) == 0 or len(columns) == 0:
        # rf is empty
        return 0, -1, 0, -1

    row1, row2 = rows.min(), rows.max()
    col1, col2 = columns.min(), columns.max()

    if (row1 >= row2) and (col1 >= col2):
        # rf is empty
        return 0, -1, 0, -1

    return row1, row2, col1, col2

def crop_relevance_field(attr, img, rf=False, alpha=0.3, vis_th=0.2, crop_th=0.1, kernel_size=19):

    filtered_heat = max_norm(gaussian_blur(attr.unsqueeze(0), kernel_size=kernel_size)[0])
    vis_mask = filtered_heat > vis_th
    
    if rf:
        row1, row2, col1, col2 = get_crop_range(filtered_heat, crop_th)
        
        img_t = img[..., row1:row2, col1:col2]
        attr_rf = attr[row1:row2, col1:col2]
        vis_mask_t = vis_mask[row1:row2, col1:col2]

        if img_t.sum() != 0 and vis_mask_t.sum() != 0:
            # check whether img_t or vis_mask_t is not empty
            img = img_t
            vis_mask = vis_mask_t

    inv_mask = ~vis_mask
    img = img * vis_mask + img * inv_mask * alpha
    img = zennit.image.imgify(img)

    if rf:
        return attr_rf, img
    else:
        return img

def load_concept_attributions(concept_file, layer_name, concept_id):
    """ Load the RelMax concepts for given concept_id.
        Additionally retrieve original image inputs.
    """

    with h5py.File(concept_file, "r", locking=False) as concept_dataset:

        concept_group = concept_dataset[layer_name][str(concept_id)]
        attrs = concept_group['attribution'][:]
        img_ids = concept_group['img_id'][:]

    return attrs, img_ids

def show_concept_examples(dataset, concept_file, layer_name, concept_id):

    attrs, img_ids = load_concept_attributions(concept_file, layer_name, concept_id)

    # Load images by img_ids
    img_samples = [dataset[ti][0] for ti in img_ids]

    # crop relevance field
    arfs, irfs = [], []
    for attr, img in zip(attrs, img_samples):
        attr_rf, img_rf = crop_relevance_field(attr, img, rf=True)
        arfs.append(attr_rf)
        irfs.append(img_rf)

    # Form grid
    fig, ax = plt.subplots(2, len(arfs))

    for a, arf in enumerate(arfs):
        ax[0, a].imshow(arf)
    
    for i, irf in enumerate(irfs):
        ax[1, i].imshow(irf)

    for axes in ax:
        for axax in axes:
            axax.set_xticks([])
            axax.set_yticks([])
    
    fig.tight_layout()
    return fig
