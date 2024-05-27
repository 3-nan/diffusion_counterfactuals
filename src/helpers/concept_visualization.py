""" Script for loading and visualizing concepts. """
from typing import Any, Dict, Iterable, Tuple, Union
import h5py
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
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

    attr = torch.from_numpy(attr)
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
        attr_rf = zennit.image.imgify(attr_rf, cmap='coldnhot', symmetric=True)
        return attr_rf, img
    else:
        return img

def load_concept_attributions(concept_file, layer_name, label, concept_id):
    """ Load the RelMax concepts for given concept_id.
        Additionally retrieve original image inputs.
    """

    with h5py.File(concept_file, "r", locking=False) as concept_dataset:

        concept_group = concept_dataset[layer_name][str(label)][str(concept_id)]

        attrs = concept_group['attribution'][:]
        img_ids = concept_group['img_id'][:]

    return attrs, img_ids

def show_concept_examples(dataset, concept_file, layer_name, label, concept_id):

    attrs, img_ids = load_concept_attributions(concept_file, layer_name, label, concept_id)

    # Load images by img_ids
    img_samples = [dataset[ti][0] for ti in img_ids]

    # crop relevance field
    arfs, irfs = [], []
    for attr, img in zip(attrs, img_samples):
        attr_rf, img_rf = crop_relevance_field(attr, img, rf=True)
        arfs.append(attr_rf)
        irfs.append(img_rf)

    # Form grid
    fig, ax = plt.subplots(2, len(arfs), figsize=(18, 6))

    for a, arf in enumerate(arfs):
        ax[0, a].imshow(arf)
    
    for i, irf in enumerate(irfs):
        ax[1, i].imshow(irf)

    for axes in ax:
        for axax in axes:
            axax.set_xticks([])
            axax.set_yticks([])
    
    # fig.tight_layout()
    plt.subplots_adjust(wspace=0., hspace=0.)
    return fig

def imgify(image: Union[Image.Image, torch.Tensor, np.ndarray], cmap: str = "bwr", vmin=None, vmax=None, symmetric=False, level=1.0, grid=False, gridfill=None, resize:int=None,
        padding=False) -> Image.Image:
    """
    Convenient wrapper around zennit.image.imgify supporting tensors, numpy arrays and PIL Images. Allows resizing while keeping the aspect
    ratio intact and padding to a square shape.

    Parameters:
    ----------
    image: torch.Tensor, np.ndarray or PIL Image
        With 2 dimensions greyscale, or 3 dimensions with 1 or 3 values in the first or last dimension (color).
    resize: None or int
        If None, no resizing is applied. If int, sets the maximal aspect ratio of the image.
    padding: boolean
        If True, pads the image into a square shape by setting the alpha channel to zero outside the image.
    vmin: float or obj:numpy.ndarray
        Manual minimum value of the array. Overrides the used norm's minimum value.
    vmax: float or obj:numpy.ndarray
        Manual maximum value of the array. Overrides the used norm's maximum value.
    cmap: str or ColorMap
        String to specify a built-in color map, code used to create a new color map, or a ColorMap instance, which will be used to create a palette. The color map will only be applied for arrays with only a single color channel. The color will be specified as a palette in the PIL Image.
    
    Returns:
    --------
    image: PIL.Image object
    """

    if isinstance(image, torch.Tensor):
        img = zennit.image.imgify(image.detach().cpu(), cmap=cmap, vmin=vmin, vmax=vmax, symmetric=symmetric, level=level, grid=grid, gridfill=gridfill)
    elif isinstance(image, np.ndarray):
        img = zennit.image.imgify(image, cmap=cmap, vmin=vmin, vmax=vmax, symmetric=symmetric, level=level, grid=grid, gridfill=gridfill)
    elif isinstance(image, Image.Image):
        img = image
    else:
        raise TypeError("Only PIL.Image, torch.Tensor or np.ndarray types are supported!")

    if resize:
        ratio = resize/max(img.size)
        new_size = tuple([int(x*ratio) for x in img.size])
        img = img.resize(new_size, Image.NEAREST)

    if padding:
        max_size = resize if resize else max(img.size)
        new_im = Image.new("RGBA", (max_size, max_size))
        new_im.putalpha(0)
        new_im.paste(img, ((max_size-img.size[0])//2, (max_size-img.size[1])//2))
        img = new_im

    return img

def plot_grid(ref_c: Dict[int, Any], concept_diff=None, cmap_dim=1, cmap="bwr", vmin=None, vmax=None, symmetric=True, resize=None, padding=True, figsize=(6, 6)):
    """
    Plots dictionary of reference images as they are returned of the 'get_max_reference' method. To every element in the list crp.imgify is applied with its respective argument values.

    Parameters:
    ----------
    ref_c: dict with keys: integer and value: several lists filled with torch.Tensor, np.ndarray or PIL Image
        To every element in the list crp.imgify is applied.
    resize: None or int
        If None, no resizing is applied. If int, sets the maximal aspect ratio of the image.
    padding: boolean
        If True, pads the image into a square shape by setting the alpha channel to zero outside the image.
    figsize: tuple or None
        Size of plt.figure
    cmap_dim: int, 0 or 1 
        Applies the remaining parameters to the first or second element of the tuple list, i.e. plot as heatmap

    REMAINING PARAMETERS: correspond to zennit.imgify

    Returns:
    --------
    shows matplotlib.pyplot plot
    """

    keys = list(ref_c.keys())
    nrows = len(keys)
    value = next(iter(ref_c.values()))

    if cmap_dim > 2 or cmap_dim < 1 or cmap_dim == None:
        raise ValueError("'cmap_dim' must be 0 or 1 or None.")

    if isinstance(value, Tuple) and isinstance(value[0], Iterable):
        nsubrows = len(value)
        ncols = len(value[0])
    elif isinstance(value, Iterable):
        nsubrows = 1
        ncols = len(value)
    else:
        raise ValueError("'ref_c' dictionary must contain an iterable of torch.Tensor, np.ndarray or PIL Image or a tuple of thereof.")

    fig = plt.figure(figsize=figsize)
    outer = gridspec.GridSpec(nrows, 1, wspace=0, hspace=0.2)

    for i in range(nrows):
        inner = gridspec.GridSpecFromSubplotSpec(nsubrows, ncols, subplot_spec=outer[i], wspace=0, hspace=0.1)

        for sr in range(nsubrows):

            if nsubrows > 1:
                img_list = ref_c[keys[i]][sr]
            else:
                img_list = ref_c[keys[i]]
            
            for c in range(ncols):
                if c < len(img_list):
                    ax = plt.Subplot(fig, inner[sr, c])

                    if sr == cmap_dim:
                        img = imgify(img_list[c], cmap=cmap, vmin=vmin, vmax=vmax, symmetric=symmetric, resize=resize, padding=padding)
                    else:
                        img = imgify(img_list[c], resize=resize, padding=padding)

                    ax.imshow(img)
                    ax.set_xticks([])
                    ax.set_yticks([])

                    if sr == 0 and c == 0:
                        if isinstance(concept_diff, np.ndarray):
                            ax.set_ylabel(f'{keys[i]}\n{concept_diff[i]:.4f}')
                        else:
                            # print(f'concept_diff is of type {type(concept_diff)}')
                            ax.set_ylabel(keys[i])

                    fig.add_subplot(ax)
                
    outer.tight_layout(fig)  
    fig.show()
