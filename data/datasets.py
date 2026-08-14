from typing import Any
from torchvision import datasets, transforms
from data.imagenet_classnames import name_map, folder_label_map
import yaml
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import os
from PIL import Image
from pathlib import Path
import random
import pickle
import linecache
import itertools


def select_subset_indices(dataset, target: str, n: int = 1000) -> np.ndarray:
    """Pick the ``n`` sample indices a counterfactual run should process.

    Shared by ``run_ldce_baseline.py`` and ``run_concept_ldce.py`` so the
    baseline and concept runs always operate on the *same* samples.

    For BoxCars116k the ``classification_splits.pkl`` lists samples grouped by
    vehicle track, not shuffled by class, so a plain ``arange(n)`` slice covers
    only a handful of classes (~95% Skoda Fabia/Octavia). Stratify instead:
    shuffle within each class (fixed seed for reproducibility) and round-robin
    across classes so every class contributes once before any class contributes
    twice, capped at ``n`` total. All other datasets use the first ``n``.
    """
    if "BoxCars116k" in target:
        rng = np.random.default_rng(0)
        by_class: dict = {}
        for idx, lbl in enumerate(dataset.labels):
            by_class.setdefault(lbl, []).append(idx)
        for idxs in by_class.values():
            rng.shuffle(idxs)
        order = [
            i for round_idxs in itertools.zip_longest(*by_class.values())
            for i in round_idxs if i is not None
        ]
        return np.array(order[:n])
    return np.arange(n)


def closest_indices_path(dataset: str, name: str | None = None, base_dir: str = "data") -> str:
    """Resolve the near-miss target file for a (dataset, classifier) setting.

    Prefers a setting-specific file ``<dataset>_closest_indices_<name>.json``
    (written by ``compute_closest_indices.py --name <name>``) and falls back to
    the legacy shared ``<dataset>_closest_indices.json`` when no setting-specific
    file exists. This keeps the original VGG runs working unchanged (their
    ``name`` is ``vgg16_bn`` and no such suffixed file exists, so they resolve to
    the legacy path) while letting resnet18 / vit_b_16 pick up their own
    classifier-derived targets once those files are generated.

    ``dataset`` is the short key used in the filename (``cars``, ``boxcars``,
    ``flowers``, ``pets``). ``name`` is ``cfg.classifier_model.name``.
    """
    if name:
        specific = os.path.join(base_dir, f"{dataset}_closest_indices_{name}.json")
        if os.path.exists(specific):
            return specific
    return os.path.join(base_dir, f"{dataset}_closest_indices.json")


class ImageNet(datasets.ImageFolder):
    classes = [name_map[i] for i in range(1000)]
    name_map = name_map

    def __init__(
            self, 
            root:str, 
            split:str="val", 
            transform=None, 
            target_transform=None, 
            class_idcs=None, 
            start_sample: float = 0., 
            end_sample: int = 50000//1000,
            return_tgt_cls: bool = False,
            idx_to_tgt_cls_path = None,
            restart_idx: int = 0, 
            image_size: int = 256,
            **kwargs
    ):
        _ = kwargs  # Just for consistency with other datasets.
        print(f"Loading ImageNet with start_sample={start_sample}, end_sample={end_sample} ")
        assert split in ["train", "val"]
        assert start_sample < end_sample and start_sample >= 0 and end_sample <= 50000//1000
        self.start_sample = start_sample

        assert 0 <= restart_idx < 50000
        self.restart_idx = restart_idx

        path = root if root[-3:] == "val" or root[-5:] == "train" else os.path.join(root, split)
        if transform:
            transform = transforms.Compose([transform, transforms.Resize(image_size)])
        super().__init__(path, transform=transform, target_transform=target_transform)
        
        with open(idx_to_tgt_cls_path, 'r') as file:
            idx_to_tgt_cls = yaml.safe_load(file)
            if isinstance(idx_to_tgt_cls, dict):
                idx_to_tgt_cls = [idx_to_tgt_cls[i] for i in range(len(idx_to_tgt_cls))]
        self.idx_to_tgt_cls = idx_to_tgt_cls

        self.return_tgt_cls = return_tgt_cls

        if class_idcs is not None:
            class_idcs = list(sorted(class_idcs))
            tgt_to_tgt_map = {c: i for i, c in enumerate(class_idcs)}
            self.classes = [self.classes[c] for c in class_idcs]
            samples = []
            idx_to_tgt_cls = []
            for i, (p, t) in enumerate(self.samples):
                if t in tgt_to_tgt_map:
                    samples.append((p, tgt_to_tgt_map[t]))
                    idx_to_tgt_cls.append(self.idx_to_tgt_cls[i])
            
            self.idx_to_tgt_cls = idx_to_tgt_cls
            #self.samples = [(p, tgt_to_tgt_map[t]) for i, (p, t) in enumerate(self.samples) if t in tgt_to_tgt_map]
            self.class_to_idx = {k: tgt_to_tgt_map[v] for k, v in self.class_to_idx.items() if v in tgt_to_tgt_map}

        if "val" == split: # reorder
            new_samples = []
            idx_to_tgt_cls = []
            for idx in range(50000//1000):
                new_samples.extend(self.samples[idx::50000//1000])
                idx_to_tgt_cls.extend(self.idx_to_tgt_cls[idx::50000//1000])
            self.samples = new_samples[int(start_sample*1000):end_sample*1000]
            self.idx_to_tgt_cls = idx_to_tgt_cls[int(start_sample*1000):end_sample*1000]

        else:
            raise NotImplementedError
        
        if self.restart_idx > 0:
            self.samples = self.samples[self.restart_idx:]
            self.idx_to_tgt_cls = self.idx_to_tgt_cls[self.restart_idx:]

        self.class_labels = {i: folder_label_map[folder] for i, folder in enumerate(self.classes)}
        self.targets = np.array(self.samples)[:, 1]
    
    def __getitem__(self, index):
        sample = super().__getitem__(index)
        if self.return_tgt_cls:
            return *sample, self.idx_to_tgt_cls[index], index + self.start_sample*1000 + self.restart_idx
        else:
            # return sample, index + self.start_sample*1000 + self.restart_idx
            return sample

class Flowers102(Dataset):
    def __init__(self, root, transform, shard: int = 0, num_shards: int = 1, return_index=True, **kwargs) -> None:
        super().__init__()
        target_transform = lambda x: x-1 # flowers starts from idx 1
        self.dataset = datasets.Flowers102(root=root, split="test", transform=transform, target_transform=target_transform, download=True)
        self.return_index = return_index
        # compute shards
        self.dataset._image_files = self.dataset._image_files[shard::num_shards]
        self.dataset._labels = self.dataset._labels[shard::num_shards]
    
    def __getitem__(self, index: Any) -> Any:
        img, label = self.dataset.__getitem__(index)
        label = label - 1
        if self.return_index:
            return img, label, index
        else:
            return img, label
    
    def __len__(self):
        return len(self.dataset)
    
class OxfordIIIPets(Dataset):
    def __init__(self, root, transform, shard: int = 0, num_shards: int = 1, return_index=True, **kwargs) -> None:
        super().__init__()
        self.dataset = datasets.OxfordIIITPet(root=root, split="test", target_types="category", transform=transform, download=True)
        self.return_index = return_index
        # compute shards
        self.dataset._images = self.dataset._images[shard::num_shards]
        self.dataset._labels = self.dataset._labels[shard::num_shards]
    
    def __getitem__(self, index: Any) -> Any:
        img, label = self.dataset.__getitem__(index)
        if self.return_index:
            return img, label, index
        else:
            return img, label
    
    def __len__(self):
        return len(self.dataset)

from torchvision.datasets import CelebA

class CelebADataset(Dataset):
    def __init__(
        self,
        image_size,
        data_dir,
        partition,
        shard=0,
        num_shards=1,
        class_cond=False,
        random_crop=True,
        random_flip=True,
        query_label=-1,
        normalize=True,
        restart_idx: int = 0,
        return_index=True,
        **kwargs
    ):
        super().__init__()
        # partition_df = pd.read_csv(os.path.join(data_dir, 'list_eval_partition.csv'))
        self.data_dir = data_dir
        self.return_index = return_index
        # data = pd.read_csv(os.path.join(data_dir, 'list_attr_celeba.csv'))

        # if partition == 'train':
        #     partition = 0
        # elif partition == 'val':
        #     partition = 1
        # elif partition == 'test':
        #     partition = 2
        # else:
        #     raise ValueError(f'Unkown partition {partition}')

        # self.data = data[partition_df['partition'] == partition]
        # self.data = self.data[shard::num_shards]
        # self.data.reset_index(inplace=True)
        # self.data.replace(-1, 0, inplace=True)

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip() if random_flip else lambda x: x,
            transforms.CenterCrop(image_size),
            transforms.RandomResizedCrop(image_size, (0.95, 1.0)) if random_crop else lambda x: x,
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],
                                 [0.5, 0.5, 0.5]) if normalize else lambda x: x
        ])

        # self.dataset = CelebA(datadir, split=)
        self.dataset = CelebA(self.data_dir, split=partition, target_type='attr', transform=self.transform)

        self.query = query_label
        self.class_cond = class_cond

        self.restart_idx = restart_idx
        if self.restart_idx > 0:
            print("TODO")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, labels = self.dataset[idx]
        # sample = self.data.iloc[idx, :]
        # labels = sample[2:].to_numpy()
        # if self.query != -1:
        #     labels = int(labels[self.query])
        # else:
        #     labels = torch.from_numpy(labels.astype('float32'))
        # img_file = sample['image_id']

        # with open(os.path.join(self.data_dir, 'img_align_celeba', img_file), "rb") as f:
        #     img = Image.open(f)
        #     img = img.convert('RGB')

        # img = self.transform(img)

        labels = labels[self.query]

        if self.return_index:
            return img, labels, idx

        if self.query != -1:
            return img, labels

        if self.class_cond:
            return img, labels
        else:
            return img, {}


class CelebAHQDataset(Dataset):
    def __init__(
        self,
        image_size,
        data_dir,
        partition,
        shard=0,
        num_shards=1,
        class_cond=False,
        random_crop=True,
        random_flip=True,
        query_label=-1,
        normalize=True,
        restart_idx: int = 0,
        **kwargs
    ):
        from io import StringIO
        # read annotation files
        with open(os.path.join(data_dir, 'CelebAMask-HQ-attribute-anno.txt'), 'r') as f:
            datastr = f.read()[6:]
            datastr = 'idx ' +  datastr.replace('  ', ' ')

        with open(os.path.join(data_dir, 'CelebA-HQ-to-CelebA-mapping.txt'), 'r') as f:
            mapstr = f.read()
            mapstr = [i for i in mapstr.split(' ') if i != '']

        mapstr = ' '.join(mapstr)

        data = pd.read_csv(StringIO(datastr), sep=' ')
        partition_df = pd.read_csv(os.path.join(data_dir, 'list_eval_partition.csv'))
        mapping_df = pd.read_csv(StringIO(mapstr), sep=' ')
        # mapping_df.rename(columns={'orig_file': 'image_id'}, inplace=True)
        partition_df = pd.merge(mapping_df, partition_df, on='idx')

        self.data_dir = data_dir

        if partition == 'train':
            partition = 0
        elif partition == 'val':
            partition = 1
        elif partition == 'test':
            partition = 2
        else:
            raise ValueError(f'Unkown partition {partition}')

        self.data = data[partition_df['split'] == partition]
        self.data = self.data[shard::num_shards]
        self.data.reset_index(inplace=True)
        self.data.replace(-1, 0, inplace=True)

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip() if random_flip else lambda x: x,
            transforms.CenterCrop(image_size),
            transforms.RandomResizedCrop(image_size, (0.95, 1.0)) if random_crop else lambda x: x,
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],
                                 [0.5, 0.5, 0.5])  if normalize else lambda x: x
        ])

        self.query = query_label
        self.class_cond = class_cond

        self.restart_idx = restart_idx
        if self.restart_idx > 0:
            self.data = self.data.iloc[self.restart_idx:]
            self.data.reset_index(inplace=True)
            self.data.replace(-1, 0, inplace=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx, :]
        labels = sample[2:].to_numpy()
        if self.query != -1:
            labels = int(labels[self.query])
        else:
            labels = torch.from_numpy(labels.astype('float32'))
        img_file = sample['idx']

        with open(os.path.join(self.data_dir, 'CelebA-HQ-img', img_file), "rb") as f:
            img = Image.open(f)
            img = img.convert('RGB')

        img = self.transform(img)

        if self.query != -1:
            return img, labels, self.restart_idx + idx

        if self.class_cond:
            return img, labels, self.restart_idx + idx
        else:
            return img, {}, self.restart_idx + idx

class CUB(Dataset):
    # Implementation from https://github.com/JonathanCrabbe/CARs

    N_ATTRIBUTES = 312
    N_CLASSES = 200
    attribute_map = [1, 4, 6, 7, 10, 14, 15, 20, 21, 23, 25, 29, 30, 35, 36, 38, 40, 44, 45, 50, 51, 53, 54, 56, 57, 59,
                     63, 64, 69, 70, 72, 75, 80, 84, 90, 91, 93, 99, 101, 106, 110, 111, 116, 117, 119, 125, 126, 131,
                     132, 134, 145, 149, 151, 152, 153, 157, 158, 163, 164, 168, 172, 178, 179, 181, 183, 187, 188, 193,
                     194, 196, 198, 202, 203, 208, 209, 211, 212, 213, 218, 220, 221, 225, 235, 236, 238, 239, 240,
                     242, 243, 244, 249, 253, 254, 259, 260, 262, 268, 274, 277, 283, 289, 292, 293, 294, 298, 299, 304,
                     305, 308, 309, 310, 311]
    train=False

    def __init__(self, root, transform, shard: int = 0, num_shards: int = 1, return_index=True, **kwargs) -> None:
        super().__init__()

        self.root = root
        self.return_idx = return_index
        images = pd.read_csv(os.path.join(root, 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(root, 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(root, 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])

        data = images.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')

        if self.train:
            self.data = self.data[self.data.is_training_img == 1]
        else:
            self.data = self.data[self.data.is_training_img == 0]

        self.data = self.data[shard::num_shards]
        self.data.reset_index(inplace=True)
        self.data.replace(-1, 0, inplace=True)

        self.transform = transform

    def __len__(self):
        return len(self.data)   #set)

    def __getitem__(self, index):
        # print(index)
        # print(self.data)
        img_data = self.data.iloc[[index]]
        # img_data = self.data[index]
        # print(img_data)
        img_path = img_data['filepath'].values[0]
        # print(img_path)
        # Trim unnecessary paths
        try:
            # idx = img_path.split('/').index('CUB_200_2011')
            # if self.image_dir != 'images':
            #     img_path = '/'.join([self.image_dir] + img_path.split('/')[idx + 1:])
            # else:
            #     img_path = '/'.join(img_path.split('/')[idx:])
            img_path = os.path.join(self.root, 'images', img_path)
            img = Image.open(img_path).convert('RGB')
        except:
            img_path_split = img_path.split('/')
            split = 'train' if self.is_train else 'test'
            img_path = '/'.join(img_path_split[:2] + [split] + img_path_split[2:])
            img = Image.open(img_path).convert('RGB')

        class_label = img_data['target'].values[0] - 1
        if self.transform:
            img = self.transform(img)

        # if self.use_attr:
        #     if self.uncertain_label:
        #         attr_label = img_data['uncertain_attribute_label']
        #     else:
        #         attr_label = img_data['target']
        #     if self.no_img:
        #         if self.n_class_attr == 3:
        #             one_hot_attr_label = np.zeros((self.N_ATTRIBUTES, self.n_class_attr))
        #             one_hot_attr_label[np.arange(self.N_ATTRIBUTES), attr_label] = 1
        #             return one_hot_attr_label, class_label
        #         else:
        #             return attr_label, class_label
        #     else:
        #         return img, class_label, attr_label
        # else:
        # print(class_label)
        if self.return_idx:
            return img, class_label, index
        else:
            return img, class_label

    def get_raw_image(self, idx: int,  resol: int = 299):
        img_data = self.data[idx]
        img_path = img_data['img_path']
        # Trim unnecessary paths
        try:
            idx = img_path.split('/').index('CUB_200_2011')
            if self.image_dir != 'images':
                img_path = '/'.join([self.image_dir] + img_path.split('/')[idx + 1:])
            else:
                img_path = '/'.join(img_path.split('/')[idx:])
            img = Image.open(img_path).convert('RGB')
        except:
            img_path_split = img_path.split('/')
            split = 'train' if self.is_train else 'test'
            img_path = '/'.join(img_path_split[:2] + [split] + img_path_split[2:])
            img = Image.open(img_path).convert('RGB')
        center_crop = transforms.Resize((resol, resol))
        return center_crop(img)

    def class_name(self, class_id) -> str:
        """
        Get the name of a class
        Args:
            class_id: integer identifying the concept

        Returns:
            String corresponding to the concept name
        """
        class_path = Path(self.root) / "classes.txt"
        name = linecache.getline(str(class_path), class_id+1)
        name = name.split(".")[1]  # Remove the line number
        name = name.replace("_", " ")  # Put spacing in class names
        name = name[:-1]  # Remove breakline character
        return name.title()

    def get_class_names(self):
        """
        Get the name of all concepts
        Returns:
            List of all concept names
        """
        return [self.class_name(i) for i in range(self.N_CLASSES)]


# ---------------------------------------------------------------------------
# Fine-grained vehicle datasets
#
# Same contract as Flowers102 / OxfordIIIPets above:
#   __init__(root, transform, shard=0, num_shards=1, return_index=True, **kwargs)
#   __getitem__ -> (img, label, index) if return_index else (img, label)
#
# All three expose ``self.classes`` as human-readable names so they can be fed
# straight into the text-conditioned LDM (miniSD) the way the Pets/Flowers
# class names are, and ``self.N_CLASSES`` for the near-miss target search.
# ---------------------------------------------------------------------------


def _pad_and_crop(img, box, padding: float = 0.15):
    """Crop to ``box`` = (x1, y1, x2, y2), expanded by ``padding`` on each side.

    Tight crops starve the diffusion model; a bit of street/background context
    gives the denoiser something to work with while the spatial conditioning
    keeps the changes on the vehicle.
    """
    if box is None:
        return img
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    dx, dy = padding * w, padding * h
    x1, y1 = max(0, int(x1 - dx)), max(0, int(y1 - dy))
    x2, y2 = min(img.width, int(x2 + dx)), min(img.height, int(y2 + dy))
    if x2 <= x1 or y2 <= y1:
        return img
    return img.crop((x1, y1, x2, y2))


def letterbox_resize(img, size, fill=114):
    """Resize so the *longer* side == size, then pad the shorter side up to
    size. Shared by finetune_classifier.py, compute_closest_indices.py and
    run_ldce_baseline.py so the classifier is trained, indexed and steered on
    the exact same preprocessing.

    Deliberately not ``Resize(size)`` + ``CenterCrop(size)`` (crops whatever
    sticks out past the shorter side -- for a landscape ``crop_to_bbox`` car
    crop, that can cut the nose/tail off) and not ``Resize((size, size))``
    (stretches, distorting proportions). This keeps every pixel of the source
    crop and the car's true proportions, at the cost of neutral-gray padding
    on the shorter axis instead.
    """
    from torchvision.transforms import functional as TF

    w, h = img.size
    scale = size / max(w, h)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    img = TF.resize(img, [new_h, new_w])
    pad_w, pad_h = size - new_w, size - new_h
    left, top = pad_w // 2, pad_h // 2
    right, bottom = pad_w - left, pad_h - top
    return TF.pad(img, [left, top, right, bottom], fill=fill)


class StanfordCars(Dataset):
    """Stanford Cars (Krause et al., 2013) -- 196 make/model/year classes.

    The original host (ai.stanford.edu) is dead and
    ``torchvision.datasets.StanfordCars(download=True)`` is a no-op, so this
    reads the devkit ``.mat`` files directly. Point ``root`` at a mirror laid
    out as::

        <root>/cars_train/*.jpg
        <root>/cars_test/*.jpg
        <root>/cars_test_annos_withlabels.mat
        <root>/devkit/cars_meta.mat
        <root>/devkit/cars_train_annos.mat

    Note that the plain ``cars_test_annos.mat`` in the devkit has no labels --
    you need the ``_withlabels`` variant (shipped with the Kaggle mirror and
    with github.com/jhpohovey/StanfordCars-Dataset).
    """

    N_CLASSES = 196

    def __init__(
        self,
        root,
        transform,
        split: str = "test",
        shard: int = 0,
        num_shards: int = 1,
        return_index: bool = True,
        crop_to_bbox: bool = True,
        bbox_padding: float = 0.15,
        min_box_size: int = 0,
        **kwargs,
    ) -> None:
        super().__init__()
        _ = kwargs  # Just for consistency with other datasets.
        from scipy.io import loadmat

        assert split in ("train", "test")
        if os.path.isdir(os.path.join(root, "stanford_cars")):
            root = os.path.join(root, "stanford_cars")
        self.root = root
        self.transform = transform
        self.return_index = return_index
        self.crop_to_bbox = crop_to_bbox
        self.bbox_padding = bbox_padding

        meta = loadmat(os.path.join(root, "devkit", "cars_meta.mat"), squeeze_me=True)
        self.classes = [str(c).strip() for c in meta["class_names"]]

        if split == "test":
            self.image_dir = os.path.join(root, "cars_test")
            candidates = [
                os.path.join(root, "cars_test_annos_withlabels.mat"),
                os.path.join(root, "devkit", "cars_test_annos_withlabels.mat"),
            ]
        else:
            self.image_dir = os.path.join(root, "cars_train")
            candidates = [
                os.path.join(root, "devkit", "cars_train_annos.mat"),
                os.path.join(root, "cars_train_annos.mat"),
            ]
        anno_path = next((p for p in candidates if os.path.isfile(p)), None)
        if anno_path is None:
            raise FileNotFoundError(
                f"No Stanford Cars annotations for split='{split}'. Looked in: {candidates}"
            )

        annos = loadmat(anno_path, squeeze_me=True)["annotations"]
        files = [str(f) for f in np.atleast_1d(annos["fname"])]
        labels = [int(c) - 1 for c in np.atleast_1d(annos["class"])]  # .mat is 1-indexed
        boxes = list(
            zip(
                np.atleast_1d(annos["bbox_x1"]).astype(int).tolist(),
                np.atleast_1d(annos["bbox_y1"]).astype(int).tolist(),
                np.atleast_1d(annos["bbox_x2"]).astype(int).tolist(),
                np.atleast_1d(annos["bbox_y2"]).astype(int).tolist(),
            )
        )

        # Drop crops that are too small to survive the 256px LDM.
        if min_box_size > 0:
            keep = [
                i
                for i, (x1, y1, x2, y2) in enumerate(boxes)
                if min(x2 - x1, y2 - y1) >= min_box_size
            ]
            files = [files[i] for i in keep]
            labels = [labels[i] for i in keep]
            boxes = [boxes[i] for i in keep]

        # compute shards
        self.files = files[shard::num_shards]
        self.labels = labels[shard::num_shards]
        self.boxes = boxes[shard::num_shards]
        self.targets = np.array(self.labels)

    def __getitem__(self, index: Any) -> Any:
        img = Image.open(os.path.join(self.image_dir, self.files[index])).convert("RGB")
        if self.crop_to_bbox:
            img = _pad_and_crop(img, self.boxes[index], self.bbox_padding)
        if self.transform is not None:
            img = self.transform(img)
        label = self.labels[index]
        if self.return_index:
            return img, label, index
        else:
            return img, label

    def __len__(self):
        return len(self.files)

    def class_name(self, class_id) -> str:
        return self.classes[class_id]

    def get_class_names(self):
        return list(self.classes)


class CompCars(Dataset):
    """CompCars (Yang et al., 2015) -- 163 makes / 1716 models / 12 body types.

    Two subsets, selected with ``subset``:

    * ``"web"`` -- web-nature images, laid out as
      ``data/image/<make_id>/<model_id>/<year>/<img>.jpg`` with a matching
      ``data/label/.../<img>.txt`` holding viewpoint, body type and a 2D box.
      Splits come from ``data/train_test_split/classification/{train,test}.txt``.
    * ``"sv"`` -- surveillance-nature images (frontal views from roadside
      cameras), ``sv_data/image/<model_id>/<img>.jpg`` with splits in
      ``sv_data/{train,test}_surveillance.txt``.

    ``label_level`` picks the granularity: ``"model"`` is the fine-grained,
    dog-breed-analogous task; ``"make"`` and ``"type"`` are coarser.

    ``viewpoints`` restricts the web subset to a fixed camera pose (web only,
    since the labels only exist there). Keeping a single viewpoint -- 4
    (front-side) is the usual choice -- makes the near-miss neighbourhood
    semantic rather than geometric, which matters a lot for concept selection.
    """

    N_CLASSES = {"model": 1716, "make": 163, "type": 12}
    TYPE_NAMES = [
        "MPV", "SUV", "sedan", "hatchback", "minibus", "fastback",
        "estate", "pickup", "hardtop convertible", "sports",
        "crossover", "convertible",
    ]
    VIEWPOINT_NAMES = {
        1: "front", 2: "rear", 3: "side", 4: "front-side", 5: "rear-side",
    }

    def __init__(
        self,
        root,
        transform,
        split: str = "test",
        subset: str = "web",
        label_level: str = "model",
        shard: int = 0,
        num_shards: int = 1,
        return_index: bool = True,
        crop_to_bbox: bool = True,
        bbox_padding: float = 0.15,
        min_box_size: int = 0,
        viewpoints=None,
        cache: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        _ = kwargs  # Just for consistency with other datasets.
        from scipy.io import loadmat

        assert split in ("train", "test")
        assert subset in ("web", "sv")
        assert label_level in ("model", "make", "type")
        if subset == "sv":
            assert label_level == "model", "the surveillance subset only ships model labels"
            assert viewpoints is None, "the surveillance subset has no viewpoint labels"

        self.root = root
        self.subset = subset
        self.label_level = label_level
        self.transform = transform
        self.return_index = return_index
        self.crop_to_bbox = crop_to_bbox and subset == "web"
        self.bbox_padding = bbox_padding
        self.N_CLASSES = CompCars.N_CLASSES[label_level]

        # ---- class names --------------------------------------------------
        if subset == "web":
            names = loadmat(os.path.join(root, "misc", "make_model_name.mat"), squeeze_me=True)
            self.image_dir = os.path.join(root, "data", "image")
            self.label_dir = os.path.join(root, "data", "label")
        else:
            names = loadmat(os.path.join(root, "sv_data", "sv_make_model_name.mat"), squeeze_me=True)
            self.image_dir = os.path.join(root, "sv_data", "image")
            self.label_dir = None

        make_names = [str(n).strip() for n in np.atleast_1d(names["make_names"])]
        model_names = [str(n).strip() for n in np.atleast_1d(names["model_names"])]
        if label_level == "make":
            self.classes = make_names
        elif label_level == "type":
            self.classes = list(CompCars.TYPE_NAMES)
        else:
            self.classes = model_names

        # ---- sample list ---------------------------------------------------
        cache_path = os.path.join(root, f".compcars_{subset}_{split}_index.pkl")
        if cache and os.path.isfile(cache_path):
            with open(cache_path, "rb") as f:
                records = pickle.load(f)
        else:
            records = self._scan(root, split, subset)
            if cache:
                try:
                    with open(cache_path, "wb") as f:
                        pickle.dump(records, f)
                except OSError:
                    pass  # read-only dataset mount; just rescan next time

        # ---- filtering -------------------------------------------------------
        if viewpoints is not None:
            viewpoints = set(viewpoints)
            records = [r for r in records if r["viewpoint"] in viewpoints]
        if min_box_size > 0:
            records = [
                r
                for r in records
                if r["box"] is not None
                and min(r["box"][2] - r["box"][0], r["box"][3] - r["box"][1]) >= min_box_size
            ]
        if label_level == "type":
            # type id 0 means "unknown" in CompCars -- drop those.
            records = [r for r in records if r["type"] > 0]

        # compute shards
        records = records[shard::num_shards]
        self.files = [r["path"] for r in records]
        self.boxes = [r["box"] for r in records]
        self.viewpoints = [r["viewpoint"] for r in records]
        if label_level == "make":
            self.labels = [r["make"] - 1 for r in records]
        elif label_level == "type":
            self.labels = [r["type"] - 1 for r in records]
        else:
            self.labels = [r["model"] - 1 for r in records]
        self.targets = np.array(self.labels)

    def _scan(self, root, split, subset):
        """Read the split file and (for the web subset) every per-image label."""
        if subset == "web":
            split_file = os.path.join(
                root, "data", "train_test_split", "classification", f"{split}.txt"
            )
        else:
            split_file = os.path.join(root, "sv_data", f"{split}_surveillance.txt")

        with open(split_file, "r") as f:
            rel_paths = [line.strip() for line in f if line.strip()]

        records = []
        for rel in rel_paths:
            rel = rel.replace("\\", "/").lstrip("/")
            parts = rel.split("/")
            if subset == "web":
                make_id, model_id = int(parts[0]), int(parts[1])
                label_path = os.path.join(self.label_dir, rel).rsplit(".", 1)[0] + ".txt"
                viewpoint, car_type, box = self._read_label(label_path)
            else:
                # sv layout is <model_id>/<img>.jpg; no make / viewpoint / box.
                make_id, model_id = -1, int(parts[0])
                viewpoint, car_type, box = -1, 0, None
            records.append(
                {
                    "path": rel,
                    "make": make_id,
                    "model": model_id,
                    "type": car_type,
                    "viewpoint": viewpoint,
                    "box": box,
                }
            )
        return records

    @staticmethod
    def _read_label(label_path):
        """CompCars label file: viewpoint / body type / 'x1 y1 x2 y2'."""
        try:
            with open(label_path, "r") as f:
                lines = [line.strip() for line in f if line.strip()]
        except OSError:
            return -1, 0, None
        viewpoint = int(float(lines[0])) if len(lines) > 0 else -1
        car_type = int(float(lines[1])) if len(lines) > 1 else 0
        box = None
        if len(lines) > 2:
            coords = [int(float(v)) for v in lines[2].split()]
            if len(coords) == 4:
                box = tuple(coords)
        return viewpoint, car_type, box

    def __getitem__(self, index: Any) -> Any:
        img = Image.open(os.path.join(self.image_dir, self.files[index])).convert("RGB")
        if self.crop_to_bbox:
            img = _pad_and_crop(img, self.boxes[index], self.bbox_padding)
        if self.transform is not None:
            img = self.transform(img)
        label = self.labels[index]
        if self.return_index:
            return img, label, index
        else:
            return img, label

    def __len__(self):
        return len(self.files)

    def class_name(self, class_id) -> str:
        return self.classes[class_id]

    def get_class_names(self):
        return list(self.classes)


class BoxCars116k(Dataset):
    """BoxCars116k (Sochor et al., 2019) -- fine-grained vehicle recognition
    from real traffic-surveillance cameras, 116k images of 27k vehicles.

    Expects the official release layout::

        <root>/images/<sample_dir>/<instance>.png
        <root>/dataset.pkl
        <root>/classification_splits.pkl

    ``part`` selects the label granularity defined by the authors' splits:
    ``"hard"`` / ``"medium"`` (make-model-submodel-year), ``"submodel"``,
    ``"model"``, ``"make"``, ``"body"``. ``"hard"`` is the fine-grained setting.

    The released images are already cropped around the vehicle with a margin,
    so ``crop_to_bbox`` defaults to False; flip it on to tighten to the stored
    2D box if you want less background in the counterfactual.
    """

    PARTS = ("hard", "medium", "body", "make", "model", "submodel")

    def __init__(
        self,
        root,
        transform,
        split: str = "test",
        part: str = "hard",
        shard: int = 0,
        num_shards: int = 1,
        return_index: bool = True,
        crop_to_bbox: bool = False,
        bbox_padding: float = 0.05,
        min_box_size: int = 0,
        **kwargs,
    ) -> None:
        super().__init__()
        _ = kwargs  # Just for consistency with other datasets.

        assert part in BoxCars116k.PARTS
        if split == "val":
            split = "validation"
        assert split in ("train", "validation", "test")

        self.root = root
        self.image_dir = os.path.join(root, "images")
        self.transform = transform
        self.return_index = return_index
        self.crop_to_bbox = crop_to_bbox
        self.bbox_padding = bbox_padding

        dataset = self._load_pickle(os.path.join(root, "dataset.pkl"))
        splits = self._load_pickle(os.path.join(root, "classification_splits.pkl"))
        split_data = splits[part]

        # types_mapping is {class_name: class_idx}; invert into an ordered list.
        mapping = split_data["types_mapping"]
        self.classes = [""] * len(mapping)
        for key, value in mapping.items():
            if isinstance(value, (int, np.integer)):
                self.classes[int(value)] = str(key)
            else:  # tolerate the reverse convention
                self.classes[int(key)] = str(value)
        self.N_CLASSES = len(self.classes)

        files, labels, boxes = [], [], []
        for sample_id, class_id in split_data[split]:
            sample = dataset["samples"][int(sample_id)]
            for instance in sample["instances"]:
                box = instance.get("2DBB")
                if box is not None:
                    x, y, w, h = (int(v) for v in np.asarray(box).reshape(-1)[:4])
                    box = (x, y, x + w, y + h)
                    if min_box_size > 0 and min(w, h) < min_box_size:
                        continue
                files.append(instance["path"])
                labels.append(int(class_id))
                boxes.append(box)

        # compute shards
        self.files = files[shard::num_shards]
        self.labels = labels[shard::num_shards]
        self.boxes = boxes[shard::num_shards]
        self.targets = np.array(self.labels)

    @staticmethod
    def _load_pickle(path):
        with open(path, "rb") as f:
            try:
                return pickle.load(f, encoding="latin-1")
            except TypeError:  # pragma: no cover - py2 pickles only
                f.seek(0)
                return pickle.load(f)

    def __getitem__(self, index: Any) -> Any:
        img = Image.open(os.path.join(self.image_dir, self.files[index])).convert("RGB")
        if self.crop_to_bbox:
            img = _pad_and_crop(img, self.boxes[index], self.bbox_padding)
        if self.transform is not None:
            img = self.transform(img)
        label = self.labels[index]
        if self.return_index:
            return img, label, index
        else:
            return img, label

    def __len__(self):
        return len(self.files)

    def class_name(self, class_id) -> str:
        return self.classes[class_id]

    def get_class_names(self):
        return list(self.classes)
