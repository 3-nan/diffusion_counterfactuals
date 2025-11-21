from __future__ import print_function, division

import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

# import torchvision
# import torchvision.transforms as transforms
from torchvision import datasets, models

import timm

# from imutils import paths
# from pathlib import Path
import os
# import time
# import copy
# import pickle
# from tqdm import tqdm

import pandas as pd
import matplotlib.pylab as plt
import numpy as np

# Local modules
from cub_tools.train import train_model
# from cub_tools.visualize import imshow, visualize_model
from cub_tools.utils import save_model_dict, save_model_full, unpickle
from cub_tools.transforms import makeDefaultTransforms

if __name__ == '__main__':

    # Runtime setup
    # Script runtime options
    # model_name = 'vgg16_bn'

    model_name = "vgg16_bn"
    # model_name = "vit_base_patch16_224"
    # model_func = models.vgg16_bn # resnet152
    root_dir = '/Data/CUB_200_2011'
    data_dir = os.path.join(root_dir,'images')
    working_dir = os.path.join('/results/models', model_name)
    batch_size = 16
    num_workers = 4
    num_epochs = 40

    if not os.path.isdir('/Data/CUB_200_2011/images/train'):
        root_dir = '/Data/CUB_200_2011/'
        data_dir = os.path.join(root_dir,'images')

        image_fnames = pd.read_csv(filepath_or_buffer=os.path.join(root_dir,'images.txt'), 
                                header=None, 
                                delimiter=' ', 
                                names=['Img ID', 'file path'])

        image_fnames['is training image?'] = pd.read_csv(filepath_or_buffer=os.path.join(root_dir,'train_test_split.txt'), 
                                                        header=None, delimiter=' ', 
                                                        names=['Img ID','is training image?'])['is training image?']

        os.makedirs(os.path.join(data_dir,'train'), exist_ok=True)
        os.makedirs(os.path.join(data_dir,'test'), exist_ok=True)

        for i_image, image_fname in enumerate(image_fnames['file path']):
            if image_fnames['is training image?'].iloc[i_image]:
                new_dir = os.path.join(data_dir,'train',image_fname.split('/')[0])
                os.makedirs(new_dir, exist_ok=True)
                shutil.copy(src=os.path.join(data_dir,image_fname), dst=os.path.join(new_dir, image_fname.split('/')[1]))
                print(i_image, ':: Image is in training set. [', bool(image_fnames['is training image?'].iloc[i_image]),']')
                print('Image:: ', image_fname)
                print('Destination:: ', new_dir)
            else:
                new_dir = os.path.join(data_dir,'test',image_fname.split('/')[0])
                os.makedirs(new_dir, exist_ok=True)
                shutil.copy(src=os.path.join(data_dir,image_fname), dst=os.path.join(new_dir, image_fname.split('/')[1]))
                print(i_image, ':: Image is in testing set. [', bool(image_fnames['is training image?'].iloc[i_image]),']')
                print('Source Image:: ', image_fname)
                print('Destination:: ', new_dir)

    os.makedirs(working_dir, exist_ok=True)
    os.chmod(working_dir, 0o777)

    # Get data transforms
    data_transforms = makeDefaultTransforms()

    # Setup data loaders with augmentation transforms
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                    for x in ['train', 'test']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                                shuffle=True, num_workers=num_workers)
                for x in ['train', 'test']}
    batch = next(iter(dataloaders['train']))
    inputs, labels = batch

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
    class_names = image_datasets['train'].classes

    print('Number of data')
    print('========================================')
    for dataset in dataset_sizes.keys():
        print(dataset,' size:: ', dataset_sizes[dataset],' images')

    print('')
    print('Number of classes:: ', len(class_names))
    print('========================================')
    for i_class, class_name in enumerate(class_names):
        print(i_class,':: ',class_name)

    # Setup the device to run the computations
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Device::', device)

    # Setup the model and optimiser
    # model_ft = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=len(class_names))
    model_ft = timm.create_model(model_name, pretrained=True, num_classes=len(class_names))
    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    # num_epochs = 20  # 40

    model_ft = train_model(model=model_ft, criterion=criterion, optimizer=optimizer_ft, scheduler=exp_lr_scheduler, 
                        device=device, dataloaders=dataloaders, dataset_sizes=dataset_sizes, num_epochs=num_epochs,
                        working_dir=working_dir)

    # Load model training history
    # model_history = f'../models/classification/{model_name}/model_history.pkl'
    model_history = os.path.join(working_dir, f'model_history.pkl')
    history = unpickle(model_history)

    # history['epoch'] = [a.cpu() for a in history['epoch']]
    # history['train_loss'] = [a.cpu() for a in history['train_loss']]
    # history['test_loss'] = [a.cpu() for a in history['test_loss']]
    history['train_acc'] = [a.cpu() for a in history['train_acc']]
    history['test_acc'] = [a.cpu() for a in history['test_acc']]

    # print(type(history['epoch']))
    # print(type(history['train_loss']))

    plt.figure(figsize=(18,10))

    plt.subplot(2,1,1)
    plt.plot(np.arange(0, np.max(history['epoch'])+1,1), history['train_loss'], 'b-', label='Train')
    plt.plot(np.arange(0, np.max(history['epoch'])+1,1), history['test_loss'], 'r-', label='Test')
    plt.grid(True)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training / Validation Loss - Caltech Birds - {}'.format(model_name))
    plt.legend()

    plt.subplot(2,1,2)
    plt.plot(np.arange(0, np.max(history['epoch'])+1,1), history['train_acc'], 'b-', label='Train')
    plt.plot(np.arange(0, np.max(history['epoch'])+1,1), history['test_acc'], 'r-', label='Test')
    plt.grid(True)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training / Validation Accuracy - Caltech Birds - {}'.format(model_name))
    plt.legend()

    plt.savefig(os.path.join(working_dir, f'training_{model_name}.png'))

    save_model_full(model=model_ft, PATH=os.path.join(working_dir, f'caltech_birds_{model_name}_full_{history["test_acc"][-1]:.3f}.pth'))
    save_model_dict(model=model_ft, PATH=os.path.join(working_dir, f'caltech_birds_{model_name}_dict.pth'))
