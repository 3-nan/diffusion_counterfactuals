
# import pandas as pd


# frames = []
# folder_path = "data/"

# for i in range(5):
#     temp_df = pd.read_csv(folder_path + "fold_" + str(i) + "_data.txt", delimiter="\t")
#     frames.append(temp_df)

# df = pd.concat(frames)

# map_dict = {
#     "13": "(08, 12)",
#     "2" : "(00, 02)",
#     "22": "(15, 20)",
#     "23": "(25, 32)",
#     "29": "(25, 32)",
#     "3" : "(00, 02)",
#     "32": "(25, 32)",
#     "34": "(25, 32)",
#     "35": "(25, 32)",
#     "36": "(38, 43)",
#     "42": "(38, 43)",
#     "45": "(38, 43)",
#     "46": "(48, 53)",
#     "55": "(48, 53)",
#     "56": "(48, 53)",
#     "57": "(60, 100)",
#     "58": "(60, 100)",
    
#     "(8, 23)" : "(08, 12)",
#     "(27, 32)": "(25, 32)",
#     "(38, 42)": "(38, 43)",
#     "(38, 48)": "(38, 43)",

#     "(00, 02)" : "(00, 02)",
#     "(04, 06)" : "(04, 06)",
#     "(08, 12)" : "(08, 12)",
#     "(15, 20)" : "(15, 20)",
#     "(25, 32)" : "(25, 32)",
#     "(38, 43)" : "(38, 43)",
#     "(48, 53)" : "(48, 53)",
#     "(60, 100)": "(60, 100)"
# }


# def map_func(x):
#     if x in map_dict:
#         return map_dict[x]
#     else:
#         return x

# df["age"] = df["age"].map(map_func)
# df["age"].value_counts()


# # Restructure directory
# import os
# import shutil
# folder_path = "/content/data/faces/"
# file_list = []
# formats = ["jpg", "png"]

# for subdir in os.listdir(folder_path):
#     subpath = os.path.join(folder_path, subdir)
#     if os.path.isdir(subpath):
#         for f in os.listdir(subpath):
#             filepath = os.path.join(subpath, f)
#             part = f.split(".")
#             if os.path.isfile(filepath) and part[-1] in formats:
#                 file_list.append((filepath, f))

# print(len(file_list))
# # DONT INTERRUPT WHILE RUNNING
# # SHOWS PROGRESS

# im_len = len(file_list)

# for i, (filepath, filename) in enumerate(file_list):

#     # get the identifiers
#     parts = filename.split(".")
#     user_id = filepath.split("/")[-2]
#     file_id = parts[-2] + "." + parts[-1]
#     face_id = int(parts[-3])

#     # find class
#     class_ = df[
#        (df["user_id"] == user_id) & 
#        (df["original_image"] == file_id) & 
#        (df["face_id"] == face_id)
#     ]["age"].values[0]

#     new_path = os.path.join(folder_path, class_)
    
#     if not os.path.exists(new_path):
#         os.makedirs(new_path)

#     # move file
#     new_path = os.path.join(new_path, filename)
#     shutil.move(filepath, new_path)
    
#     # progress
#     prog = (20 * (i + 1)) // im_len
#     print("\r[" + "="*prog + "_"*(20-prog) + "]", end="")
# # delete unwanted files or empty folders
# folder_path = "/content/data/faces/"

# for subdir in os.listdir(folder_path):
#     subpath = os.path.join(folder_path, subdir)

#     if os.path.isdir(subpath):
#         if subdir[0] != "(":
#             shutil.rmtree(subpath)
#     elif os.path.isfile(subpath):
#         os.remove(subdir)
# # list the classes
# # ! ls data/faces



# import torch
# import source.models as models
# import source.worker as worker
# import source.loader as loader
# # the mean and std of dataset are found by running this
# # takes some time to iterate twice
# loader.find_mean_std("data/faces")
# loader.random_scale = (0.8, 1.0)
# loader.mean = [0.437, 0.340, 0.304]
# loader.std  = [0.286, 0.252, 0.236]
# # Dataset Loader to feed into network
# # 20% of data is used for validation
# loaders = loader.split_loader("data/faces", valid_frac=0.2, batch_size=32)
# # pretrained weights - for convolution layers
# state = loader.load_pth("weight/vgg_face_dag.pth")
# # Model initialization
# model = models.vgg16(num_classes=8)
# # pretrained vgg-face
# model.load_weights(state)

# worker.train(model, loaders, lr=0.01, epochs=3)

# check = loader.load_pth("checkpoint.pth")
# model.load_weights(check["state_dict"])
# valid_loader = loaders[1]
# conf_mat = worker.confusion_matrix(model, valid_loader)

# for row in conf_mat:
#     for elem in row:
#         print("%.2f"%(elem*100), end="\t")
#     print("")

# # one-off accuracy
# ncls = len(conf_mat)
# tot_acc = 0

# for i in range(ncls):
    
#     acc = conf_mat[i][i]

#     # add left
#     if i > 0:
#         acc += conf_mat[i][i-1]
    
#     if i < ncls - 1:
#         acc += conf_mat[i][i+1]
    
#     tot_acc += acc

# tot_acc = tot_acc / ncls
# print("%.2f" % (tot_acc * 100))

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from shutil import copyfile

import timm
import torch
import torch.autograd.variable as Variable
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as utils
from torch.utils.data import Dataset, DataLoader


class AdienceDataset(Dataset):
    
    def __init__(self, txt_file, root_dir, transform):
        self.txt_file = txt_file
        self.root_dir = root_dir
        self.transform = transform
        self.data = self.read_from_txt_file()
    
    def __len__(self):
        return len(self.data)

    def read_from_txt_file(self):
        data = []
        f = open(self.txt_file)
        for line in f.readlines():
            image_file, label = line.split()
            label = int(label)
            if 'gender' in self.txt_file:
                label += 8
            data.append((image_file, label))
        return data
    
    def __getitem__(self, idx):
        img_name, label = self.data[idx]
        image = Image.open(self.root_dir + '/' + img_name)
        
        if self.transform:
            image = self.transform(image)
            
        return {
            'image': image,
            'label': label
        }

transforms_list = [
    transforms.Resize(256),
    transforms.CenterCrop(227),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.RandomCrop(227)
]

transforms_dict = {
    'train': {
        0: list(transforms_list[i] for i in [0, 1, 3]),        # no transformation
        1: list(transforms_list[i] for i in [0, 1, 2, 3]),     # random horizontal flip
        2: list(transforms_list[i] for i in [0, 4, 2, 3])      # random crop and random horizontal flip
    },
    'val': {
        0: list(transforms_list[i] for i in [0, 1, 3])
    },
    'test': {
        0: list(transforms_list[i] for i in [0, 1, 3])
    }
}

def get_dataloader(s, c, fold, transform_index, minibatch_size):
    """
    Args:
        s: A string. Equals either "train", "val", or "test".
        c: A string. Equals either "age" or "gender".
        fold: An integer. Lies in the range [0, 4] as there are five folds present.
        transform_index: An integer. The transforms in the list correesponding
            to this index in the dictionary will be applied on the images.
        minibatch_size: An integer.

    Returns:
        An instance of the DataLoader class.
    """
    txt_file = f'/Data/Adience/test_fold_is_{fold}/{c}_{s}.txt'
    root_dir = '/Data/Adience'
    
    transformed_dataset = AdienceDataset(txt_file, root_dir,
                                         transforms.Compose(transforms_dict[s][transform_index]))
    dataloader = DataLoader(transformed_dataset, batch_size=minibatch_size, shuffle=True, num_workers=4)
    
    return dataloader


def train(net, train_dataloader, epochs, filename, checkpoint_frequency, val_dataloader=None):
    """
    Args:
        net: An instance of PyTorch's Net class.
        train_dataloader: An instance of PyTorch's Dataloader class.
        epochs: An integer.
        filename: A string. Name of the model saved to drive.
        checkpoint_frequency: An integer. Represents how frequent (in terms
            of number of iterations) the model should be saved to drive.
        val_dataloader: An instance of PyTorch's Dataloader class.
    
    Returns:
        net: An instance of PyTorch's Net class. The trained network.
        training_loss: A list of numbers that represents the training loss at each checkpoint.
        validation_loss: A list of numbers that represents the validation loss at each checkpoint.
    """
    net.train()
    optimizer = optim.Adam(net.parameters(), lr)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10000])
    
    training_loss, validation_loss = [], []
    checkpoint = 0
    iteration = 0
    running_loss = 0
    
    for epoch in range(epochs):
        
        for i, batch in enumerate(train_dataloader):
            scheduler.step()
            optimizer.zero_grad()
            images, labels = batch['image'].to(device), batch['label'].to(device)
            outputs = net(images)
            loss = criterion(outputs, labels)
            running_loss += float(loss.item())
            loss.backward()
            optimizer.step()
                                    
            if (iteration+1) % checkpoint_frequency == 0 and val_dataloader is not None:
                training_loss.append(running_loss/checkpoint_frequency)
                validation_loss.append(validate(net, val_dataloader))
                print(f'minibatch:{i}, epoch:{epoch}, iteration:{iteration}, training_error:{training_loss[-1]}, validation_error:{validation_loss[-1]}')
                save_network(net, f'{filename}_checkpoint{checkpoint}')
                checkpoint += 1
                running_loss = 0
            
            iteration += 1

    return net, training_loss, validation_loss

def validate(net, dataloader):
    net.train()
    total_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            images, labels = batch['image'].to(device), batch['label'].to(device)
            outputs = net(images)
            loss = criterion(outputs, labels)
            total_loss += float(loss.item())

    return total_loss/(i+1)
     

def get_validation_error(c, fold, train_transform_index):
    filename = get_model_filename(c, fold, train_transform_index)
    net = Net().to(device)
    net.load_state_dict(torch.load(f'{PATH_TO_MODELS}/{filename}'))
    return validate(net, get_dataloader('val', c, fold, 0, minibatch_size))

def test(net, dataloader, c):
    result = {
        'exact_match': 0,
        'total': 0
    }
    if c == 'age':
        result['one_off_match'] = 0

    with torch.no_grad():
        net.eval()
        for i, batch in enumerate(dataloader):
            images, labels = batch['image'].to(device), batch['label'].to(device)
            outputs = net(images)
            outputs = torch.tensor(list(map(lambda x: torch.max(x, 0)[1], outputs))).to(device)
            result['total'] += len(outputs)
            result['exact_match'] += sum(outputs == labels).item()
            if c == 'age':
                result['one_off_match'] += (sum(outputs==labels) +
                                            sum(outputs==labels-1) +
                                            sum(outputs==labels+1)).item()

    return result

def save_network(net, filename):
    torch.save(net.state_dict(), f'{PATH_TO_MODELS}/{filename}.pt')


def train_save(c, fold, train_transform_index, checkpoint_frequency=50):
    """
    Args:
        c: A string. Equals either "age" or "gender".
        fold: An integer. Lies in the range [0, 4] as there are five folds present.
        train_transform_index: An integer. The transforms in the list correesponding
            to this index in the dictionary will be applied on the images.
        checkpoint_frequency: An integer. Represents how frequent (in terms
            of number of iterations) the model should be saved to drive.   
    Returns:
        validation_loss: A list of numbers that represents the validation loss at each checkpoint.
    """
    minibatch_size = 8
    trained_net, training_loss, validation_loss = train(
        timm.create_model(model_name, pretrained=True, num_classes=8).to(device),
        get_dataloader('train', c, fold, train_transform_index, minibatch_size),
        30,
        f'{fold}_{c}_train_{train_transform_index}',
        checkpoint_frequency,
        get_dataloader('val', c, fold, 0, minibatch_size)
    )
    
    plt.plot(list(map(lambda x: checkpoint_frequency * x, (list(range(1, len(validation_loss)+1))))), validation_loss, label='validation_loss')
    plt.plot(list(map(lambda x: checkpoint_frequency * x, (list(range(1, len(training_loss)+1))))), training_loss, label='training_loss')
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
    plt.xlabel('iterations')
    plt.ylabel('loss')
    plt.show()
    
    choose_model_with_least_val_error(c, fold, train_transform_index, validation_loss)
    
    return validation_loss
     

def choose_model_with_least_val_error(c, fold, train_transform_index, validation_loss):
    index = validation_loss.index(min(validation_loss))
    filename = f'{fold}_{c}_train_{train_transform_index}'
    for file in os.listdir(PATH_TO_MODELS):
        if file.startswith(filename):
            if file.startswith(f'{filename}_checkpoint{index}'):
                pass
            else:
                os.remove(f'{PATH_TO_MODELS}/{file}')

def pick_best_model(c):
    """
    Args:
        s: A string. Equals either "train", "val", or "test".
        c: A string. Equals either "age" or "gender".
    """
    def fn_filter(file):
        file_split = file.split('_')
        return True if (len(file_split) == 5 and file_split[1] == c) else False
    
    def fn_map(file):
        file_split = file.split('_')
        return get_validation_error(c, file_split[0], file_split[3])
    
    files = list(filter(fn_filter, os.listdir(PATH_TO_MODELS)))
    val_errors = list(map(fn_map, files))
    min_val_error, file = min(zip(val_errors, files))
    best_model = f'{PATH_TO_MODELS}/{file.split(".")[0]}_best.pt'
    copyfile(f'{PATH_TO_MODELS}/{file}', best_model)
    
    print(f'Picking {best_model} as the best model for {c}...')

def get_performance(c):
    """
    Args:
        c: A string. Equals either "age" or "gender".
    Returns:
        A dictionary containing accuracy (and one-off accuracy for age) of the model.
    """    
    file = get_best_model_filename(c).split('_')
    return get_performance_of_a_model('test', file[1], file[0], file[3])
     

def get_best_model_filename(c):
    def fn_filter(file):
        file_split = file.split('_')
        return True if (len(file_split) == 6 and file_split[1] == c) else False
    
    return list(filter(fn_filter, os.listdir(PATH_TO_MODELS)))[0]
     

def get_performance_of_a_model(s, c, fold, train_transform_index):
    """
    Args:
        s: A string. Equals either "train", "val", or "test".
        c: A string. Equals either "age" or "gender".
        fold: An integer. Lies in the range [0, 4] as there are five folds present.
        transform_index: An integer. The transforms in the list correesponding
            to this index in the dictionary will be applied on the images.
    Returns:
        A dictionary containing accuracy (and one-off accuracy for age) of the model.
    """
    filename = get_model_filename(c, fold, train_transform_index)
    net = Net().to(device)
    net.load_state_dict(torch.load(f'{PATH_TO_MODELS}/{filename}'))
    performance = test(
        net,
        get_dataloader(s, c, fold, 0, minibatch_size),
        c
    )
    if c == 'age':
        return {
            'accuracy': performance['exact_match']/performance['total'],
            'one-off accuracy': performance['one_off_match']/performance['total']
        }
    elif c == 'gender':
        return {
            'accuracy': performance['exact_match']/performance['total']
        }
     

def get_model_filename(c, fold, train_transform_index):
    start_of_filename = f'{fold}_{c}_train_{train_transform_index}_checkpoint'
    for file in os.listdir(PATH_TO_MODELS):
        if file.startswith(start_of_filename):
            return file

if __name__ == '__main__':

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        

    PATH_TO_MODELS = "/results/models"
    model_name = 'vgg16'

    # pretrained weights - for convolution layers
    state = torch.load(os.path.join(PATH_TO_MODELS, "vgg_face_dag.pth"))
    # Model initialization
    # model = models.vgg16(num_classes=8)
    model = timm.create_model(model_name, pretrained=True, num_classes=8)
    model = model.to(device)
    # pretrained vgg-face
    # model.load_state_dict(state)

    # dataloader = get_dataloader("train", "age")

    criterion = nn.NLLLoss()

    train_save('age', {0,1,2,3,4}, {0,1,2})
