from datetime import datetime
from tqdm import tqdm
import timm
import torch
from torch import nn
from torchvision import datasets
import torchvision.transforms as T
from torchvision.models.vgg import vgg16_bn#, VGG16_BN_Weights
from torch.utils.tensorboard import SummaryWriter
# import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def get_model(model_name, num_classes):

    if model_name == "vgg16bn":
        model = vgg16_bn(pretrained=True)
        set_parameter_requires_grad(model, False)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, num_classes)
    else:
        model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
    return model

num_classes = 102

model_name = "vit_base_patch16_224"
# weights = VGG16_BN_Weights.DEFAULT
# model = vgg16_bn(weights=weights)
# model = vgg16_bn(pretrained=True)
# set_parameter_requires_grad(model, False)
# num_ftrs = model.classifier[6].in_features
# model.classifier[6] = nn.Linear(num_ftrs, num_classes)

model = get_model(model_name, num_classes)

model.to(device)

# transforms = model.transforms()

normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
transforms = T.Compose([T.Resize((224, 224)), T.ToTensor(), normalize])

training_data = datasets.Flowers102(
    root="/Data",
    split="train",
    download=True,
    transform=transforms,
    # collate_fn=utils.collate_fn
)

test_data = datasets.Flowers102(
    root="/Data",
    split="val",
    download=True,
    transform=transforms,
    # collate_fn=utils.collate_fn
)

# print(vars(training_data))
# print(training_data.class_to_idx)
# print(f'Number of classes: {len(training_data.class_to_idx)}')

# raise ValueError


training_loader = torch.utils.data.DataLoader(training_data, batch_size=128, shuffle=True, num_workers=8)
validation_loader = torch.utils.data.DataLoader(test_data, batch_size=128, shuffle=False, num_workers=8)

loss_fn = torch.nn.CrossEntropyLoss()

# NB: Loss functions expect data in batches, so we're creating batches of 4
# Represents the model's confidence in each of the 10 classes for a given input
# dummy_outputs = torch.rand(4, 10)
# # Represents the correct class among the 10 being tested
# dummy_labels = torch.tensor([1, 5, 3, 7])

# print(dummy_outputs)
# print(dummy_labels)

# loss = loss_fn(dummy_outputs, dummy_labels)

optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(training_loader))

def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.

    print(f'Starting epoch {epoch_index}')

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(tqdm(training_loader)):
        # Every data instance is an input + label pair
        inputs, labels = data[0].to(device), data[1].type(torch.LongTensor).to(device)

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        with torch.autocast('cuda'):
            # Compute the loss and its gradients
            loss = loss_fn(outputs, labels)
            loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % len(training_loader) == (len(training_loader) - 1):
            last_loss = running_loss / len(training_loader) # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss

# Initializing in a separate cell so we can easily add more epochs to the same run
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('runs/pets_trainer_{}'.format(timestamp))
epoch_number = 0

EPOCHS = 80

best_vloss = 1_000_000.

for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch_number + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss = train_one_epoch(epoch_number, writer)

    scheduler.step()

    running_vloss = 0.0
    # Set the model to evaluation mode, disabling dropout and using population
    # statistics for batch normalization.
    model.eval()

    correct = 0
    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        for i, vdata in enumerate(validation_loader):
            vinputs, vlabels = vdata[0].to(device), vdata[1].to(device)
            vinputs = vinputs.to(device)
            vlabels = vlabels.to(device)
            voutputs = model(vinputs)
            vloss = loss_fn(voutputs, vlabels)
            running_vloss += vloss

            vpred = voutputs.max(1).indices
            correct += (vpred == vlabels).float().sum()

    accuracy = correct / len(validation_loader.dataset)
    # trainset, not train_loader

    avg_vloss = running_vloss / (i + 1)
    print(f'LOSS train {avg_loss} valid {avg_vloss} acc {accuracy}')

    # Log the running loss averaged per batch
    # for both training and validation
    writer.add_scalars('Training vs. Validation Loss',
                    { 'Training' : avg_loss, 'Validation' : avg_vloss },
                    epoch_number + 1)
    writer.flush()

    # Track best performance, and save the model's state
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = f'/results/models/{model_name}_flowers_{timestamp}_{epoch_number}_{accuracy:.3f}'
        torch.save(model.state_dict(), model_path)

    epoch_number += 1
