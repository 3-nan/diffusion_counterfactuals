
import numpy as np
import timm
import torch
from torch import nn, optim
from torchvision import transforms
from torchvision.datasets import CelebA
from torch.utils.data import DataLoader


def generate_class_weights(class_series, class_labels):
    mlb = None
    n_samples = len(class_series)
    n_classes = len(class_series[0])

    class_count = [0] * n_classes
    for classes in class_series:
        for index in range(n_classes):
            if classes[index] != 0:
                class_count[index] += 1
    
    class_weights = [n_samples / (n_classes * freq) if freq > 0 else 1 for freq in class_count]
    class_labels = range(len(class_weights)) if mlb is None else mlb.classes_
    return dict(zip(class_labels, class_weights))

# class CelebADataset(Dataset):
#     def __init__(self, data, labels, transform=None):
#         self.data = data
#         self.labels = labels
#         self.transform = transform
#         self.classes = list(pd.read_csv(label_path).columns)
        
#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         img = Image.open(self.data[idx]).convert('RGB')
#         label = torch.Tensor(self.labels[idx])
#         if self.transform:
#             img = self.transform(img)
#         sample = {'image': img, 'label': label}
#         return sample

# def CelebA_DataLoader(batch_size, device):
#     num_workers = 0 if device.type == 'cuda' else 2
#     pin_memory = True if device.type == 'cuda' else False
#     classes = train_dataset.classes
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory,shuffle=True)
#     val_loader = DataLoader(val_dataset, num_workers=num_workers, pin_memory=pin_memory, batch_size=batch_size, shuffle=False)
#     test_loader = DataLoader(test_dataset, num_workers=num_workers, pin_memory=pin_memory, batch_size=batch_size, shuffle=False)
#     return train_loader, val_loader, test_loader


def hamming_score(y_true, y_pred):
    num_samples = len(y_true)
    total_correct = 0
    
    for true_labels, pred_labels in zip(y_true, y_pred):
        correct_labels = (true_labels == pred_labels).sum()
        total_correct += correct_labels
    hamming_score = total_correct / (num_samples * len(y_true[0]))
    return hamming_score

def get_multilabel_evaluation(model, test_loader):
    all_predictions = []
    all_targets = []
    model.eval()
    with torch.no_grad():
        for dir_ in test_loader:
            inputs, targets = dir_  # .values()
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            predictions = (outputs > 0.01).float()
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    return all_predictions, all_targets

import matplotlib.pyplot as plt

def plot_evaluation(train_losses, train_hamming_scores, val_losses, val_hamming_scores, precision, recall, f_score):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot Losses
    axes[0].plot(train_losses, label='Train Losses', marker='o')
    axes[0].plot(val_losses, label='Validation Losses', marker='o')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Train and Validation Losses vs. Epochs')
    axes[0].legend()
    axes[0].grid(True)

    # Plot Hamming Scores
    axes[1].plot(train_hamming_scores, label='Train Hamming Scores', marker='o')
    axes[1].plot(val_hamming_scores, label='Validation Hamming Scores', marker='o')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Hamming Score')
    axes[1].set_title('Train and Validation Hamming Scores')
    axes[1].legend()
    axes[1].grid(True)

    # Plot Precision, Recall, F-Score
    metrics = ['Precision', 'Recall', 'F-Score']
    values = [precision, recall, f_score]
    axes[2].bar(metrics, values, color=['blue', 'green', 'red'])
    axes[2].set_title('Score')
    axes[2].set_ylabel('Score')
    axes[2].grid(axis='y')
    axes[2].set_ylim(0, 1) 

    plt.tight_layout()
    plt.show()

from tqdm import tqdm

def train_model(model, device, train_loader, val_loader, criterion, optimizer, num_epochs=5):
    train_losses = []
    train_hamming_scores = []
    val_losses = []
    val_hamming_scores = []
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        total_train_loss = 0
        
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch')
        for dict_ in train_bar:
            inputs, labels = dict_  #.values()
            # print(type(inputs))
            # print(inputs.size())
            # print(inputs[0, 0, :10, :10])
            # print(labels)
            # print(labels.size())
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = torch.sigmoid(outputs)
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            predicted_labels = (outputs > 0.5).float()
            hamming_score_value = hamming_score(labels.cpu().numpy(), predicted_labels.cpu().numpy())
            train_bar.set_postfix(loss=loss.item(), hamming_score=hamming_score_value)
            
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)
        train_hamming_scores.append(hamming_score_value)

        model.eval()
        total_val_loss = 0
        
        with torch.no_grad():
            for dict_ in val_loader:
                inputs, labels = dict_      # .values()
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                outputs = torch.sigmoid(outputs)
                loss = criterion(outputs, labels.float())
                total_val_loss += loss.item() * inputs.size(0)

                predicted_labels = (outputs > 0.5).float()
                val_hamming_score = hamming_score(labels.cpu().numpy(), predicted_labels.cpu().numpy())

        val_loss = total_val_loss / len(val_loader.dataset)
        val_losses.append(val_loss)
        val_hamming_scores.append(val_hamming_score)

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Hamming Score: {hamming_score_value:.4f}, Val Loss: {val_loss:.4f}, Val Hamming Score: {val_hamming_score:.4f}")

        torch.save(model.state_dict(), f'/results/models/celeba_{model_name}_{val_hamming_score:.3f}.pth')
    
    return [train_losses, train_hamming_scores, val_losses, val_hamming_scores]

# def define_model(train_loader, val_loader, learning_rate, num_features, dropout_prob, device):
#     Model = resnet50(weights=ResNet50_Weights.DEFAULT)
#     Model.fc = nn.Sequential(
#         nn.Linear(Model.fc.in_features, num_features),
#         nn.ReLU(inplace=True),
#         nn.Dropout(dropout_prob),
#         nn.Linear(num_features, len(train_dataset.classes))
#     )
#     Model = Model.to(device)
#     weight_tensor = torch.tensor(list(class_weights.values()), dtype=torch.float).to(device)
#     criterion = nn.BCEWithLogitsLoss(weight=weight_tensor)    
#     optimizer = optim.Adam(Model.parameters(), lr=learning_rate)
#     Metric = []
#     Metric.append(train_model(Model, device, train_loader, val_loader, criterion, optimizer, num_epochs=10))

#     true, pred = get_multilabel_evaluation(Model, test_loader) 
#     predictions_np = np.concatenate(pred)
#     targets_np = np.concatenate(true)
#     precision, recall, f_score, _ = precision_recall_fscore_support(targets_np, predictions_np, average='weighted')
#     Metric.append([precision, recall, f_score])
#     return Model, Metric

def get_model(model_name, device):

    model = timm.create_model(model_name, pretrained=True, num_classes=40).to(device)

    return model

if __name__ == '__main__':

    # image_folder_path = r'Z:\ndm\Img-20240114T105609Z-001\Img\img_align_celeba\img_align_celeba'
    # label_path = r'attribute.csv'
    # eval_path = r'eval.csv'
    # eval_list = pd.read_csv(eval_path)['eval'].values  
    # eval_name = pd.read_csv(eval_path)['name'].values
    # labels = pd.read_csv(label_path).values

    # indx, indy, recall = [0]*3, [0]*3, 0
    # for i in eval_list:
    #     if recall == i - 1:
    #         recall = i
    #         indy[recall] += indy[recall - 1] + 1
    #         indx[recall] = indy[recall]
    #     else:
    #         indy[recall] += 1
    # print(indx,'\n',indy)

    # Dataset transforms
    # image_size = (64, 64)
    # data_transform=transforms.Compose([
    #     transforms.Resize(image_size),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.5, 0.5, 0.5],
    #                         std=[0.5, 0.5, 0.5])
    # ])
    # train_dataset = CelebADataset(train_list, train_label_list, data_transform)
    # val_dataset = CelebADataset(val_list, val_label_list, data_transform)
    # test_dataset = CelebADataset(test_list, test_label_list, data_transform)

    model_name = 'vit_base_patch16_224'         # vgg16_bn      resnet18     vit_b_16       vit_base_patch16_224
    root_dir = '/Data/'
    batch_size = 128    #64
    num_workers = 4
    learning_rate = 0.0001

    img_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    train_dataset = CelebA(root_dir, split='train', target_type='attr', transform=img_transforms)
    val_dataset = CelebA(root_dir, split='valid', target_type='attr', transform=img_transforms)
    test_dataset = CelebA(root_dir, split='test', target_type='attr', transform=img_transforms, target_transform=transforms.ToTensor())

    print(val_dataset.attr)

    class_weights = generate_class_weights(train_dataset.attr, np.arange(0,39))
    # class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y), y=y)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_model(model_name, device)

    weight_tensor = torch.tensor(list(class_weights.values()), dtype=torch.float).to(device)
    # criterion = nn.BCEWithLogitsLoss(weight=weight_tensor)
    criterion = nn.BCELoss(weight=weight_tensor)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses, train_hamming_scores, val_losses, val_hamming_scores = train_model(model, device, train_loader, val_loader, criterion, optimizer, num_epochs=5)

    # torch.save(model.state_dict(), f'/results/models/celeba_{model_name}_{np.mean(val_hamming_scores):.3f}.pth')
