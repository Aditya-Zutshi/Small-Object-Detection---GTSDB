import os
import numpy as np
import pandas as pd
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
import torchvision.transforms as T
from PIL import Image
from sklearn.model_selection import train_test_split

# Define custom dataset class for German Traffic Sign Detection Benchmark
class GermanTrafficSignDataset(torch.utils.data.Dataset):
    def __init__(self, root, annotations_file, transform=None):
        self.root = root
        self.transform = transform
        self.annotations = pd.read_csv(annotations_file, delimiter=';', header=None)
        self.annotations.columns = ['image_id', 'x1', 'y1', 'x2', 'y2', 'label']
        self.grouped_data = list(self.annotations.groupby('image_id').groups.items())[:10]

    def __len__(self):
        return len(self.grouped_data)

    def __getitem__(self, idx):
        image_id, group_indices = self.grouped_data[idx]
        group = self.annotations.iloc[group_indices]

        image = Image.open(os.path.join(self.root, image_id))
        boxes = torch.tensor(group[['x1', 'y1', 'x2', 'y2']].values, dtype=torch.float32)
        labels = torch.tensor(group['label'].values, dtype=torch.int64)

        targets = {'boxes': boxes, 'labels': labels}

        if self.transform:
            image = self.transform(image)

        return image, targets
def collate_fn(batch):
    return tuple(zip(*batch))

# Define transformation parameters
transform = T.Compose([
    T.ToTensor(),  # Convert to PyTorch tensor
])

# Load dataset and split into train and validation sets
dataset = GermanTrafficSignDataset('dataset/train', 'dataset/gt.txt', transform=transform)

# Create data loaders
dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0, collate_fn=collate_fn)
# Load the pretrained Faster R-CNN model
model = fasterrcnn_resnet50_fpn(pretrained=True)

# Replace the classifier with a new one for traffic sign detection
num_classes = 43  # Assuming there are 43 traffic sign classes
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# Train the model
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    count = 0
    for images, targets in dataloader:
        # Transfer data to device
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        loss = sum(loss for loss in loss_dict.values())

        # Check for NaNs in the loss
        if torch.isnan(loss):
            print(f'NaN loss encountered at epoch {epoch}, skipping batch...')
            continue

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        count += len(images)  # Update count based on batch size

        # Print training progress (every 100 images)
        if count % 32 == 0:
            print(f'Epoch {epoch}, Images Trained: {count}, Train Loss: {loss.item()}')
    print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {loss.item()}')

# Save the trained model
torch.save(model.state_dict(), 'trained_fasterrcnn_model.pth')
