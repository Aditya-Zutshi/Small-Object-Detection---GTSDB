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
    def __init__(self, root, annotations_file, transform=None, type="train"):
        self.root = root
        self.transform = transform
        self.annotations = pd.read_csv(annotations_file, delimiter=';')
        if type == "train":
            self.data = self.annotations.iloc[:int(0.8 * len(self.annotations))]
        else:
            self.data = self.annotations.iloc[int(0.8 * len(self.annotations)):]
        print(f'{type} size: ', len(self.data))

    def __getitem__(self, idx):
        img_name = os.path.join(self.root, self.data.iloc[idx, 0])
        image = Image.open(img_name).convert("RGB")  # Load image using PIL and convert to RGB
        target = {}
        target['boxes'] = torch.tensor([
            [
                self.data.iloc[idx, 1],  # leftmost column
                self.data.iloc[idx, 2],  # upmost row
                self.data.iloc[idx, 3],  # rightmost column
                self.data.iloc[idx, 4]  # downmost row
            ]
        ], dtype=torch.float32)
        target['labels'] = torch.tensor([self.data.iloc[idx, 5]], dtype=torch.int64)  # class ID

        if self.transform is not None:
            image = self.transform(image)

        return [image, target]

    def __len__(self):
        return len(self.data)

def separate_targets(input_data):
    output_data = []
    for key, values in input_data.items():
        for i, value in enumerate(values):
            if len(output_data) <= i:
                output_data.append({})
            output_data[i][key] = value

    return output_data

def calculate_validation_loss(predicted_boxes, expected_boxes):
    val_losses = []
    for pred_dict, gt_dict in zip(predicted_boxes, expected_boxes):
        pred_boxes = pred_dict['boxes'].to(device)
        pred_labels = pred_dict['labels'].to(device)
        gt_boxes = gt_dict['boxes'].to(device)
        gt_labels = gt_dict['labels'].to(device)

        if len(pred_boxes) == 0:
            # Handle case where there are no predicted boxes
            continue

        if len(pred_labels) == 0:
            # Handle case where there are no predicted labels
            continue

        # Check if sizes match for loss calculations
        if pred_boxes.size(0) != gt_boxes.size(0):
            print("Size mismatch for boxes:", pred_boxes.size(), gt_boxes.size())
            continue

        if pred_labels.size(0) != gt_labels.size(0):
            print("Size mismatch for labels:", pred_labels.size(), gt_labels.size())
            continue

        # Calculate losses only if there are predictions
        box_loss = torch.nn.functional.smooth_l1_loss(pred_boxes, gt_boxes)
        label_loss = torch.nn.functional.cross_entropy(pred_labels, gt_labels)

        total_loss = box_loss + label_loss
        val_losses.append(total_loss.item())
    return val_losses

# Define transformation parameters
transform = T.Compose([
    # TODO: check if there is restriction on input image size for rcnn
    T.Resize(256),  # Resize to 256x256
    T.RandomHorizontalFlip(p=0.5),  # Apply horizontal flip with 50% probability
    T.ToTensor(),  # Convert to PyTorch tensor
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
])

# Usage example
train_dataset = GermanTrafficSignDataset('../dataset/train', '../dataset/gt.txt', transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)

val_dataset = GermanTrafficSignDataset('../dataset/train', '../dataset/gt.txt', transform=transform, type="val")
val_dataloader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=0)

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
    for images, targets in train_dataloader:
        # Transfer data to device
        images = list(image.to(device) for image in images)
        targets = separate_targets(targets.copy())

        # Train the model
        loss_dict = model(images, targets)
        loss = sum(loss for loss in loss_dict.values())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        count += len(images)  # Update count based on batch size

        # Print training progress (every 100 images)
        if count % 4 == 0:
            print(f'Epoch {epoch}, Images Trained: {count}')

    # Validation
    model.eval()
    val_loss = 0.0

    val_losses = []
    with torch.no_grad():
        for val_images, val_targets in val_dataloader:
            val_images = list(image.to(device) for image in val_images)
            val_targets = separate_targets(val_targets.copy())

            # Calculate validation loss
            predictions = model(val_images)
            val_losses += calculate_validation_loss(predictions, val_targets)

    avg_val_loss = np.mean(val_losses)
    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {loss.item()}, Validation Loss: {avg_val_loss}')

# Save the trained model
torch.save(model.state_dict(), 'trained_fasterrcnn_model.pth')
