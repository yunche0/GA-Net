import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel, ViTModel
import dgl
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import imghdr
from sklearn.metrics.pairwise import cosine_similarity
import csv

def load_data(label_file):
    images = []
    texts = []
    labels = []
    with open(label_file, 'r') as f:
        for line in f:
            parts = line.strip().split(';')
            if len(parts) < 3:
                print(f"Skipping incomplete line: {line}")
                continue
            image_path, text_description, label = parts
            if not os.path.exists(image_path):
                print(f"Image not found: {image_path}, skipping.")
                continue
            try:
                label = int(label)
            except ValueError:
                print(f"Invalid label for image {image_path}, using default label.")
                label = default_label
            images.append(image_path)
            texts.append(text_description)
            labels.append(label)
    print(f"Loaded {len(images)} images, {len(texts)} texts, and {len(labels)} labels.")
    return images, texts, labels

def is_image_file(filepath):
    image_formats = ("jpg", "jpeg", "png", "bmp", "gif")
    _, ext = os.path.splitext(filepath)
    ext = ext[1:].lower()
    if ext in image_formats and imghdr.what(filepath) is not None:
        return True
    return False

class CustomDataset(Dataset):
    def __init__(self, label_file, default_label=0):
        self.image_paths, self.texts, self.labels = load_data(label_file)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def process_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        return image

    def preprocess_texts(self, texts):
        return self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

    def __getitem__(self, idx):
        text = self.texts[idx]
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        image = self.process_image(image_path)
        return text, image, label

    def __len__(self):
        return len(self.image_paths)

def freeze_parameters(model):
    for param in model.text_encoder.parameters():
        param.requires_grad = False
    for param in model.visual_encoder.parameters():
        param.requires_grad = False

class LinearModel(nn.Module):
    def __init__(self, num_classes):
        super(LinearModel, self).__init__()
        self.text_encoder = BertModel.from_pretrained('bert-base-uncased')
        self.visual_encoder = ViTModel.from_pretrained('google/vit-base-patch16-224')
        self.visual_encoder.pooler = nn.Identity()
        self.linear = nn.Linear(768 + 768, num_classes)

    def forward(self, text_inputs, image_inputs):
        text_features = self.text_encoder(**text_inputs).last_hidden_state
        text_features = text_features[:, 0, :]
        visual_features = self.visual_encoder(pixel_values=image_inputs).last_hidden_state[:, 0, :]
        combined_features = torch.cat([text_features, visual_features], dim=1)
        outputs = self.linear(combined_features)
        return outputs

def evaluate(model, dataloader):
    model.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for texts, images, labels in dataloader:
            text_inputs = dataloader.dataset.preprocess_texts(texts)
            labels = labels.to(torch.long)
            outputs = model(text_inputs, images)
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    accuracy = total_correct / total_samples
    return accuracy

def log_experiment_results(epoch, loss, accuracy, filepath='experiment_results.csv'):
    with open(filepath, mode='a') as file:
        writer = csv.writer(file)
        writer.writerow([epoch, loss.item(), accuracy])

# 主程序
train_label_file = 'test/word/pet_file_list_train.txt'
test_label_file = 'test/word/pet_file_list_test.txt'
default_label = 0

train_dataset = CustomDataset(train_label_file, default_label)
test_dataset = CustomDataset(test_label_file, default_label)

train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

model = LinearModel(num_classes=37)

freeze_parameters(model)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (texts, images, labels) in enumerate(train_dataloader):
        optimizer.zero_grad()
        text_inputs = train_dataset.preprocess_texts(texts)
        labels = labels.to(torch.long)
        outputs = model(text_inputs, images)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    epoch_loss = running_loss / len(train_dataloader)
    accuracy = evaluate(model, test_dataloader)

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.4f}")

    log_experiment_results(epoch + 1, loss, accuracy)
