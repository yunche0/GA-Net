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
import imghdr  # Python built-in library for detecting image file types
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
            if not is_image_file(image_path):
                print(f"File is not an image: {image_path}, skipping.")
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
    """Check if the file is an image file"""
    image_formats = ("jpg", "jpeg", "png", "bmp", "gif")
    _, ext = os.path.splitext(filepath)
    ext = ext[1:].lower()  # Remove the dot at the beginning and convert to lowercase
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
    # Freeze text encoder parameters
    for param in model.text_encoder.parameters():
        param.requires_grad = False
    # Freeze visual encoder parameters
    for param in model.visual_encoder.parameters():
        param.requires_grad = False

class GraphAdapter(nn.Module):
    def __init__(self, num_classes):
        super(GraphAdapter, self).__init__()
        self.text_encoder = BertModel.from_pretrained('bert-base-uncased')
        self.visual_encoder = ViTModel.from_pretrained('google/vit-base-patch16-224')
        self.visual_encoder.pooler = nn.Identity()
        self.gcn = dgl.nn.GraphConv(768 + 768, num_classes)

    def forward(self, text_inputs, image_inputs, graph):
        text_features = self.text_encoder(**text_inputs).last_hidden_state
        text_features = text_features[:, 0, :]
        visual_features = self.visual_encoder(pixel_values=image_inputs).last_hidden_state[:, 0, :]
        combined_features = torch.cat([text_features, visual_features], dim=1)
        graph.ndata['h'] = combined_features
        graph_outputs = self.gcn(graph, graph.ndata['h'])
        return graph_outputs

class AdapterGNN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(AdapterGNN, self).__init__()
        self.down = nn.Linear(in_dim, out_dim)
        self.up = nn.Linear(out_dim, in_dim)
        self.gnn = dgl.nn.GraphConv(out_dim, out_dim)

    def forward(self, graph, features):
        down_features = self.down(features)
        gnn_features = self.gnn(graph, down_features)
        up_features = self.up(gnn_features)
        return up_features

class EWC:
    def __init__(self, model, dataloader, tokenizer):
        self.model = model
        self.dataloader = dataloader
        self.tokenizer = tokenizer
        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self._means = {}
        self._precision_matrices = self._diag_fisher()
        for n, p in self.params.items():
            self._means[n] = p.data

    def _diag_fisher(self):
        precision_matrices = {n: torch.zeros_like(p) for n, p in self.params.items()}
        self.model.eval()
        for texts, images, labels in self.dataloader:
            text_inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
            labels = labels.to(torch.long)
            src, dst = create_graph_based_on_similarity(texts, images, self.model, self.dataloader.dataset)
            graph = dgl.graph((src, dst), num_nodes=len(texts))
            graph = dgl.add_self_loop(graph)
            self.model.zero_grad()
            outputs = self.model(text_inputs, images, graph)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            for n, p in self.model.named_parameters():
                if p.requires_grad:
                    precision_matrices[n].data += p.grad.data ** 2 / len(self.dataloader)
        precision_matrices = {n: p for n, p in precision_matrices.items()}
        return precision_matrices

    def penalty(self, model):
        loss = 0
        for n, p in model.named_parameters():
            if n in self._means:
                _loss = self._precision_matrices[n] * (p - self._means[n]) ** 2
                loss += _loss.sum()
        return loss

def evaluate(model, dataloader):
    model.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for texts, images, labels in dataloader:
            text_inputs = dataloader.dataset.preprocess_texts(texts)
            labels = labels.to(torch.long)
            src, dst = create_graph_based_on_similarity(texts, images, model, dataloader.dataset)
            graph = dgl.graph((src, dst), num_nodes=len(texts))
            graph = dgl.add_self_loop(graph)
            text_features = model.text_encoder(**text_inputs).last_hidden_state
            text_features = text_features[:, 0, :]
            visual_features = model.visual_encoder(pixel_values=images).last_hidden_state[:, 0, :]
            combined_features = torch.cat([text_features, visual_features], dim=1)
            adapter_outputs = adapter_gnn(graph, combined_features)
            outputs = model.gcn(graph, adapter_outputs)
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    accuracy = total_correct / total_samples
    return accuracy

def create_graph_based_on_similarity(texts, images, model, dataset):
    # Get features
    text_inputs = dataset.preprocess_texts(texts)
    with torch.no_grad():
        text_features = model.text_encoder(**text_inputs).last_hidden_state
        text_features = text_features[:, 0, :]
        visual_features = model.visual_encoder(pixel_values=images).last_hidden_state[:, 0, :]
        combined_features = torch.cat([text_features, visual_features], dim=1)

    # Compute similarity
    similarity_matrix = cosine_similarity(combined_features.cpu().numpy())

    # Create graph based on similarity
    src, dst = [], []
    threshold = 0.7   # Set a similarity threshold
    for i in range(len(similarity_matrix)):
        for j in range(i + 1, len(similarity_matrix)):
            if similarity_matrix[i][j] > threshold:
                src.append(i)
                dst.append(j)
                src.append(j)
                dst.append(i)
    return src, dst

def log_experiment_results(epoch, loss, accuracy, filepath='experiment_results.csv'):
    with open(filepath, mode='a') as file:
        writer = csv.writer(file)
        writer.writerow([epoch, loss.item(), accuracy])

# Main program
# Data paths configuration
train_label_file = 'test/word/flower_file_list_train.txt'   # Training set label file path
test_label_file = 'test/word/flower_file_list_test.txt'    # Test set label file path
default_label = 0  # Default label value

# 创建数据集和数据加载器
train_dataset = CustomDataset(train_label_file, default_label)
test_dataset = CustomDataset(test_label_file, default_label)

train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

model = GraphAdapter(num_classes=102)
adapter_gnn = AdapterGNN(768 + 768, 512)


freeze_parameters(model)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
ewc = EWC(model, train_dataloader, train_dataset.tokenizer)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (texts, images, labels) in enumerate(train_dataloader):
        optimizer.zero_grad()
        text_inputs = train_dataset.preprocess_texts(texts)
        labels = labels.to(torch.long)
        src, dst = create_graph_based_on_similarity(texts, images, model, train_dataset)
        graph = dgl.graph((src, dst), num_nodes=len(texts))
        graph = dgl.add_self_loop(graph)
        text_features = model.text_encoder(**text_inputs).last_hidden_state
        text_features = text_features[:, 0, :]
        visual_features = model.visual_encoder(pixel_values=images).last_hidden_state[:, 0, :]
        combined_features = torch.cat([text_features, visual_features], dim=1)
        adapter_outputs = adapter_gnn(graph, combined_features)
        outputs = model.gcn(graph, adapter_outputs)
        //outputs=adapter_outputs
        loss = F.cross_entropy(outputs, labels)
        ewc_loss = ewc.penalty(model)
        total_loss = loss + ewc_loss
        total_loss.backward()
        optimizer.step()
        running_loss += total_loss.item()

    epoch_loss = running_loss / len(train_dataloader)
    accuracy = evaluate(model, test_dataloader)


    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.4f}")


    log_experiment_results(epoch + 1, total_loss, accuracy)
