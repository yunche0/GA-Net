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
import faiss

# 检查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
                label = 0  # 默认标签值
            images.append(image_path)
            texts.append(text_description)
            labels.append(label)
    print(f"Loaded {len(images)} images, {len(texts)} texts, and {len(labels)} labels.")
    return images, texts, labels


def is_image_file(filepath):
    """Check if the file is an image file"""
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


class AdaptiveFeatureFusion(nn.Module):
    def __init__(self, text_dim=768, image_dim=768, hidden_dim=256):
        super().__init__()
        # 特征投影层（统一维度）
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.image_proj = nn.Linear(image_dim, hidden_dim)

        # 注意力权重生成器
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
            nn.Softmax(dim=-1)
        )

    def forward(self, text_feat, image_feat):
        # 特征投影
        text_proj = self.text_proj(text_feat)
        image_proj = self.image_proj(image_feat)

        # 拼接特征生成注意力权重
        combined = torch.cat([text_proj, image_proj], dim=1)
        attn_weights = self.attention(combined)  # [batch_size, 2]

        # 加权融合
        text_weight = attn_weights[:, 0].unsqueeze(1)
        image_weight = attn_weights[:, 1].unsqueeze(1)
        fused_feature = text_weight * text_feat + image_weight * image_feat

        return fused_feature


class GraphAdapter(nn.Module):
    def __init__(self, num_classes):
        super(GraphAdapter, self).__init__()
        self.text_encoder = BertModel.from_pretrained('bert-base-uncased')
        self.visual_encoder = ViTModel.from_pretrained('google/vit-base-patch16-224')
        self.visual_encoder.pooler = nn.Identity()

        # 新增自适应融合模块
        self.fusion_layer = AdaptiveFeatureFusion()

        # 调整GCN输入维度（保持与原始维度一致）
        self.gcn = dgl.nn.GraphConv(768, num_classes)  # 输入维度改为768
        #self.gcn = dgl.nn.SAGEConv(768, num_classes, aggregator_type='mean')
        self.to(device)

    def forward(self, text_inputs, image_inputs, graph):
        # 特征提取
        text_features = self.text_encoder( ** text_inputs).last_hidden_state[:, 0, :]
        visual_features = self.visual_encoder(pixel_values=image_inputs).last_hidden_state[:, 0, :]

        # 自适应特征融合（替换原始拼接）
        combined_features = self.fusion_layer(text_features, visual_features)

        # 图网络处理
        graph.ndata['h'] = combined_features
        graph = graph.to(device)
        graph_outputs = self.gcn(graph, graph.ndata['h'])
        return graph_outputs


class AdapterGNN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(AdapterGNN, self).__init__()
        # 调整维度参数
        self.down = nn.Linear(in_dim, out_dim)
        self.up = nn.Linear(out_dim, 768)  # 输出维度调整为768
        #self.gnn = dgl.nn.SAGEConv(out_dim, out_dim, aggregator_type='mean')
        self.gnn = dgl.nn.GraphConv(out_dim, out_dim)

    def forward(self, graph, features):
        down_features = self.down(features)
        gnn_features = self.gnn(graph, down_features)
        up_features = self.up(gnn_features)
        return up_features


class EWC:
    def __init__(self, model, dataloader, tokenizer, modal_weights=None):
        self.model = model
        self.dataloader = dataloader
        self.tokenizer = tokenizer

        # 改为按参数名称分组（解决维度不匹配问题）
        self.modal_groups = {
            "visual": [n for n, p in model.named_parameters() if "visual_encoder" in n and p.requires_grad],
            "text": [n for n, p in model.named_parameters() if "text_encoder" in n and p.requires_grad],
            "cross": [n for n, p in model.named_parameters() if "cross_attention" in n and p.requires_grad],
            "fusion": [n for n, p in model.named_parameters() if "fusion" in n and p.requires_grad],
            "gcn": [n for n, p in model.named_parameters() if "gcn" in n and p.requires_grad]
        }

        self.modal_weights = modal_weights or {
            "visual": 0.3,
            "text": 0.3,
            "cross": 1.0,
            "fusion": 0.8,
            "gcn": 0.8
        }

        # 存储参数名称与初始值的映射（避免维度变化问题）
        self._means = {name: p.data.clone() for name, p in model.named_parameters() if p.requires_grad}
        self._precision_matrices = self._diag_fisher()

    def _diag_fisher(self):
        """基于参数名称的分模态Fisher计算（修复维度错误）"""
        precision_matrices = {name: torch.zeros_like(p) for name, p in self.model.named_parameters() if p.requires_grad}

        self.model.eval()
        for texts, images, labels in self.dataloader:
            # 保持原有数据处理流程
            text_inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
            labels = labels.to(torch.long).to(device)
            src, dst = create_sparse_graph_based_on_similarity(texts, images, self.model, self.dataloader.dataset)
            graph = dgl.graph((src, dst), num_nodes=len(texts))
            graph = dgl.add_self_loop(graph).to(device)

            # 前向传播
            self.model.zero_grad()
            outputs = self.model(text_inputs.to(device), images.to(device), graph)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()

            # 按参数名称匹配模态组（关键修复点）
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    # 查找参数所属模态
                    matched = False
                    for modal in self.modal_groups:
                        if name in self.modal_groups[modal]:
                            precision_matrices[name] += (param.grad  ** 2) * self.modal_weights[modal]
                            matched = True
                            break
                    if not matched:  # 未匹配到任何模态组
                        precision_matrices[name] += param.grad  ** 2 * 0.5

        # 平均处理
        for name in precision_matrices:
            precision_matrices[name] /= len(self.dataloader)

        return precision_matrices

    def penalty(self, current_model):
        """基于参数名称的惩罚项计算"""
        loss = 0
        for name, param in current_model.named_parameters():
            if name in self._precision_matrices:
                loss += (self._precision_matrices[name] * (param - self._means[name])  ** 2).sum()
        return loss


def evaluate(model, dataloader):
    model.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for texts, images, labels in dataloader:
            text_inputs = dataloader.dataset.preprocess_texts(texts)
            labels = labels.to(torch.long).to(device)
            src, dst = create_sparse_graph_based_on_similarity(texts, images, model, dataloader.dataset)
            graph = dgl.graph((src, dst), num_nodes=len(texts))
            graph = dgl.add_self_loop(graph).to(device)
            text_features = model.text_encoder(**text_inputs.to(device)).last_hidden_state
            text_features = text_features[:, 0, :]
            visual_features = model.visual_encoder(pixel_values=images.to(device)).last_hidden_state[:, 0, :]
            combined_features = torch.cat([text_features, visual_features], dim=1)
            adapter_outputs = adapter_gnn(graph, combined_features)
            outputs = model.gcn(graph, adapter_outputs)
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    accuracy = total_correct / total_samples
    return accuracy


def create_sparse_graph_based_on_similarity(texts, images, model, dataset, threshold=0.4, max_neighbors=5):
    text_inputs = dataset.preprocess_texts(texts)
    with torch.no_grad():
        text_features = model.text_encoder(**text_inputs.to(device)).last_hidden_state
        visual_features = model.visual_encoder(pixel_values=images.to(device)).last_hidden_state

        # 使用交叉注意力机制融合特征
        #attended_text = model.cross_attention(text_features, visual_features, visual_features)
        #attended_visual = model.cross_attention(visual_features, text_features, text_features)

        # 提取CLS标记特征
        attended_text_cls = text_features[:, 0, :]
        attended_visual_cls = visual_features[:, 0, :]

        # 融合特征
        fused_features = torch.cat([attended_text_cls, attended_visual_cls], dim=1)

    # 转换为numpy数组以便使用FAISS
    fused_features_np = fused_features.cpu().numpy().astype('float32')

    # 使用FAISS进行精确相似性搜索
    index = faiss.IndexFlatIP(fused_features_np.shape[1])  # 内积相似度
    faiss.normalize_L2(fused_features_np)  # 归一化以匹配余弦相似度
    index.add(fused_features_np)

    # 搜索每个节点的最近邻
    src, dst = [], []
    for i in range(len(fused_features_np)):
        # 搜索最近邻
        distances, indices = index.search(fused_features_np[i].reshape(1, -1), max_neighbors + 1)
        # 跳过自身（如果存在）
        valid_indices = indices[0][1:]
        valid_distances = distances[0][1:]
        for j, dist in zip(valid_indices, valid_distances):
            if dist > threshold:
                src.append(i)
                dst.append(j)
    return src, dst


def log_experiment_results(epoch, loss, accuracy, filepath='experiment_results.csv'):
    with open(filepath, mode='a') as file:
        writer = csv.writer(file)
        writer.writerow([epoch, loss.item(), accuracy])


# Main program
train_label_file = 'test/word/pet_file_list_train.txt'
test_label_file = 'test/word/pet_file_list_test.txt'
default_label = 0

train_dataset = CustomDataset(train_label_file, default_label)
test_dataset = CustomDataset(test_label_file, default_label)

train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

model = GraphAdapter(num_classes=37).to(device)
adapter_gnn = AdapterGNN(768 + 768, 512).to(device)

freeze_parameters(model)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
ewc = EWC(model, train_dataloader, train_dataset.tokenizer)

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (texts, images, labels) in enumerate(train_dataloader):
        optimizer.zero_grad()
        text_inputs = train_dataset.preprocess_texts(texts)
        labels = labels.to(torch.long).to(device)
        src, dst = create_sparse_graph_based_on_similarity(texts, images, model, train_dataset)
        graph = dgl.graph((src, dst), num_nodes=len(texts))
        graph = dgl.add_self_loop(graph).to(device)
        text_features = model.text_encoder(**text_inputs.to(device)).last_hidden_state
        text_features = text_features[:, 0, :]
        visual_features = model.visual_encoder(pixel_values=images.to(device)).last_hidden_state[:, 0, :]
        combined_features = torch.cat([text_features, visual_features], dim=1)
        adapter_outputs = adapter_gnn(graph, combined_features)
        outputs = model.gcn(graph, adapter_outputs)
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
