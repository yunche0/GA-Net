import os
import json
import re
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm  # 在文件顶部加入这行

# -------------------- 图结构配置 --------------------
graph_rank = 8  # 图隐层维度
num_relations = 4  # 边类型数量
use_gating = True  # 使用门控机制

# -------------------- 配置参数 --------------------
local_model_path = "/root/autodl-tmp/blip-model"
dataset_path = "data/flickr8k/images/"
annotation_file = "data/flickr8k/dataset_flickr8k.json"
device = "cuda" if torch.cuda.is_available() else "cpu"
lora_rank = 8
batch_size = 8
num_epochs = 1
learning_rate = 1e-4

# -------------------- 数据处理系统 --------------------
class FlickrDataset(Dataset):
    """全功能的Flickr8k数据集加载器"""

    def __init__(self, samples, processor, split='train'):
        self.samples = samples
        self.processor = processor
        self.split = split

        self.transform = Compose([
            Resize((384, 384)),  # BLIP原生尺寸
            ToTensor(),
            Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                      std=[0.26862954, 0.26130258, 0.27577711])
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # 图像处理
        img_path = os.path.join(dataset_path, sample["filename"])
        image = Image.open(img_path).convert('RGB')
        pixel_values = self.transform(image)

        # 文本处理
        captions = [s["raw"] for s in sample["sentences"]]
        caption = np.random.choice(captions)

        text_inputs = self.processor(
            text=caption,
            padding="max_length",
            max_length=32,
            truncation=True,
            return_tensors="pt"
        )

        return {
            "pixel_values": pixel_values,
            "input_ids": text_inputs["input_ids"].squeeze(),
            "attention_mask": text_inputs["attention_mask"].squeeze(),
            "labels": text_inputs["input_ids"].squeeze()
        }


# -------------------- 图神经网络模块修正 --------------------
class GraphRelationLayer(nn.Module):
    """修正后的多关系图注意力层"""

    def __init__(self, in_dim, out_dim, num_relations):
        super().__init__()
        self.num_relations = num_relations
        self.out_dim = out_dim

        # 关系特定的转换矩阵
        self.relation_proj = nn.ModuleList([
            nn.Linear(in_dim, out_dim) for _ in range(num_relations)
        ])

        # 注意力机制
        self.attn_proj = nn.Linear(2 * out_dim, 1)

        # 门控机制
        if use_gating:
            self.gate = nn.Linear(out_dim, 1)

    def forward(self, node_features, adj_matrix):
        """
        修正维度处理:
        node_features: [batch_size, num_nodes, in_dim]
        adj_matrix: [batch_size, num_relations, num_nodes, num_nodes]
        """
        batch_size = node_features.size(0)
        num_nodes = node_features.size(1)

        # 多关系投影
        relation_features = []
        for proj in self.relation_proj:
            # 每个关系独立映射 [batch, nodes, out_dim]
            rel_feat = proj(node_features)
            relation_features.append(rel_feat.unsqueeze(1))  # 添加关系维度

        relation_features = torch.cat(relation_features, dim=1)  # [batch, num_rels, nodes, out_dim]

        # 注意力计算
        attn_scores = []
        for rel in range(self.num_relations):
            # 获取当前关系的特征 [batch, nodes, out_dim]
            curr_rel_feat = relation_features[:, rel]

            # 计算特征对 [batch, nodes, nodes, 2*out_dim]
            h_i = curr_rel_feat.unsqueeze(2)  # [batch, nodes, 1, out_dim]
            h_j = curr_rel_feat.unsqueeze(1)  # [batch, 1, nodes, out_dim]
            pair_features = torch.cat([h_i.expand(-1, -1, num_nodes, -1),
                                       h_j.expand(-1, num_nodes, -1, -1)], dim=-1)

            # 计算注意力分数 [batch, nodes, nodes]
            score = self.attn_proj(pair_features).squeeze(-1)

            # 应用邻接矩阵mask
            score = score.masked_fill(adj_matrix[:, rel] == 0, -1e9)
            attn_scores.append(F.softmax(score, dim=-1))

        # 多关系聚合
        aggregated = torch.zeros_like(curr_rel_feat)
        for rel in range(self.num_relations):
            # [batch, nodes, nodes] @ [batch, nodes, out_dim] -> [batch, nodes, out_dim]
            agg = torch.bmm(attn_scores[rel], relation_features[:, rel])
            aggregated += agg

        # 门控机制
        if use_gating:
            gate = torch.sigmoid(self.gate(aggregated))
            aggregated = gate * aggregated

        return aggregated


class GraphLoraLayer(nn.Module):
    """修正后的图参数生成器"""

    def __init__(self, in_dim, out_dim, rank=8, num_relations=4):
        super().__init__()
        self.rank = rank
        self.num_nodes = 2  # 输入和输出节点

        # 初始化节点特征 [batch=1, nodes, dim]
        self.node_emb = nn.Parameter(torch.randn(1, 2, graph_rank))

        # 图关系层
        self.graph_layer = GraphRelationLayer(graph_rank, graph_rank, num_relations)

        # 参数生成器
        self.A_generator = nn.Linear(graph_rank, in_dim * rank)
        self.B_generator = nn.Linear(graph_rank, rank * out_dim)

        # 可学习的邻接矩阵 [batch=1, relations, nodes, nodes]
        self.adj_matrix = nn.Parameter(
            torch.eye(self.num_nodes).unsqueeze(0).unsqueeze(0).repeat(1, num_relations, 1, 1)
        )

    def forward(self):
        # 图信息传递 [1, nodes, dim]
        node_features = self.graph_layer(self.node_emb, self.adj_matrix)

        # 生成动态参数
        A = self.A_generator(node_features[:, 0]).view(self.rank, -1)
        B = self.B_generator(node_features[:, 1]).view(-1, self.rank)

        return A, B


# -------------------- 基于图的LoRA模块 --------------------
class GraphLoraInjectedLinear(nn.Module):
    """基于图结构参数的LoRA层"""

    def __init__(self, base_layer, rank=8):
        super().__init__()
        self.base_layer = base_layer
        self.graph_lora = GraphLoraLayer(
            base_layer.in_features,
            base_layer.out_features,
            rank=rank
        )

        # 冻结原始参数
        for param in base_layer.parameters():
            param.requires_grad = False

    def forward(self, x):
        base_output = self.base_layer(x)
        A, B = self.graph_lora()
        lora_output = (x @ A.T) @ B.T
        return base_output + 0.1 * lora_output  # 缩放防止干扰初始权重


# -------------------- 模型改造系统 --------------------
def apply_graph_lora(model):
    """应用图结构LoRA到模型"""
    # 全局冻结
    for param in model.parameters():
        param.requires_grad = False

    # 注入图结构层
    for name, module in model.text_decoder.named_modules():
        if isinstance(module, nn.Linear):
            if re.search(r'self\.(query|value)$', name) or re.search(r'crossattention\.(query|value)$', name):
                parts = name.split('.')
                parent = model.text_decoder
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                original_layer = getattr(parent, parts[-1])
                setattr(parent, parts[-1], GraphLoraInjectedLinear(original_layer, lora_rank))

    # 激活图结构参数
    for name, param in model.named_parameters():
        if 'graph_lora' in name:
            param.requires_grad = True

    return model


# -------------------- 主执行流程改造 --------------------
if __name__ == "__main__":
    # 初始化模型
    processor = BlipProcessor.from_pretrained(local_model_path)
    model = BlipForConditionalGeneration.from_pretrained(local_model_path)

    # 应用图结构LoRA改造
    model = apply_graph_lora(model)
    model.to(device)

    # ...（其余部分与原始代码相同）...

    # 参数状态验证
    #validate_parameters(model)

    # 加载数据集
    with open(annotation_file) as f:
        data = json.load(f)

    # 划分数据集
    all_images = data["images"]
    np.random.shuffle(all_images)
    split_idx = int(0.8 * len(all_images))

    train_dataset = FlickrDataset(all_images[:split_idx], processor)
    test_dataset = FlickrDataset(all_images[split_idx:], processor)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # 训练系统初始化
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

    # 在训练循环中使用 tqdm 包裹数据加载器
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        # 使用 tqdm 包裹训练数据加载器，显示训练进度
        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch + 1} Training")
        for batch in train_loader_tqdm:
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(
                pixel_values=batch["pixel_values"],
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"]
            )

            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            train_loader_tqdm.set_postfix({"loss": loss.item()})  # 更新进度条显示当前 batch 损失

        avg_train_loss = total_loss / len(train_loader)

        # 验证阶段
        model.eval()
        test_loss = 0
        # 使用 tqdm 包裹验证数据加载器，显示验证进度
        test_loader_tqdm = tqdm(test_loader, desc=f"Epoch {epoch + 1} Validation")
        with torch.no_grad():
            for batch in test_loader_tqdm:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                test_loss += outputs.loss.item()
                test_loader_tqdm.set_postfix({"loss": outputs.loss.item()})  # 更新进度条显示当前 batch 损失

        avg_test_loss = test_loss / len(test_loader)

        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print(f"训练损失: {avg_train_loss:.4f} | 测试损失: {avg_test_loss:.4f}")

    # 保存LoRA权重
    torch.save(model.state_dict(), "blip_graph_weights.pth")
    print("训练完成，权重已保存")
