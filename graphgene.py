# graphgene.py - 纯推理专用文件

import os
import re
import torch
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch.nn as nn
import torch.nn.functional as F

# -------------------- 图结构配置 --------------------
graph_rank = 8
num_relations = 4
use_gating = True


# -------------------- 模型定义模块 --------------------
class GraphRelationLayer(nn.Module):
    """推理专用图关系层（移除非必要组件）"""

    def __init__(self, in_dim, out_dim, num_relations):
        super().__init__()
        self.num_relations = num_relations
        self.relation_proj = nn.ModuleList([
            nn.Linear(in_dim, out_dim) for _ in range(num_relations)
        ])
        self.attn_proj = nn.Linear(2 * out_dim, 1)
        if use_gating:
            self.gate = nn.Linear(out_dim, 1)

    def forward(self, node_features, adj_matrix):
        batch_size, num_nodes = node_features.shape[:2]
        relation_features = torch.stack([proj(node_features) for proj in self.relation_proj], dim=1)

        attn_scores = []
        for rel in range(self.num_relations):
            curr_feat = relation_features[:, rel]
            pair_features = torch.cat([
                curr_feat.unsqueeze(2).expand(-1, -1, num_nodes, -1),
                curr_feat.unsqueeze(1).expand(-1, num_nodes, -1, -1)
            ], dim=-1)
            score = self.attn_proj(pair_features).squeeze(-1)
            score = score.masked_fill(adj_matrix[:, rel] == 0, -1e9)
            attn_scores.append(F.softmax(score, dim=-1))

        aggregated = torch.zeros_like(curr_feat)
        for rel in range(self.num_relations):
            aggregated += torch.bmm(attn_scores[rel], relation_features[:, rel])

        if use_gating:
            aggregated = torch.sigmoid(self.gate(aggregated)) * aggregated

        return aggregated


class GraphLoraLayer(nn.Module):
    """精简版图参数生成器"""

    def __init__(self, in_dim, out_dim, rank=8):
        super().__init__()
        self.rank = rank
        self.node_emb = nn.Parameter(torch.randn(1, 2, graph_rank))  # 固定2个节点
        self.graph_layer = GraphRelationLayer(graph_rank, graph_rank, num_relations)
        self.adj_matrix = nn.Parameter(
            torch.eye(2).unsqueeze(0).unsqueeze(0).repeat(1, num_relations, 1, 1)
        )
        self.A_generator = nn.Linear(graph_rank, in_dim * rank)
        self.B_generator = nn.Linear(graph_rank, rank * out_dim)

    def forward(self):
        nodes = self.graph_layer(self.node_emb, self.adj_matrix)
        return (
            self.A_generator(nodes[:, 0]).view(self.rank, -1),
            self.B_generator(nodes[:, 1]).view(-1, self.rank)
        )


class GraphLoraInjectedLinear(nn.Module):
    """推理专用参数注入层"""

    def __init__(self, base_layer):
        super().__init__()
        self.base_layer = base_layer
        self.graph_lora = GraphLoraLayer(
            base_layer.in_features,
            base_layer.out_features,
            rank=8
        )

    def forward(self, x):
        base = self.base_layer(x)
        A, B = self.graph_lora()
        return base + 0.1 * ((x @ A.T) @ B.T)


# -------------------- 模型加载工具 --------------------
def load_inference_model(weight_path, device='cuda'):
    """加载推理专用模型"""
    # 初始化基础模型
    processor = BlipProcessor.from_pretrained("/root/autodl-tmp/blip-model")
    model = BlipForConditionalGeneration.from_pretrained("/root/autodl-tmp/blip-model")

    # 应用图结构改造
    for name, module in model.text_decoder.named_modules():
        if isinstance(module, nn.Linear) and re.search(r'(self|crossattention)\.(query|value)$', name):
            parts = name.split('.')
            parent = model.text_decoder
            for part in parts[:-1]:
                parent = getattr(parent, part)
            original = getattr(parent, parts[-1])
            setattr(parent, parts[-1], GraphLoraInjectedLinear(original))

    # 加载训练权重
    model.load_state_dict(torch.load(weight_path, map_location=device))
    return processor, model.to(device).eval()


# -------------------- 预处理流水线 --------------------
def get_transform():
    """获取与训练一致的预处理"""
    return Compose([
        Resize((384, 384)),
        ToTensor(),
        Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                  std=[0.26862954, 0.26130258, 0.27577711])
    ])


# -------------------- 生成接口 --------------------
def generate_description(image_path, weight_path="blip_graph_weights.pth", device='cuda'):
    """端到端生成接口"""
    # 输入验证
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file {image_path} not found")
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"Model weights {weight_path} not found")

    # 初始化组件
    processor, model = load_inference_model(weight_path, device)
    transform = get_transform()

    # 处理图像
    image = Image.open(image_path).convert('RGB')
    pixel_values = transform(image).unsqueeze(0).to(device)

    # 生成文本
    with torch.no_grad():
        outputs = model.generate(
            pixel_values=pixel_values,
            max_length=50,
            num_beams=4,
            early_stopping=True
        )

    return processor.decode(outputs[0], skip_special_tokens=True)


# -------------------- 命令行接口 --------------------
if __name__ == "__main__":
    # 直接在代码中设置参数（按需修改以下路径）
    image_path = "/root/autodl-tmp/PIXIU/data/0.05fooddata/apple_pie_44873.jpg"  # 替换为实际图像路径
    weights_path = "blip_graph_weights.pth"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        caption = generate_description(image_path, weights_path, device)
        print("\n生成结果:")
        print("=" * 50)
        print(caption)
        print("=" * 50)
    except Exception as e:
        print(f"生成失败: {str(e)}")
