# demo.py
from graphgene import CaptionGenerator

# 初始化生成器
generator = CaptionGenerator(
    adapter_path="blip_graph_adapter0.pth",
    model_path="/root/autodl-tmp/blip-model"
)

# 生成描述
caption = generator.generate("/root/autodl-tmp/PIXIU/data/0.05fooddata/apple_pie_11613.jpg", 
                           max_length=20,
                           threshold=0.6)

print(f"Generated Caption: {caption}")
