import os
import random
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

# Load BLIP model and processor
local_model_path = "/root/autodl-tmp/blip-model"
processor = BlipProcessor.from_pretrained(local_model_path)
model = BlipForConditionalGeneration.from_pretrained(local_model_path)

# Define dataset path and output files
dataset_path = '/root/autodl-tmp/PIXIU/data/Oxford_pets'
output_train_file = "test/word/pet_file_list_train.txt"
output_test_file = "test/word/pet_file_list_test.txt"


os.makedirs(os.path.dirname(output_train_file), exist_ok=True)
os.makedirs(os.path.dirname(output_test_file), exist_ok=True)


image_files = [f for f in os.listdir(dataset_path) if f.endswith(('.png', '.jpg', '.jpeg'))]


random.shuffle(image_files)

# Function: Extract category (including full model type) from filename
def get_category(filename):
    parts = filename.split('_')
    return f"{parts[0]}_{parts[1]}_{parts[2]}"

# Assign a numeric label to each category
categories = sorted(set(get_category(f) for f in image_files))
category_to_label = {category: i for i, category in enumerate(categories)}

# Split dataset into 80% training and 20% test
split_index = int(0.8 * len(image_files))
train_files = image_files[:split_index]
test_files = image_files[split_index:]


def generate_descriptions(image_files, output_file):
    with open(output_file, 'w') as f:
        for image_file in image_files:
            image_path = os.path.join(dataset_path, image_file)
            image = Image.open(image_path)


            category = get_category(image_file)
            label = category_to_label[category]


            inputs = processor(images=image, return_tensors="pt")
            outputs = model.generate(**inputs)
            caption = processor.decode(outputs[0], skip_special_tokens=True)

            filtered_caption = ' '.join([word for word in caption.split() if category.lower() not in word.lower()])

            f.write(f"{image_path};{filtered_caption};{label}\n")

            print(f"Processed {image_file}")

generate_descriptions(train_files, output_train_file)

generate_descriptions(test_files, output_test_file)

print("All descriptions have been generated and saved to train_descriptions.txt and test_descriptions.txt")
