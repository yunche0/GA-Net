import os
import random


# Define dataset path and output files
dataset_path = '/root/autodl-tmp/PIXIU/data/Oxford_pets'
output_train_file = "test/word/PET_file_list_train.txt"
output_test_file = "test/word/PET_file_list_test.txt"


os.makedirs(os.path.dirname(output_train_file), exist_ok=True)
os.makedirs(os.path.dirname(output_test_file), exist_ok=True)

image_files = [f for f in os.listdir(dataset_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

random.shuffle(image_files)

# Function: Extract category from filename
def get_category(filename):
    return filename.split('_')[0]

# Assign a numeric label to each category
categories = sorted(set(get_category(f) for f in image_files))
category_to_label = {category: i for i, category in enumerate(categories)}

# Split dataset into 80% training and 20% test
split_index = int(0.8 * len(image_files))
train_files = image_files[:split_index]
test_files = image_files[split_index:]

# Function: Replace descriptions with "xxxx"
def replace_descriptions(image_files, output_file):
    with open(output_file, 'w') as f:
        for image_file in image_files:
            image_path = os.path.join(dataset_path, image_file)
            category = get_category(image_file)
            label = category_to_label[category]
            # Use fixed description "xxxx"
            fixed_description = "xxxx"
            f.write(f"{image_path};{fixed_description};{label}\n")
            print(f"Processed {image_file}")


replace_descriptions(train_files, output_train_file)

replace_descriptions(test_files, output_test_file)

print("All descriptions have been replaced and saved to the specified files.")
